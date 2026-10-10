#!/usr/bin/env python3
"""Point the FCB camera at a person and keep it there, on a 360-servo head.

    FCB video -> YOLO person detection + ByteTrack IDs -> pick one person
      -> pixel error from frame centre -> PD speed per axis
      -> Arduino (pantilt_servo.ino) -> pan servo, tilt servo

The servos are continuous-rotation ones, so a command is a speed, not an
angle. That suits tracking: the error in the picture becomes how fast to
turn, and the head stops once the person is inside a small box around the
centre. With no position feedback there is also no "home" -- if the head
ends up pointing somewhere odd, pause (space) and jog it back with WASD.

Target choice: the largest person in view is locked onto and followed by
tracker ID. Click someone to switch to them, press r to drop the lock. If
the target disappears the head carries on briefly (they may only have been
missed for a frame) and then stops; after --reacquire seconds it takes
whoever is nearest to where they were last seen.

Kill switch: it starts KILLED -- servos off -- so you can check the feed
and the target before anything moves. Press g to arm.
    k, or click the red KILL button    stop the servos at once, latched
    g                                  arm again
    Ctrl+C, q, Esc                     quit (servos stop)
    a button from Arduino D2 to GND    hardware kill, works even if this
                                       program hangs; press again to re-arm

Zoom (the FCB's optical zoom, over VISCA):
    + / -   zoom in / out while held (also the on-screen + and - buttons,
            and the mouse wheel)
    0       zoom all the way out
    Tracking gain is divided by the magnification, since at 10x the same
    servo speed sweeps the picture ten times as fast.

Keys (window mode):
    f       full screen on / off (starts full screen; --windowed not)
    space   pause / resume tracking (servos stop when paused)
    w a s d jog the head by hand while paused
    r       release the target and pick again
    click   track the person clicked on
    q, Esc  quit

First run, with the servos connected but nothing to track:
    ./fcb_track.py --servo-test       turns each servo each way in turn,
                                      so you can check direction and wiring
then:
    ./fcb_track.py                    camera and servo board autodetected
    ./fcb_track.py --dry-run          tracking only, no servo board
    ./fcb_track.py --video clip.mp4 --dry-run     replay a recording
"""
import argparse
import math
import logging
import os
import sys
import threading
import time

import cv2
import numpy as np

from fcb_base_driver import devices
from fcb_base_driver.frame_grabber import FrameGrabber
from fcb_base_driver.pantilt import (Axis, DryRunPanTilt, PanTilt,
                                     PanTiltError, PositionAxis, find_port)
from fcb_base_driver.zoom_drive import ZoomDrive

log = logging.getLogger("fcb_track")

PERSON = 0          # COCO class id
WINDOW = "fcb_track"
JOG_SPEED = 0.4
JOG_HOLD = 0.25     # s a jog key keeps the head moving (covers key repeat)
#: A zoom key keeps the lens moving this long. The first press has to
#: outlast the keyboard's auto-repeat delay (~0.5 s), or holding the key
#: would zoom, stop, then zoom again; repeats after that only need to
#: bridge the gap between two of them.
ZOOM_KEY_FIRST = 0.55
ZOOM_KEY_REPEAT = 0.12
ZOOM_WHEEL_STEP = 0.15  # s of zoom per mouse-wheel notch
#: Stop the servos if detection produces nothing for this long: the last
#: speed would otherwise carry on for as long as the detector is stuck.
DETECTION_STALE = 0.5


# -- video sources ---------------------------------------------------------

class FileSource:
    """A recording played back at its own frame rate, same API as FrameGrabber.

    FrameGrabber reads as fast as the source delivers, which for a file is
    the whole clip in seconds; this paces it like a live camera instead.
    """

    def __init__(self, path):
        self._capture = cv2.VideoCapture(path)
        if not self._capture.isOpened():
            raise RuntimeError(f"cannot open {path}")
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        self._period = 1.0 / fps if 1.0 < fps < 1000.0 else 1.0 / 30
        self._seq = 0
        self._next = time.monotonic()
        self.ended = False

    def wait_for_frame(self, after_seq, timeout=1.0):
        delay = self._next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        # Skip frames we are too slow to process, as a live camera would.
        while time.monotonic() - self._next > self._period:
            self._capture.grab()
            self._next += self._period
        ok, frame = self._capture.read()
        self._next += self._period
        if not ok:
            self.ended = True
            return None, after_seq, None
        self._seq += 1
        return frame, self._seq, time.monotonic()

    def fps(self):
        return 1.0 / self._period

    def stop(self):
        self._capture.release()


def open_camera(args):
    """The FCB through its board, set up the way fcb_record.py sets it up."""
    device = args.video or devices.autodetect_video(
        log=lambda msg: log.info("video: %s", msg))
    if device is None:
        log.critical("no FCB camera found")
        return None
    capture = devices.open_capture(device, args.width, args.height,
                                   args.fourcc)
    if capture is None:
        log.critical("could not open %s -- is fcb_record.py (or the neohd "
                     "recorder in tmux) holding the camera?", device)
        return None
    pixel_format = devices.fourcc_of(capture)
    repeats = devices.usb_id(device) in devices.REPEATING_FRAME_USB_IDS
    log.info("video: %s at %dx%d %s%s", device,
             int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
             pixel_format or "format not reported",
             ", dropping the board's repeated frames" if repeats else "")
    try:
        capture.set(cv2.CAP_PROP_CONVERT_RGB, 1.0)
    except Exception:
        pass
    return FrameGrabber(capture, devices.yuv422_to_bgr_code(pixel_format),
                        drop_repeats=repeats)


# -- control ---------------------------------------------------------------

class PID:
    """PID from normalised image error (-1..1) to servo speed (-1..1).

    P turns towards the person in proportion to how far off-centre they are.
    I builds up while they stay off-centre. A 360 servo with friction or
    a heavy load can sit stalled on a small command -- in the logs pan held
    the same error for 12 s at a steady pulse -- and the integral keeps
    growing until the servo breaks free. D brakes as the person approaches
    the centre, against overshoot; it acts on the filtered rate of change,
    since detections jitter frame to frame.

    Anti-windup: the integral is clamped to +-`i_limit` of output, it does
    not grow while the output is already saturated in that direction, and
    it is cleared once the person is centred, so a stored push from the
    last move cannot carry the head past them.

    Stopping: the axis stops once the error is inside the stop zone -- the
    crosshair inside the box drawn round the person's centre, passed in per
    update as `zone`, or `deadband` when none is given. It only starts
    again once the error is outside 1.25x that, or a person standing right
    on the edge would make it twitch on and off.

    Stopping early (`lead`, seconds): a 360 servo coasts after its pulses
    stop, and a detection is a frame or two old by the time it acts, so
    stopping only once the crosshair is inside the box lands it past the
    person. While closing in, the stop test uses where the error will be
    `lead` seconds on, at the current rate of approach.

    Smoothing (`smooth`, seconds): the person's box wobbles a few pixels
    every frame, and at 60 updates a second that wobble went straight into
    the servo speed -- the D term, which works on the rate of change, most
    of all -- and made the head jerky. The error is low-pass filtered with
    this time constant before P, I and D see it. 0 turns it off.

    Speed changes are rate-limited so the camera does not jerk: speeding up
    at `accel` (speed units per second), slowing down while still moving at
    2.5x that. Coming to a stop is not limited at all, since overshooting
    the person is worse than a hard stop.
    """

    def __init__(self, kp, ki, kd, deadband, max_speed, accel, i_limit=0.5,
                 lead=0.0, smooth=0.0):
        self.lead, self.smooth = lead, smooth
        self.kp, self.ki, self.kd = kp, ki, kd
        self.deadband, self.max_speed, self.accel = deadband, max_speed, accel
        self.i_limit = i_limit
        self.reset()

    def reset(self):
        self.output = 0.0
        self.terms = (0.0, 0.0, 0.0)   # last (P, I, D), for display
        self._integral = 0.0           # in output units: ki is folded in
        self._prev_error = None
        self._prev_time = None
        self._rate = 0.0
        self._settled = True
        self._smoothed = None

    def update(self, error, now, zone=None):
        dt = (now - self._prev_time) if self._prev_time is not None else None
        if self._smoothed is None or not self.smooth or not dt or dt >= 0.5:
            self._smoothed = error
        else:
            self._smoothed += (1 - math.exp(-dt / self.smooth)) * \
                (error - self._smoothed)
        error = self._smoothed
        if self._prev_error is not None and dt and 0 < dt < 0.5:
            raw = (error - self._prev_error) / dt
            self._rate += 0.4 * (raw - self._rate)   # detections are noisy
        self._prev_error, self._prev_time = error, now

        zone = self.deadband if zone is None else zone
        threshold = zone * (1.25 if self._settled else 1.0)
        judged = error
        if self.lead and error * self._rate < 0:     # closing in
            judged = error + self._rate * self.lead
            if judged * error < 0:                   # would cross centre
                judged = 0.0
        if abs(judged) < threshold:
            self._settled = True
            self._integral = 0.0
            self.terms = (0.0, 0.0, 0.0)
            wanted = 0.0
        else:
            self._settled = False
            p = self.kp * error
            d = self.kd * self._rate
            if dt and 0 < dt < 0.5:
                step = self.ki * error * dt
                pushing_on = (self.output >= self.max_speed and step > 0) or \
                    (self.output <= -self.max_speed and step < 0)
                if not pushing_on:
                    self._integral = max(-self.i_limit,
                                         min(self.i_limit, self._integral + step))
            # A change of side means the person is now the other way: what
            # was stored pushes in the wrong direction, so drop it.
            if self._integral * error < 0:
                self._integral = 0.0
            self.terms = (p, self._integral, d)
            wanted = p + self._integral + d
            wanted = max(-self.max_speed, min(self.max_speed, wanted))

        if wanted != 0.0 and wanted != self.output:
            # The first update after a reset has no dt; it still must not
            # jump straight to full speed, so it gets one nominal frame.
            speeding_up = (abs(wanted) > abs(self.output)
                           or wanted * self.output < 0)
            rate = self.accel if speeding_up else 2.5 * self.accel
            step = rate * min(dt or 1 / 30, 0.1)
            wanted = max(self.output - step, min(self.output + step, wanted))
        self.output = wanted
        return wanted


class EgoMotion:
    """How fast the camera is actually turning, measured from the picture.

    The servos have no encoder, but when the head pans the whole scene
    slides across the frame. Phase correlation between consecutive frames,
    on a small greyscale copy (~1 ms), gives that slide; divided by the
    frame interval it is the camera's turning rate, in half-frames per
    second -- the same units as the tracking error, so +pan is the camera
    turning right (the scene sliding left) and +tilt it tilting up.

    A walking person moves too, but the background is most of the picture
    and dominates the correlation. A weak correlation peak (a blank wall,
    motion blur) is reported as no measurement rather than as zero speed.
    """

    SIZE = (320, 180)
    MIN_RESPONSE = 0.04
    MIN_TEXTURE = 4.0     # grey-level spread below which there is nothing to track

    def __init__(self):
        self._prev = None
        self._prev_at = None
        self._window = cv2.createHanningWindow(self.SIZE, cv2.CV_32F)
        self.pan_rate = self.tilt_rate = 0.0
        self.ok = False

    def update(self, frame, stamp):
        small = cv2.cvtColor(cv2.resize(frame, self.SIZE,
                                        interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY).astype(np.float32)
        prev, prev_at = self._prev, self._prev_at
        self._prev, self._prev_at = small, stamp
        self.ok = False
        if prev is None or stamp is None or prev_at is None:
            return
        dt = stamp - prev_at
        if not 0 < dt < 0.2:
            return
        # A featureless picture (a blank wall, the lens cap) correlates
        # strongly with itself at zero shift -- a confident, false "not
        # moving" that would make the speed loop shove a moving servo.
        if small.std() < self.MIN_TEXTURE:
            return
        (dx, dy), response = cv2.phaseCorrelate(prev, small, self._window)
        if response < self.MIN_RESPONSE:
            return
        half_w, half_h = self.SIZE[0] / 2, self.SIZE[1] / 2
        self.pan_rate = -dx / half_w / dt
        self.tilt_rate = dy / half_h / dt
        self.ok = True


class SpeedPID:
    """Inner loop: make the head actually turn at the speed asked for.

    The position PID decides how fast the camera should turn. This compares
    that with how fast it *is* turning (EgoMotion) and corrects the servo
    command: a servo that is told to move but is stuck -- friction, a heavy
    load, a cable -- gets pushed harder until it breaks free, and one going
    faster than wanted is eased off.

    The target speed for a command is learnt while running (`full_rate`:
    camera speed at full command, half-frames per second at 1x zoom), from
    moments when the servo is clearly moving, so it needs no calibration
    and tracks voltage changes. A stuck moment is never learnt from, or a
    stall would teach it that the servo is slow and stop it pushing.

    Bounds: the correction is at most +-`max_correction`, the result never
    turns the servo the other way from the position PID's request, and a
    stop request is passed straight through -- stopping must stop.
    """

    #: Fastest the correction may change, per second. The measured speed is
    #: noisy frame to frame, and passing that straight to the servo at 60 Hz
    #: made the head jerky (2026-10-06); this lets a correction build up over
    #: a fraction of a second instead.
    CORRECTION_SLEW = 1.5

    def __init__(self, kp, ki, full_rate, max_correction=0.4, tau=0.15):
        self.kp, self.ki = kp, ki
        self.full_rate = full_rate
        self.max_correction, self.tau = max_correction, tau
        self.measured = None
        self.reset()

    def reset(self):
        self._integral = 0.0
        self.target = 0.0
        self.correction = 0.0

    def measure(self, rate, ok, dt):
        if not ok or not dt:
            return
        if self.measured is None:
            self.measured = rate
        else:
            self.measured += (1 - math.exp(-dt / self.tau)) * (rate - self.measured)

    def update(self, command, zoom_ratio, dt, measuring):
        if command == 0.0:
            self.reset()
            return 0.0
        scale = self.full_rate * zoom_ratio
        self.target = command * scale
        if self.measured is None or not measuring or not dt:
            # Hold the correction through a dropped measurement. Snapping it
            # to zero for one frame and back the next was a jerk of its own.
            out = max(-1.0, min(1.0, command + self.correction))
            return 0.0 if out * command < 0 else out
        error = (self.target - self.measured) / scale     # in command units
        self._integral = max(-self.max_correction, min(
            self.max_correction, self._integral + self.ki * error * dt))
        wanted = max(-self.max_correction, min(
            self.max_correction, self.kp * error + self._integral))
        step = self.CORRECTION_SLEW * min(dt, 0.1)
        self.correction = max(self.correction - step,
                              min(self.correction + step, wanted))
        out = max(-1.0, min(1.0, command + self.correction))
        if out * command < 0:
            out = 0.0
        # Learn how fast the head really turns, from clearly moving moments.
        if abs(out) > 0.25 and self.measured * out > 0:
            seen = abs(self.measured) / zoom_ratio / abs(out)
            if seen > 0.2 * self.full_rate:              # not a stall
                self.full_rate += 0.02 * (seen - self.full_rate)
                self.full_rate = max(0.3, min(8.0, self.full_rate))
        return out


class TargetSelector:
    """Which tracked person to follow, by ByteTrack ID."""

    #: A new ID this close to where the target was last seen (fraction of
    #: frame width) is taken as the same person after a tracker ID switch.
    SAME_PERSON_DISTANCE = 0.12

    def __init__(self, reacquire):
        self.reacquire = reacquire
        self.locked_id = None
        self.last_box = None
        self.last_seen = None

    @staticmethod
    def _centre(box):
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _nearest(self, people, limit=None):
        cx, cy = self._centre(self.last_box)
        best = min(people, key=lambda p: (self._centre(p[1])[0] - cx) ** 2
                   + (self._centre(p[1])[1] - cy) ** 2)
        if limit is not None:
            bx, by = self._centre(best[1])
            if ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5 > limit:
                return None
        return best

    def _lock(self, person, now, why):
        if person[0] != self.locked_id:
            log.info("target: person %s (%s)", person[0], why)
        self.locked_id, self.last_box, self.last_seen = person[0], person[1], now

    def update(self, people, now, frame_width):
        """`people` is [(id, (x1, y1, x2, y2)), ...]; returns the box or None."""
        for person in people:
            if self.locked_id is not None and person[0] == self.locked_id:
                self.last_box, self.last_seen = person[1], now
                return person[1]
        if not people:
            return None
        if self.last_box is None:
            largest = max(people, key=lambda p: (p[1][2] - p[1][0])
                          * (p[1][3] - p[1][1]))
            self._lock(largest, now, "largest in view")
            return largest[1]
        same = self._nearest(people, self.SAME_PERSON_DISTANCE * frame_width)
        if same is not None:
            self._lock(same, now, "tracker ID changed, same place")
            return same[1]
        if now - self.last_seen > self.reacquire:
            nearest = self._nearest(people)
            self._lock(nearest, now, "nearest to where the last one was lost")
            return nearest[1]
        return None

    def select_at(self, people, x, y, now):
        for person in people:
            x1, y1, x2, y2 = person[1]
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._lock(person, now, "clicked")
                return True
        return False

    def release(self):
        log.info("target released")
        self.locked_id = self.last_box = self.last_seen = None

    def seen_within(self, now, seconds):
        return self.last_seen is not None and now - self.last_seen <= seconds


def detect_people(model, frame, args):
    results = model.track(frame, persist=True, classes=[PERSON],
                          conf=args.conf, imgsz=args.imgsz, verbose=False,
                          device=args.device or None, tracker="bytetrack.yaml")
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    # Unconfirmed tracks carry no ID; give them one that cannot collide.
    ids = (boxes.id.cpu().numpy().astype(int).tolist()
           if boxes.id is not None else [-1 - i for i in range(len(xyxy))])
    return [(ids[i], tuple(float(v) for v in xyxy[i])) for i in range(len(xyxy))]


class Detector:
    """Person detection on its own thread, always on the newest frame.

    Run inline, detection held the whole loop to its own pace -- about
    20 fps at 48 ms a frame on a laptop CPU -- so the picture stuttered as
    well. Here the display runs at the camera's rate and draws the latest
    detections it has; the servos are updated once per detection, which
    is the only rate at which there is anything new to act on. Frames
    that arrive while a detection is running are skipped, not queued.
    """

    def __init__(self, model, args):
        self._model, self._args = model, args
        self._cond = threading.Condition()
        self._frame, self._frame_seq = None, 0
        self._result = (0, [], None)   # (frame seq, people, finished at)
        self.ms = 0.0
        self._done = 0
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame, seq):
        with self._cond:
            self._frame, self._frame_seq = frame, seq
            self._cond.notify()

    def latest(self):
        with self._cond:
            return self._result

    def take_count(self):
        """Detections finished since the last call, for a rate readout."""
        with self._cond:
            done, self._done = self._done, 0
            return done

    def _run(self):
        last = 0
        while self._running:
            with self._cond:
                while self._running and self._frame_seq == last:
                    self._cond.wait(0.5)
                frame, last = self._frame, self._frame_seq
            if not self._running:
                return
            t0 = time.monotonic()
            people = detect_people(self._model, frame, self._args)
            now = time.monotonic()
            self.ms += 0.2 * ((now - t0) * 1000 - self.ms)
            with self._cond:
                self._result = (last, people, now)
                self._done += 1

    def stop(self):
        self._running = False
        with self._cond:
            self._cond.notify()
        self._thread.join(timeout=2.0)


def centre_zone(args):
    """Half-size of the centre box round the person, in (pan, tilt) error units.

    A fixed share of the frame -- `--box-w` of its width by `--box-h` of
    its height -- not scaled by the person's size (the user asked for that
    on 2026-10-06, after a person-sized box let near people sit too far off
    centre). Error is measured in half-frames, so a box `--box-w` of the
    width wide reaches `--box-w` either side of the aim point.
    """
    return args.box_w, args.box_h


def vertical_error(box, aim_y, height):
    """Tilt error for this box: -1 top edge .. +1 bottom, or 0 when unknown.

    A person close enough to fill the frame top to bottom is cut off at
    both edges, and the aim point then sits at the same fraction of the
    frame however the camera tilts -- 2026-10-06 it read -0.40 for 20 s
    straight while tilt kept pushing up. A box touching both edges says
    nothing about which way to tilt, so tilt holds still.
    """
    margin = 0.01 * height
    if box[1] <= margin and box[3] >= height - margin:
        return 0.0
    return (aim_y - height / 2) / (height / 2)


def aim_point(box, aim):
    """Where on the person to centre: `aim` of the way down from the top.

    Not the box centre -- that is the waist, and centring it puts the head
    near the top edge, where a crouch or a step closer cuts it off.
    """
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, y1 + aim * (y2 - y1)


# -- display ---------------------------------------------------------------

KILL_BUTTON = (150, 50)   # w, h at a 960-wide view, top-right corner
RED = (0, 0, 230)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def kill_button_rect(view_width):
    """Button corners in view pixels; it grows with the view, as text does."""
    u = view_width / 960
    bw, bh = KILL_BUTTON
    return (int(view_width - (bw + 10) * u), int(10 * u),
            int(view_width - 10 * u), int((10 + bh) * u))


def zoom_button_rects(view_width):
    """{"in": rect, "out": rect}, side by side under the KILL button."""
    u = view_width / 960
    x1, _, x2, y2 = kill_button_rect(view_width)
    gap, bh = int(10 * u), int(50 * u)
    mid = (x1 + x2) // 2
    top = y2 + gap
    return {"out": (x1, top, mid - gap // 2, top + bh),
            "in": (mid + gap // 2, top, x2, top + bh)}


def in_rect(rect, x, y):
    return rect is not None and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def draw(frame, people, target, aim, args, status, killed, why_killed,
         zoom=None, zooming=0):
    """The annotated view, sized to `args.display_width`.

    Every size is in units of a 960-pixel-wide view (`u`), so text, lines
    and the KILL button keep their proportions full screen at 1920.
    """
    h, w = frame.shape[:2]
    scale = args.display_width / w
    view = (frame.copy() if args.display_width == w
            else cv2.resize(frame, (args.display_width, int(h * scale))))
    vh, vw = view.shape[:2]
    u = vw / 960
    line = max(1, round(u))
    for pid, box in people:
        x1, y1, x2, y2 = (int(v * scale) for v in box)
        colour = (0, 255, 0) if box == target else (160, 160, 160)
        cv2.rectangle(view, (x1, y1), (x2, y2), colour, 2 * line)
        cv2.putText(view, str(pid), (x1 + int(3 * u), y1 + int(18 * u)),
                    FONT, 0.55 * u, colour, 2 * line)
    if target is not None and aim is not None:
        # The person's centre box: green while the crosshair is inside it
        # (servos holding still), orange while they are driving to it.
        zone_x, zone_y = centre_zone(args)
        half_x, half_y = zone_x * w / 2, zone_y * h / 2
        inside = (abs(aim[0] - w / 2) < half_x
                  and abs(vertical_error(target, aim[1], h)) < zone_y)
        cv2.rectangle(view, (int((aim[0] - half_x) * scale),
                             int((aim[1] - half_y) * scale)),
                      (int((aim[0] + half_x) * scale),
                       int((aim[1] + half_y) * scale)),
                      (0, 255, 0) if inside else (0, 165, 255), 2 * line)
    cv2.drawMarker(view, (vw // 2, vh // 2), (0, 200, 255),
                   cv2.MARKER_CROSS, int(24 * u), line)
    if aim is not None:
        ax, ay = int(aim[0] * scale), int(aim[1] * scale)
        cv2.circle(view, (ax, ay), int(6 * u), (0, 255, 0), -1)
        cv2.line(view, (vw // 2, vh // 2), (ax, ay), (0, 255, 0), line)
    x1, y1, x2, y2 = kill_button_rect(vw)
    cv2.rectangle(view, (x1, y1), (x2, y2), RED, -1)
    cv2.rectangle(view, (x1, y1), (x2, y2), (255, 255, 255), 2 * line)
    label = "KILLED" if killed else "KILL (k)"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.8 * u, 2 * line)
    cv2.putText(view, label, (x1 + (x2 - x1 - tw) // 2,
                              y1 + (y2 - y1 + th) // 2),
                FONT, 0.8 * u, (255, 255, 255), 2 * line)
    if zoom is not None:
        rects = zoom_button_rects(vw)
        for name, sign, glyph in (("out", -1, "-"), ("in", 1, "+")):
            bx1, by1, bx2, by2 = rects[name]
            fill = (0, 160, 255) if zooming == sign else (60, 60, 60)
            cv2.rectangle(view, (bx1, by1), (bx2, by2), fill, -1)
            cv2.rectangle(view, (bx1, by1), (bx2, by2), (255, 255, 255),
                          2 * line)
            (tw, th), _ = cv2.getTextSize(glyph, FONT, 1.2 * u, 3 * line)
            cv2.putText(view, glyph, ((bx1 + bx2 - tw) // 2, (by1 + by2 + th) // 2),
                        FONT, 1.2 * u, (255, 255, 255), 3 * line)
        label = f"zoom {zoom.ratio:.1f}x"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.7 * u, 2 * line)
        lx = (rects["out"][0] + rects["in"][2] - tw) // 2
        ly = rects["in"][3] + int(10 * u) + th
        cv2.putText(view, label, (lx, ly), FONT, 0.7 * u, (0, 0, 0), 5 * line)
        cv2.putText(view, label, (lx, ly), FONT, 0.7 * u, (255, 255, 255),
                    2 * line)
    if killed:
        cv2.rectangle(view, (0, 0), (vw - 1, vh - 1), RED, int(8 * u))
        banner = f"SERVOS KILLED ({why_killed})" + \
            ("  -  press g to arm" if why_killed == "software" else
             "  -  press the Arduino button to re-arm")
        (tw, th), _ = cv2.getTextSize(banner, FONT, 0.7 * u, 2 * line)
        cv2.rectangle(view, (0, vh - th - int(24 * u)), (vw, vh), RED, -1)
        cv2.putText(view, banner, ((vw - tw) // 2, vh - int(12 * u)),
                    FONT, 0.7 * u, (255, 255, 255), 2 * line)
    for i, text in enumerate(status):
        org = (int(10 * u), int((24 + 22 * i) * u))
        cv2.putText(view, text, org, FONT, 0.6 * u, (0, 0, 0), 4 * line)
        cv2.putText(view, text, org, FONT, 0.6 * u, (255, 255, 255), line)
    return view, scale


# -- modes -----------------------------------------------------------------

def servo_test(head, speed):
    """Turn each servo each way in turn so direction can be checked by eye."""
    head.arm()    # a 180 tilt is centred here, before it is moved
    steps = [
        ("pan RIGHT  (camera should turn to its right)", speed, 0.0),
        ("pan LEFT", -speed, 0.0),
        ("tilt UP    (camera should tilt up)", 0.0, speed),
        ("tilt DOWN", 0.0, -speed),
    ]
    for label, pan, tilt in steps:
        pulses = head.set_speed(pan, tilt)
        log.info("servo test: %-46s pulses pan=%s tilt=%s us", label, *pulses)
        time.sleep(1.5)
        head.stop()
        time.sleep(0.7)
    log.info("servo test done. A wrong direction: flip it with "
             "--pan-invert / --no-pan-invert or --tilt-invert / "
             "--no-tilt-invert. Not moving at all: raise --pan-deadzone-us / "
             "--tilt-deadzone-us. "
             "Creeping when stopped: the servo needs a signal-loss stop, "
             "see the README")


def build_head(args):
    pan = Axis(args.pan_neutral, args.pan_deadzone_us, args.pan_max_us,
               args.pan_invert, args.pan_left_boost)
    if args.tilt_servo == 180:
        tilt = PositionAxis(args.tilt_low_us, args.tilt_high_us,
                            args.tilt_centre_us, args.tilt_invert,
                            args.tilt_rate_us)
    else:
        tilt = Axis(args.tilt_neutral, args.tilt_deadzone_us, args.tilt_max_us,
                    args.tilt_invert)
    if args.dry_run:
        log.info("servos: dry run, nothing will move")
        return DryRunPanTilt(pan, tilt)
    port = args.servo_port or find_port(
        log=lambda msg: log.info("servos: %s", msg))
    if port is None:
        return None
    try:
        return PanTilt(port, pan, tilt,
                       log=lambda msg: log.info("servos: %s", msg))
    except PanTiltError as exc:
        log.critical("servos: %s", exc)
        return None


def build_zoom(args):
    """The zoom drive, or None -- a camera without VISCA still tracks."""
    if args.no_zoom or (args.video and os.path.isfile(args.video)):
        return None
    try:
        return ZoomDrive(args.visca_port, args.zoom_speed,
                         log=lambda msg: log.info("%s", msg))
    except Exception as exc:
        log.warning("zoom: %s -- running without zoom control", exc)
        return None


def track(args, head):
    try:
        from ultralytics import YOLO
    except ImportError:
        log.critical("ultralytics not installed -- pip3 install ultralytics "
                     "(on the Orin: install JetPack's torch first, see README)")
        return 1
    model = YOLO(args.model)
    # The first inference is several times slower (CUDA/TensorRT setup);
    # done on live video it skips enough frames to lose a moving person.
    log.info("detection: loading %s", args.model)
    model.predict(np.zeros((args.imgsz, args.imgsz, 3), np.uint8),
                  imgsz=args.imgsz, device=args.device or None, verbose=False)

    is_file = args.video and os.path.isfile(args.video)
    if args.video and not is_file and not args.video.startswith("/dev/"):
        log.critical("no such video file: %s", args.video)
        return 1
    source = FileSource(args.video) if is_file else open_camera(args)
    if source is None:
        return 1
    zoom = build_zoom(args)
    detector = Detector(model, args)

    window = not args.headless
    if window and not os.environ.get("DISPLAY") \
            and not os.environ.get("WAYLAND_DISPLAY"):
        log.warning("no display -- running headless (pass --headless to "
                    "silence this)")
        window = False

    ego = EgoMotion()
    pan_speed = SpeedPID(args.speed_kp, args.speed_ki, args.pan_full_rate)
    tilt_speed = SpeedPID(args.speed_kp, args.speed_ki, args.tilt_full_rate)
    last_stamp = None
    selector = TargetSelector(args.reacquire)
    pan_ctl = PID(args.kp * args.pan_gain, args.pan_ki,
                  args.kd * args.pan_gain, args.deadband,
                  args.max_speed, args.accel, args.i_limit, args.pan_lead,
                  args.smooth)
    tilt_ctl = PID(args.kp * args.tilt_gain, args.tilt_ki,
                   args.kd * args.tilt_gain, args.deadband,
                   args.max_speed, args.accel, args.i_limit, args.tilt_lead,
                   args.smooth)
    ui = {"paused": False, "click": None, "jog": (0.0, 0.0), "jog_until": 0.0,
          "view_width": args.display_width,
          # zoom: held by mouse button, or by key / wheel until a deadline
          "zoom_mouse": 0, "zoom_key": 0, "zoom_until": 0.0, "zoom_last_key": 0.0}
    if args.armed:
        head.arm()
        log.info("starting ARMED -- servos will move as soon as someone is "
                 "in view")
    else:
        head.kill()
        log.info("starting KILLED -- press g in the window to arm the servos")

    def on_mouse(event, x, y, flags, *_):
        vw = ui["view_width"]
        if event == cv2.EVENT_LBUTTONDOWN:
            if in_rect(kill_button_rect(vw), x, y):
                head.kill()       # right here, not on the next loop pass
                log.warning("KILL button pressed -- servos off (g to arm)")
                return
            if zoom is not None:
                rects = zoom_button_rects(vw)
                for name, sign in (("in", 1), ("out", -1)):
                    if in_rect(rects[name], x, y):
                        ui["zoom_mouse"] = sign
                        return
            ui["click"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            ui["zoom_mouse"] = 0
        elif event == cv2.EVENT_MOUSEWHEEL and zoom is not None:
            sign = 1 if cv2.getMouseWheelDelta(flags) > 0 else -1
            now = time.monotonic()
            if ui["zoom_key"] != sign or now >= ui["zoom_until"]:
                ui["zoom_until"] = now
            ui["zoom_key"] = sign
            ui["zoom_until"] = min(ui["zoom_until"] + ZOOM_WHEEL_STEP, now + 1.5)

    if window:
        # NORMAL, not AUTOSIZE: only a resizable window can go full screen.
        # Clicks still arrive in image pixels, so the button hit tests do
        # not care how big the window is drawn.
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        if args.windowed:
            cv2.resizeWindow(WINDOW, args.display_width,
                             args.display_width * 9 // 16)
        else:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(WINDOW, on_mouse)
    last_saved = 0.0

    seq, frames, last_report = 0, 0, time.monotonic()
    det_seq, people, det_at = 0, [], None
    target = aim = None
    pan = tilt = 0.0
    state = "starting"
    zooming = 0
    try:
        while True:
            frame, new_seq, stamp = source.wait_for_frame(seq, timeout=1.0)
            now = time.monotonic()
            if frame is None:
                head.stop()
                if is_file and source.ended:
                    log.info("end of %s", args.video)
                    break
                log.warning("no video for 1 s -- servos stopped")
                continue
            seq = new_seq
            frames += 1
            h, w = frame.shape[:2]
            detector.submit(frame, seq)
            frame_dt = (stamp - last_stamp) if (stamp and last_stamp) else None
            last_stamp = stamp
            if args.speed_pid:
                ego.update(frame, stamp)
                pan_speed.measure(ego.pan_rate, ego.ok, frame_dt)
                tilt_speed.measure(ego.tilt_rate, ego.ok, frame_dt)
            ratio = max(1.0, zoom.ratio) if zoom is not None else 1.0

            # -- zoom: mouse hold wins, then a key/wheel still running
            if zoom is not None:
                wanted = ui["zoom_mouse"] or (
                    ui["zoom_key"] if now < ui["zoom_until"] else 0)
                if wanted != zooming:
                    zoom.drive(wanted)
                    zooming = wanted
                ratio = max(1.0, zoom.ratio)
                for ctl, gain, ki in ((pan_ctl, args.pan_gain, args.pan_ki),
                                      (tilt_ctl, args.tilt_gain, args.tilt_ki)):
                    ctl.kp = args.kp * gain / ratio
                    ctl.ki = ki / ratio
                    ctl.kd = args.kd * gain / ratio

            # -- a new detection: choose the target, update the control
            result_seq, result_people, result_at = detector.latest()
            fresh = result_seq != det_seq
            if fresh:
                det_seq, people, det_at = result_seq, result_people, result_at
                if ui["click"] is not None:
                    x, y = ui["click"]
                    ui["click"] = None
                    scale = ui["view_width"] / w
                    if not selector.select_at(people, x / scale, y / scale,
                                              now):
                        log.info("no person under the click")
                target = selector.update(people, now, w)
                aim = aim_point(target, args.aim) if target is not None else None
            stale = det_at is None or now - det_at > DETECTION_STALE

            killed = head.killed or head.hardware_killed
            if killed:
                pan_ctl.reset()
                tilt_ctl.reset()
                pan, tilt = 0.0, 0.0
                state = "KILLED" + (f", would track {selector.locked_id}"
                                    if aim is not None else "")
            elif ui["paused"]:
                pan_ctl.reset()
                tilt_ctl.reset()
                pan, tilt = ui["jog"] if now < ui["jog_until"] else (0.0, 0.0)
                state = "PAUSED (space resumes, WASD jogs)"
            elif stale:
                pan_ctl.reset()
                tilt_ctl.reset()
                pan, tilt = 0.0, 0.0
                state = "waiting for detection"
            elif fresh:
                if aim is not None:
                    ex = (aim[0] - w / 2) / (w / 2)
                    ey = vertical_error(target, aim[1], h)
                    zone_x, zone_y = centre_zone(args)
                    pan = pan_ctl.update(ex, det_at, zone_x)
                    tilt = (tilt_ctl.update(-ey, det_at, zone_y)
                            if not args.no_tilt else 0.0)
                    state = (f"tracking {selector.locked_id}  "
                             f"err x{ex:+.2f} y{ey:+.2f}")
                elif selector.seen_within(now, args.lost_hold):
                    pan, tilt = pan_ctl.output, tilt_ctl.output  # coast briefly
                    state = "target missed, coasting"
                else:
                    pan_ctl.reset()
                    tilt_ctl.reset()
                    pan, tilt = 0.0, 0.0
                    state = ("target lost, holding"
                             if selector.last_box is not None
                             else "searching for a person")

            # -- inner loop: correct the command by the speed actually seen
            if args.speed_pid and not killed and not ui["paused"]:
                pan_out = pan_speed.update(pan, ratio, frame_dt, ego.ok)
                # A 180 tilt servo holds the angle it is sent: it cannot
                # stall the way a 360 one does, so no speed correction.
                tilt_out = (tilt if head.positional_tilt else
                            tilt_speed.update(tilt, ratio, frame_dt, ego.ok))
            else:
                pan_speed.reset()
                tilt_speed.reset()
                pan_out, tilt_out = pan, tilt
            pulses = head.set_speed(pan_out, tilt_out)
            if not head.alive:
                log.critical("servo board stopped answering -- quitting")
                return 1

            if now - last_report >= 1.0:
                span = now - last_report
                ui["fps"] = frames / span
                ui["det_fps"] = detector.take_count() / span
                log.info("%5.1f fps  detect %4.1f/s %3.0f ms  %d people  %s  "
                         "speed pan%+.2f tilt%+.2f  pulses %s/%s us%s",
                         ui["fps"], ui["det_fps"], detector.ms, len(people),
                         state, pan, tilt, *pulses,
                         f"  zoom {zoom.ratio:.1f}x" if zoom else "")
                frames, last_report = 0, now

            save_due = args.save_view and now - last_saved >= 1.0
            if window or save_due:
                status = [state,
                          f"speed pan {pan:+.2f} tilt {tilt:+.2f}  "
                          f"pulses {pulses[0]}/{pulses[1]} us",
                          "PID pan P{:+.2f} I{:+.2f} D{:+.2f}   "
                          "tilt P{:+.2f} I{:+.2f} D{:+.2f}".format(
                              *pan_ctl.terms, *tilt_ctl.terms),
                          (f"speed want/seen pan {pan_speed.target:+.2f}/"
                           f"{(pan_speed.measured or 0):+.2f} "
                           f"({pan_speed.correction:+.2f})  tilt "
                           f"{tilt_speed.target:+.2f}/"
                           f"{(tilt_speed.measured or 0):+.2f} "
                           f"({tilt_speed.correction:+.2f})"
                           + ("" if ego.ok else "  [no motion fix]"))
                          if args.speed_pid else "speed PID off",
                          f"video {ui.get('fps', 0):.0f} fps  detect "
                          f"{ui.get('det_fps', 0):.0f}/s ({detector.ms:.0f} ms)"
                          f"  servos {head.port}"]
                why = "hardware" if head.hardware_killed else "software"
                view, _ = draw(frame, people, target, aim, args, status,
                               killed, why, zoom, zooming)
                ui["view_width"] = view.shape[1]
                if save_due:
                    tmp = args.save_view + ".tmp.jpg"
                    cv2.imwrite(tmp, view)
                    os.replace(tmp, args.save_view)
                    last_saved = now
            if window:
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("f"):
                    full = cv2.getWindowProperty(
                        WINDOW, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                    cv2.setWindowProperty(
                        WINDOW, cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_NORMAL if full else cv2.WINDOW_FULLSCREEN)
                elif key in (ord("+"), ord("="), ord("-"), ord("_")) and zoom:
                    sign = 1 if key in (ord("+"), ord("=")) else -1
                    repeat = (ui["zoom_key"] == sign
                              and now - ui["zoom_last_key"] < ZOOM_KEY_FIRST)
                    ui["zoom_key"] = sign
                    ui["zoom_until"] = now + (ZOOM_KEY_REPEAT if repeat
                                              else ZOOM_KEY_FIRST)
                    ui["zoom_last_key"] = now
                elif key == ord("0") and zoom:
                    ui["zoom_until"] = 0.0
                    zoom.go_wide()
                    zooming = 0
                    log.info("zoom: all the way out")
                elif key == ord("k"):
                    head.kill()
                    log.warning("KILL -- servos off (g to arm)")
                elif key == ord("g") and head.killed:
                    head.arm()
                    pan_ctl.reset()
                    tilt_ctl.reset()
                    log.warning("ARMED -- servos live")
                elif key == ord(" "):
                    ui["paused"] = not ui["paused"]
                    log.info("paused" if ui["paused"] else "resumed")
                elif key == ord("r"):
                    selector.release()
                elif ui["paused"] and key in map(ord, "wasd"):
                    ui["jog"] = {ord("w"): (0.0, JOG_SPEED),
                                 ord("s"): (0.0, -JOG_SPEED),
                                 ord("a"): (-JOG_SPEED, 0.0),
                                 ord("d"): (JOG_SPEED, 0.0)}[key]
                    ui["jog_until"] = now + JOG_HOLD
    except KeyboardInterrupt:
        pass
    finally:
        head.stop()
        detector.stop()
        if zoom is not None:
            zoom.close()
        source.stop()
        if window:
            cv2.destroyAllWindows()
    return 0


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("video")
    g.add_argument("--video", help="UVC device or a video file "
                   "(default: autodetect the FCB)")
    g.add_argument("--width", type=int, default=1920)
    g.add_argument("--height", type=int, default=1080)
    g.add_argument("--fourcc", help="force a pixel format, e.g. YUYV")

    g = p.add_argument_group("detection")
    g.add_argument("--model", default="yolo11n.pt",
                   help="YOLO weights; on the Orin export a TensorRT .engine "
                   "(default: %(default)s, downloaded on first use)")
    g.add_argument("--imgsz", type=int, default=640)
    g.add_argument("--conf", type=float, default=0.4)
    g.add_argument("--device", default="",
                   help="cpu, 0 (first GPU), ... (default: best available)")

    g = p.add_argument_group("tracking")
    # 2026-10-06: overshooting on both axes, so less P (via --pan-gain and
    # --tilt-gain) and twice the D, which brakes on the approach.
    g.add_argument("--kp", type=float, default=0.9,
                   help="speed per unit of image error (error is -1..1 "
                   "from centre to edge); lower it at high zoom")
    # Pan servo supply raised 5 V -> 7.2 V (2026-10-06): ~1.44x the speed per
    # command, so pan's speed settings were scaled by 5/7.2 (~0.7).
    g.add_argument("--pan-ki", type=float, default=0.6,
                   help="pan integral gain, per second: how fast a stalled "
                   "or lagging pan is pushed harder (default: %(default)s; "
                   "raised from 0.4, then 0.7, on 2026-10-06)")
    g.add_argument("--tilt-ki", type=float, default=0.25,
                   help="tilt integral gain (default: %(default)s)")
    g.add_argument("--kd", type=float, default=0.25,
                   help="derivative gain: brakes approaching centre "
                   "(default: %(default)s)")
    g.add_argument("--i-limit", type=float, default=0.5,
                   help="most the integral can add to the speed "
                   "(default: %(default)s)")
    # Pan doubled (2026-10-06) when it took ~5 s to centre someone, then
    # brought back down -- 1.4, then 1.0 -- once a freed cable and a higher
    # servo voltage made it overshoot (-0.41 to +0.58 of the frame in 1 s).
    # 1.0 then felt a touch slow, so 1.15.
    # 1.15 overshot, so 0.95; still overshooting, so 0.8 (with --pan-lead
    # and a lower --pan-max-us).
    # Pan servo supply raised 5 V -> 7.2 V (2026-10-06): ~1.44x the speed per
    # command, so pan's speed settings were scaled by 5/7.2 (~0.7).
    g.add_argument("--pan-gain", type=float, default=0.55,
                   help="pan P and D as a multiple of --kp/--kd "
                   "(default: %(default)s)")
    # 0.8 oscillated up and down about a person (2026-10-06); 0.5 was a
    # touch slow, so 0.6.
    # 0.6 overshot, so 0.5.
    g.add_argument("--tilt-gain", type=float, default=0.5,
                   help="tilt P and D as a multiple of --kp/--kd "
                   "(default: %(default)s)")
    g.add_argument("--box-w", type=float, default=0.06,
                   help="centre box round the person: its width as a "
                   "fraction of the frame's. Pan stops while the crosshair "
                   "is inside (default: %(default)s)")
    g.add_argument("--box-h", type=float, default=0.12,
                   help="... and its height as a fraction of the frame's; "
                   "tilt stops while the crosshair is inside "
                   "(default: %(default)s)")
    g.add_argument("--speed-pid", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="inner speed loop: measure how fast the head really "
                   "turns from the picture and correct the servo command, "
                   "so a stuck servo is pushed harder (default: on; "
                   "--no-speed-pid to turn off)")
    g.add_argument("--speed-kp", type=float, default=0.1,
                   help="speed loop P (default: %(default)s)")
    g.add_argument("--speed-ki", type=float, default=1.0,
                   help="speed loop I, per second: how fast a stuck servo "
                   "is pushed harder (default: %(default)s)")
    g.add_argument("--pan-full-rate", type=float, default=1.5,
                   help="starting guess for pan's speed at full command, "
                   "half-frames/s at 1x zoom; learnt while running "
                   "(default: %(default)s)")
    g.add_argument("--tilt-full-rate", type=float, default=1.0,
                   help="the same for tilt (default: %(default)s)")
    g.add_argument("--pan-lead", type=float, default=0.2,
                   help="stop pan this many seconds early, judging where "
                   "it is heading, to allow for servo coast and detection "
                   "delay (default: %(default)s)")
    g.add_argument("--tilt-lead", type=float, default=0.1,
                   help="the same for tilt (default: %(default)s)")
    g.add_argument("--deadband", type=float, default=0.06,
                   help="smallest stop zone, fraction of the half-frame "
                   "(default: %(default)s)")
    g.add_argument("--max-speed", type=float, default=1.0)
    g.add_argument("--accel", type=float, default=3.0,
                   help="how fast speed may build up, per second; it may "
                   "drop 2.5x as fast (default: %(default)s, 4.0 was jerky)")
    g.add_argument("--smooth", type=float, default=0.08,
                   help="smoothing time constant on the person's position, "
                   "seconds, against jerky servo commands; 0 turns it off "
                   "(default: %(default)s)")
    g.add_argument("--aim", type=float, default=0.3,
                   help="centre this far down the person's box (0=head top)")
    g.add_argument("--lost-hold", type=float, default=0.3,
                   help="keep moving this long after losing the target")
    g.add_argument("--reacquire", type=float, default=2.0,
                   help="wait this long before switching to someone else")
    g.add_argument("--no-tilt", action="store_true", help="pan only")

    g = p.add_argument_group("servos")
    g.add_argument("--servo-port", help="Arduino serial port "
                   "(default: autodetect)")
    g.add_argument("--dry-run", action="store_true",
                   help="no servo board; track and log only")
    g.add_argument("--armed", action="store_true",
                   help="start with the servos live instead of killed")
    g.add_argument("--servo-test", action="store_true",
                   help="turn each servo each way, then exit")
    g.add_argument("--pan-neutral", type=int, default=1500)
    g.add_argument("--tilt-neutral", type=int, default=1500)
    # Per axis: the two servos on this rig need very different pulses to
    # start turning. The pan servo did not move at all on +50 us
    # (2026-10-06), which the tilt servo turns fine on.
    # Pan servo supply raised 5 V -> 7.2 V (2026-10-06): ~1.44x the speed per
    # command, so the smallest step is 80 us: it turns
    # on less now, and 100 was a bigger speed jump (jerk).
    g.add_argument("--pan-deadzone-us", type=int, default=80,
                   help="smallest offset from neutral that turns the pan "
                   "servo (default: %(default)s)")
    # Full range: the pan servo spun properly only once properly powered,
    # and the user asked for its maximum (2026-10-06).
    # 500 (full range) overshot once the servo had proper power: 350.
    # Pan servo supply raised 5 V -> 7.2 V (2026-10-06): ~1.44x the speed per
    # command, so 250 us gives the old top speed.
    g.add_argument("--pan-max-us", type=int, default=250,
                   help="pan offset at full speed (default: %(default)s)")
    # Pan turned right fine but barely left, even at full signal
    # (2026-10-06), so left got 1.4x longer pulses. Back to 1.0 now the pan
    # turns on a bearing (2026-10-08): that weakness was friction and drag.
    g.add_argument("--pan-left-boost", type=float, default=1.0,
                   help="multiply pan's pulse offsets turning left "
                   "(default: %(default)s)")
    # Tilt is a positional 180 servo since 2026-10-08: its pulse is an angle.
    g.add_argument("--tilt-servo", type=int, choices=(180, 360), default=180,
                   help="tilt servo type: 180 (positional) or 360 "
                   "(continuous) (default: %(default)s)")
    g.add_argument("--tilt-low-us", type=int, default=1000,
                   help="180 tilt: lowest pulse it may be sent -- one end of "
                   "its travel (default: %(default)s)")
    g.add_argument("--tilt-high-us", type=int, default=2000,
                   help="180 tilt: highest pulse -- the other end "
                   "(default: %(default)s)")
    g.add_argument("--tilt-centre-us", type=int, default=1500,
                   help="180 tilt: angle it goes to when armed "
                   "(default: %(default)s)")
    g.add_argument("--tilt-rate-us", type=float, default=800.0,
                   help="180 tilt: how fast the angle moves at full speed, "
                   "us per second (default: %(default)s)")
    g.add_argument("--tilt-deadzone-us", type=int, default=30,
                   help="360 tilt only: smallest offset that turns it "
                   "(default: %(default)s)")
    g.add_argument("--tilt-max-us", type=int, default=150,
                   help="360 tilt only: offset at full speed "
                   "(default: %(default)s)")
    # Pan as mounted (2026-10-06) turned the opposite way to the one asked
    # for, so it is reversed. The 180 tilt servo is new (2026-10-08) and its
    # direction is unchecked: run --servo-test and flip it if needed.
    g.add_argument("--pan-invert", action=argparse.BooleanOptionalAction,
                   default=True, help="reverse pan (default: %(default)s; "
                   "--no-pan-invert to undo)")
    g.add_argument("--tilt-invert", action=argparse.BooleanOptionalAction,
                   default=False, help="reverse tilt (default: %(default)s)")

    g = p.add_argument_group("zoom")
    g.add_argument("--no-zoom", action="store_true",
                   help="do not open the camera's VISCA port")
    g.add_argument("--visca-port", help="camera control port "
                   "(default: the camera board's own)")
    g.add_argument("--zoom-speed", type=int, default=4,
                   help="VISCA zoom speed 0-7 (default: %(default)s)")

    g = p.add_argument_group("display")
    g.add_argument("--headless", action="store_true")
    g.add_argument("--windowed", action="store_true",
                   help="start in a window instead of full screen (f toggles)")
    g.add_argument("--display-width", type=int,
                   help="width the view is drawn at (default: the camera's "
                   "own width full screen, 960 in a window)")
    g.add_argument("--save-view", metavar="JPG",
                   help="also write the annotated view here once a second")
    args = p.parse_args()

    if args.display_width is None:
        args.display_width = 960 if args.windowed else args.width
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    head = build_head(args)
    if head is None:
        return 1
    try:
        if args.servo_test:
            servo_test(head, 0.5)
            return 0
        return track(args, head)
    finally:
        head.close()


if __name__ == "__main__":
    sys.exit(main())
