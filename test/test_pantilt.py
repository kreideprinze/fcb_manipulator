"""fcb_track.py + pantilt.py: servo link, speed mapping, control, target choice.

No Arduino and no camera: the board is faked on a pseudo-terminal that
speaks the pantilt_servo.ino protocol.
"""
import os, sys, pty, threading, time, tty
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fcb_base_driver.pantilt import Axis, PanTilt, PanTiltError, PositionAxis, DryRunPanTilt
import types
import numpy as np, cv2
from fcb_track import (PID, TargetSelector, aim_point, vertical_error,
                       centre_zone, EgoMotion, SpeedPID)

fails = []
def check(label, ok, detail=""):
    if not ok: fails.append(label)
    print(("  ok   " if ok else "  FAIL ") + label + (("  -- " + str(detail)) if detail and not ok else ""))


class FakeBoard:
    """The sketch's side of the serial link, on a pty."""

    def __init__(self, identity="PANTILT 1"):
        self.master, slave = pty.openpty()
        tty.setraw(slave)
        self.path = os.ttyname(slave)
        self.identity = identity
        self.commands = []
        self._stop = False
        os.write(self.master, f"{identity} READY\n".encode())
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        buf = b""
        while not self._stop:
            try:
                buf += os.read(self.master, 256)
            except OSError:
                return
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.decode().strip()
                if line == "?":
                    os.write(self.master, f"{self.identity}\n".encode())
                elif line:
                    self.commands.append((time.monotonic(), line))


# -- speed -> pulse mapping ----------------------------------------------
a = Axis(neutral_us=1500, deadzone_us=30, max_us=150)
check("zero speed sends no pulses", a.pulse(0.0) == 0)
check("smallest speed starts past the servo's dead zone",
      1530 <= a.pulse(0.001) <= 1531, a.pulse(0.001))
check("full speed is neutral + max_us", a.pulse(1.0) == 1650)
check("full reverse", a.pulse(-1.0) == 1350)
check("speed is clamped", a.pulse(5.0) == 1650)
check("invert flips direction",
      Axis(1500, 30, 150, invert=True).pulse(0.5) == 1500 - (30 + 60))
b = Axis(1500, 100, 500, invert=True, negative_scale=1.4)
check("boost lengthens only the weak (negative) direction",
      b.pulse(1.0) == 1000 and b.pulse(-1.0) == 2200, (b.pulse(1.0), b.pulse(-1.0)))
check("boost applies to the smallest step as well",
      1640 <= b.pulse(-0.001) <= 1641, b.pulse(-0.001))
check("per-servo neutral is the centre of the mapping",
      Axis(1490, 30, 150).pulse(-1.0) == 1340)

# -- serial link -----------------------------------------------------------
board = FakeBoard()
head = PanTilt(board.path, Axis(), Axis(), log=lambda m: None)
pulses = head.set_speed(1.0, -1.0)
check("set_speed returns the pulses sent", pulses == (1650, 1350), pulses)
time.sleep(0.35)
sent = [c for _, c in board.commands]
check("command reaches the board", "S 1650 1350" in sent, sent[-3:])
recent = [t for t, c in board.commands if c == "S 1650 1350"]
gaps = [b - a for a, b in zip(recent, recent[1:])]
check("re-sent well inside the 300 ms watchdog",
      len(recent) >= 4 and max(gaps) < 0.15, gaps)
head.close()
time.sleep(0.1)
check("closing stops both servos", board.commands[-1][1] == "X",
      board.commands[-2:])

# -- kill switches ---------------------------------------------------------
board = FakeBoard()
head = PanTilt(board.path, Axis(), Axis(), log=lambda m: None)
head.set_speed(1.0, 1.0)
head.kill()
check("software kill: set_speed reports nothing going out",
      head.set_speed(1.0, 1.0) == (0, 0))
time.sleep(0.2)
last = [c for _, c in board.commands][-2:]
check("software kill: the board is told to stop", "X" in [c for _, c in board.commands]
      and all(c == "S 0 0" for c in last), last)
head.arm()
time.sleep(0.15)
check("arm: commands flow again", board.commands[-1][1] == "S 1650 1650",
      board.commands[-1])
os.write(board.master, b"KILLED\n")
time.sleep(0.2)
check("hardware kill is read back", head.hardware_killed)
check("hardware kill: set_speed reports nothing going out",
      head.set_speed(1.0, 1.0) == (0, 0))
os.write(board.master, b"ARMED\n")
time.sleep(0.2)
check("hardware re-arm is read back", not head.hardware_killed)
head.close()

impostor = FakeBoard(identity="SOMETHING ELSE")
t0 = time.monotonic()
try:
    PanTilt(impostor.path, log=lambda m: None)
    check("a board without the sketch is refused", False)
except PanTiltError as exc:
    check("a board without the sketch is refused", "flashed" in str(exc), exc)
check("...and refused within a few seconds", time.monotonic() - t0 < 6)

# -- 180 tilt servo: speed requests move an angle ---------------------
pa = PositionAxis(1000, 2000, 1500, invert=False, rate_us=800)
check("180 tilt: not driven before it is centred", pa.step(1.0, 0.0) == 0)
pa.centre(); pa.step(0.0, 0.0)
check("180 tilt: centred on arm", pa.current() == 1500)
pa.step(0.5, 0.0)
check("180 tilt: speed moves the angle (0.5 x 800 us/s x 0.1 s = +40)",
      pa.step(0.5, 0.1) == 1540, pa.current())
check("180 tilt: zero speed holds the angle", pa.step(0.0, 0.5) == 1540)
for i in range(1, 40):
    pa.step(1.0, 0.5 + i * 0.1)
check("180 tilt: never past its high end", pa.current() == 2000, pa.current())
for i in range(40, 100):
    pa.step(-1.0, 0.5 + i * 0.1)
check("... nor its low end", pa.current() == 1000, pa.current())
inv = PositionAxis(1000, 2000, 1500, invert=True, rate_us=800)
inv.centre(); inv.step(0.0, 0.0)
check("180 tilt: invert flips the direction", inv.step(0.5, 0.1) == 1460, inv.current())
long_gap = PositionAxis(1000, 2000, 1500, rate_us=800)
long_gap.centre(); long_gap.step(0.0, 0.0)
check("180 tilt: a long gap between updates does not jump it",
      long_gap.step(1.0, 5.0) <= 1500 + 80, long_gap.current())

board = FakeBoard()
head = PanTilt(board.path, Axis(), PositionAxis(1000, 2000, 1500, rate_us=800),
               log=lambda m: None)
time.sleep(0.15)
check("180 tilt: no pulses at all until armed",
      all(c.endswith(" 0") for _, c in board.commands if c.startswith("S")),
      board.commands[-2:])
head.arm()
head.set_speed(0.0, 0.0)
time.sleep(0.15)
check("180 tilt: armed -> sent its centre angle", board.commands[-1][1] == "S 0 1500",
      board.commands[-1])
head.set_speed(1.0, 0.5); time.sleep(0.2); head.set_speed(1.0, 0.5)
held = head.tilt_axis.current()
head.kill()
time.sleep(0.2)
last = board.commands[-1][1]
check("kill with a 180 tilt: pan stops, tilt HELD (not released)",
      last == f"S 0 {held}" and held > 1500, (last, held))
check("... and no X (which would let it go limp)",
      "X" not in [c for _, c in board.commands[-5:]])
check("killed: set_speed does not move tilt", head.set_speed(1.0, 1.0) == (0, held))
head.close()

dry = DryRunPanTilt(Axis(), PositionAxis(1000, 2000, 1500, rate_us=800))
dry.arm()
check("dry run: 180 tilt centred on arm", dry.set_speed(0, 0)[1] == 1500)

# -- controller ------------------------------------------------------------
c = PID(kp=1.0, ki=0.0, kd=0.0, deadband=0.06, max_speed=1.0, accel=4.0)
check("inside the deadband: stopped", c.update(0.03, 0.0) == 0.0)
out = [c.update(0.8, 0.05 * i) for i in range(1, 12)]
check("speed builds up, not a jump", out[0] <= 0.21, out[:3])
check("reaches kp * error", abs(out[-1] - 0.8) < 1e-6, out[-1])
check("stops at once when back in the deadband", c.update(0.02, 0.7) == 0.0)
check("hysteresis: just outside the deadband stays stopped",
      c.update(0.07, 0.75) == 0.0)
check("...until clearly outside", c.update(0.2, 0.8) != 0.0)
c2 = PID(1.0, 0.0, 0.0, 0.06, 0.5, 100.0)
c2.update(-0.9, 0.0)
check("negative error turns the other way, capped at max_speed",
      c2.update(-0.9, 0.05) == -0.5, c2.output)

# -- integral ----------------------------------------------------------
# A stalled servo: the error never shrinks. P alone sits at kp * error for
# ever; the integral has to keep pushing harder.
st = PID(kp=0.5, ki=0.4, kd=0.0, deadband=0.06, max_speed=1.0, accel=100.0)
outs = [st.update(0.3, 0.05 * i) for i in range(0, 100)]   # 5 s stalled
check("P alone at the start", abs(outs[1] - 0.15) < 0.02, outs[1])
check("integral pushes harder the longer it stays off-centre",
      outs[60] > outs[20] > outs[2], (outs[2], outs[20], outs[60]))
check("integral is capped (anti-windup)", st.terms[1] <= 0.5 + 1e-9, st.terms)
check("... and so is the output", max(outs) <= 1.0)
st.update(0.02, 5.1)
check("centring clears the integral", st.terms == (0.0, 0.0, 0.0), st.terms)
check("... and the next move starts from P alone",
      abs(st.update(0.3, 5.15) - 0.15) < 0.03, st.output)

sat = PID(kp=5.0, ki=1.0, kd=0.0, deadband=0.06, max_speed=1.0, accel=100.0)
for i in range(40):
    sat.update(0.5, 0.05 * i)                   # P alone saturates
check("no wind-up while the output is already at full speed",
      abs(sat.terms[1]) < 0.06, sat.terms)

side = PID(kp=0.5, ki=1.0, kd=0.0, deadband=0.06, max_speed=1.0, accel=100.0)
for i in range(40):
    side.update(0.3, 0.05 * i)
side.update(-0.3, 2.05)
check("crossing to the other side drops the stored push",
      side.terms[1] <= 0 and side.output < 0, (side.terms, side.output))

# -- person centre box ---------------------------------------------------
opts = types.SimpleNamespace(box_w=0.06, box_h=0.12, deadband=0.06)
check("centre box is a fixed share of the frame, whatever the person's size",
      centre_zone(opts) == (0.06, 0.12))
z = PID(kp=1.0, ki=0.5, kd=0.0, deadband=0.06, max_speed=1.0, accel=100.0)
z.update(0.4, 0.0, zone=0.15)
check("outside the centre box: drives", z.update(0.4, 0.05, zone=0.15) > 0)
check("crosshair inside the centre box: stops, integral cleared",
      z.update(0.12, 0.10, zone=0.15) == 0.0 and z.terms == (0.0, 0.0, 0.0))
check("stays stopped just outside the box (hysteresis)",
      z.update(0.17, 0.15, zone=0.15) == 0.0)
check("... and drives again once clearly outside",
      z.update(0.25, 0.20, zone=0.15) > 0)

# -- stopping early -----------------------------------------------------
def approach(lead):
    c = PID(kp=1.0, ki=0.0, kd=0.0, deadband=0.06, max_speed=1.0,
            accel=100.0, lead=lead)
    t, e = 0.0, 0.6
    while e > 0.15:                        # closing in at 2.0 per second
        c.update(e, t, zone=0.1)
        t, e = t + 0.05, e - 0.1
    return c.update(e, t, zone=0.1)       # error 0.15: still outside the box
check("without lead: still driving just outside the box", approach(0.0) > 0)
check("with lead: stops early, where it will be in 0.2 s",
      approach(0.2) == 0.0)
far = PID(kp=1.0, ki=0.0, kd=0.0, deadband=0.06, max_speed=1.0, accel=100.0,
          lead=0.2)
far.update(0.5, 0.0, zone=0.1)
check("lead does not stop it while moving away", far.update(0.55, 0.05, zone=0.1) > 0)

# -- smoothness ----------------------------------------------------------
def jitter(smooth):
    c = PID(kp=1.0, ki=0.0, kd=0.25, deadband=0.06, max_speed=1.0,
            accel=100.0, smooth=smooth)
    outs = [c.update(0.4 + (0.03 if i % 2 else -0.03), i / 60) for i in range(120)]
    tail = outs[60:]
    return max(abs(b - a) for a, b in zip(tail, tail[1:]))
raw, smoothed = jitter(0.0), jitter(0.08)
check("smoothing cuts frame-to-frame speed jitter from a wobbling box",
      smoothed < raw / 3, (raw, smoothed))
sd = PID(kp=1.0, ki=0.0, kd=0.0, deadband=0.06, max_speed=1.0, accel=3.0)
for i in range(60):
    sd.update(0.8, i / 60)                 # cruising at 0.8
step_down = sd.output - sd.update(0.3, 1.0)
check("slowing down while moving is ramped, not a step",
      0 < step_down <= 2.5 * 3.0 / 60 + 1e-9, step_down)
check("... but stopping in the box is still immediate",
      sd.update(0.01, 1.02) == 0.0)

# -- measuring the head's real speed from the picture -------------------
rng = np.random.default_rng(1)
scene = cv2.GaussianBlur((rng.random((900, 2400)) * 255).astype(np.uint8), (0, 0), 3)
scene = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
def view_at(x, y):            # what a 1920x1080 camera sees, looking at x, y
    return scene[y:y + 540, x:x + 960].copy()
ego = EgoMotion()
ego.update(view_at(400, 200), 0.0)
ego.update(view_at(424, 200), 1 / 60)    # camera turned right: scene slides left
check("camera turning right reads as +pan", ego.ok and ego.pan_rate > 0, (ego.ok, ego.pan_rate))
check("... at the right speed (24 of 960 px a frame = 3 half-frames/s)",
      abs(ego.pan_rate - 24 / 480 * 60) < 0.3, ego.pan_rate)
ego.update(view_at(424, 188), 2 / 60)    # camera tilted up: scene slides down
check("camera tilting up reads as +tilt", ego.ok and ego.tilt_rate > 0, ego.tilt_rate)
blank = EgoMotion()
blank.update(np.full((540, 960, 3), 128, np.uint8), 0.0)
blank.update(np.full((540, 960, 3), 128, np.uint8), 1 / 60)
check("a blank picture gives no measurement, not 'stopped'", not blank.ok)

# -- speed PID ---------------------------------------------------------
sp = SpeedPID(kp=0.3, ki=1.0, full_rate=1.5)
outs = []
for i in range(60):                     # told to turn at 0.4, but stuck
    sp.measure(0.0, True, 1 / 60)
    outs.append(sp.update(0.4, 1.0, 1 / 60, True))
check("stuck servo: pushed harder and harder", outs[-1] > outs[5] > 0.4, (outs[5], outs[-1]))
check("... but only by up to the correction limit", outs[-1] <= 0.4 + 0.4 + 1e-9)
check("a stall does not teach it the servo is slow", sp.full_rate == 1.5, sp.full_rate)
check("stop request passes straight through", sp.update(0.0, 1.0, 1 / 60, True) == 0.0)
fast = SpeedPID(kp=0.3, ki=1.0, full_rate=1.5)
for i in range(60):                     # asked for 0.2 (0.3 hf/s), turning at 1.2
    fast.measure(1.2, True, 1 / 60)
    o = fast.update(0.2, 1.0, 1 / 60, True)
check("too fast: eased off, but never turned the other way", 0.0 <= o < 0.2, o)
nm = SpeedPID(kp=0.3, ki=1.0, full_rate=1.5)
check("no motion fix: command passes through unchanged",
      nm.update(0.5, 1.0, 1 / 60, False) == 0.5)
hold = SpeedPID(kp=0.1, ki=1.0, full_rate=1.5)
for i in range(60):                     # stuck: correction builds up
    hold.measure(0.0, True, 1 / 60)
    built = hold.update(0.4, 1.0, 1 / 60, True)
check("a dropped measurement holds the correction, no jump",
      abs(hold.update(0.4, 1.0, 1 / 60, False) - built) < 1e-9)
ramp = SpeedPID(kp=0.1, ki=50.0, full_rate=1.5)
ramp.measure(0.0, True, 1 / 60)
first = ramp.update(0.4, 1.0, 1 / 60, True) - 0.4
check("the correction ramps, even with a huge I", 0 < first <= 1.5 / 60 + 1e-9, first)
learn = SpeedPID(kp=0.0, ki=0.0, full_rate=1.5)
for i in range(400):                    # servo really does 3 hf/s at full
    learn.measure(3.0 * 0.6, True, 1 / 60)
    learn.update(0.6, 1.0, 1 / 60, True)
check("learns the servo's real speed while it moves", abs(learn.full_rate - 3.0) < 0.15, learn.full_rate)

# -- target choice ---------------------------------------------------------
s = TargetSelector(reacquire=2.0)
small, big = (1, (0, 0, 50, 100)), (2, (500, 0, 700, 400))
check("locks onto the largest person", s.update([small, big], 0.0, 1920) == big[1])
check("follows the locked ID even when it is not the largest",
      s.update([(2, (510, 0, 560, 50)), (7, (0, 0, 900, 900))], 0.1, 1920)
      == (510, 0, 560, 50))
check("new ID in the same place is the same person",
      s.update([(9, (515, 0, 565, 50))], 0.2, 1920) == (515, 0, 565, 50)
      and s.locked_id == 9)
far = (4, (1700, 0, 1800, 100))
check("someone else far away is not taken straight away",
      s.update([far], 0.3, 1920) is None)
check("...but is after the reacquire time", s.update([far], 2.5, 1920) == far[1])
check("click selects the person under it",
      s.select_at([small, far], 20, 50, 3.0) and s.locked_id == 1)
check("aim point is near the top of the box, not the waist",
      aim_point((0, 0, 100, 1000), 0.3) == (50, 300))

check("person filling the frame top to bottom: no tilt error",
      vertical_error((100, 0, 400, 1080), 324, 1080) == 0.0)
check("person cut off only at the top: tilt up (negative error)",
      vertical_error((100, 0, 400, 700), 210, 1080) < 0)
check("person fully in frame: error from the aim point",
      abs(vertical_error((100, 270, 400, 810), 432, 1080) + 0.2) < 1e-9)

print("FAILED: " + ", ".join(fails) if fails else "all passed")
sys.exit(1 if fails else 0)
