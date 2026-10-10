"""Pan/tilt servos on an Arduino: a 360 servo for pan, a 180 one for tilt.

Since 2026-10-08 tilt is a positional (180) servo -- its pulse width is an
angle -- and pan is still a continuous-rotation (360) one, whose pulse
width is a speed. `Axis` below maps a speed to a 360 servo's pulse;
`PositionAxis` turns the same speed request into a moving angle for a 180
servo. The tracker asks both axes for a speed either way.

A 360 servo's pulse width sets a speed, not an angle: ~1500 us is "stop",
and the further a pulse is from that, the faster it turns. So this takes a
speed per axis in [-1, 1] and turns it into microseconds. Two quirks of
these servos shape the mapping:

- Every one stops at a slightly different pulse (1480-1520 us is typical)
  and drifts slowly at anything else. Rather than trim each one, a zero
  speed is sent as "no pulses at all" (0 in the protocol), which stops any
  analogue 360 servo dead regardless of where its neutral sits.
- Near neutral there is a band of pulse widths that do nothing. A small
  speed is therefore mapped to start at the edge of that band
  (`deadzone_us`), not at neutral, or small corrections would never move
  the head and the tracker would sit just off-target.

The Arduino stops both servos if it hears nothing for 300 ms, so this
re-sends the current command from a background thread. Only a host that
has actually died lets the watchdog fire.

Two kill switches, both latched. `kill()` is the software one: the
command is forced to stop until `arm()`. The hardware one is a button on
the Arduino (D2 to GND) that the sketch obeys on its own, so it still
works when this process has hung; its state is read back as
`hardware_killed`.

A 180 tilt servo is never released while the head is in use, killed or
not: with no pulses it goes limp and the camera's weight drags it down.
So `kill()` stops pan and *holds* tilt where it is. Only the hardware
button, the Arduino's watchdog (this process gone) and `close()` release
it.

Sign convention: +pan turns the camera right, +tilt tilts it up. The
`invert` flags fix up whichever way round the servos happen to be mounted.
"""
import threading
import time
from dataclasses import dataclass

try:
    import serial
    from serial.tools import list_ports
except ImportError:  # reported when a port is actually opened
    serial = None
    list_ports = None

BAUD = 115200
IDENTITY = "PANTILT 1"
KEEPALIVE_INTERVAL = 0.05  # 20 Hz, well inside the sketch's 300 ms watchdog

#: USB vendor IDs of boards that run the sketch: Arduino (two), the CH340
#: and FTDI and CP210x bridges on clones, Adafruit, and the Raspberry Pi Pico.
MCU_USB_VIDS = {0x2341, 0x2A03, 0x1A86, 0x0403, 0x10C4, 0x239A, 0x2E8A}
#: The camera board's own CDC serial port. Never a candidate: writing servo
#: commands into the VISCA channel would at best be ignored.
CAMERA_USB_VIDS = {0x04B4}


class PanTiltError(RuntimeError):
    pass


@dataclass
class Axis:
    neutral_us: int = 1500
    deadzone_us: int = 30   # smallest offset from neutral that moves at all
    max_us: int = 150       # offset at full speed (speed 1.0)
    invert: bool = False
    #: Scales both offsets for a *negative* requested speed (pan left, tilt
    #: down), before `invert`. A 360 servo whose real stop point is off
    #: centre turns slower one way for the same distance from neutral; this
    #: gives the weak direction longer pulses to match.
    negative_scale: float = 1.0

    def pulse(self, speed):
        """Pulse width for `speed` in [-1, 1]; 0 means stopped (no pulses)."""
        speed = max(-1.0, min(1.0, float(speed)))
        if speed == 0.0:
            return 0
        scale = self.negative_scale if speed < 0 else 1.0
        deadzone, top = self.deadzone_us * scale, self.max_us * scale
        offset = deadzone + abs(speed) * max(0.0, top - deadzone)
        if self.invert:
            speed = -speed
        return int(round(self.neutral_us + (offset if speed > 0 else -offset)))


class PositionAxis:
    """A 180 (positional) servo driven by speed requests.

    Its pulse width sets an angle, so a speed request moves a target angle
    at `rate_us` microseconds per second (at speed 1.0), between `min_us`
    and `max_us`. It holds that angle when asked for zero speed. It is not
    driven at all -- no pulses -- until `centre()` is called, so the servo
    does not jump the moment the program connects, only when the head is
    armed.
    """

    def __init__(self, min_us=1000, max_us=2000, centre_us=1500,
                 invert=False, rate_us=800.0):
        self.min_us, self.max_us, self.centre_us = min_us, max_us, centre_us
        self.invert, self.rate_us = invert, rate_us
        self.pulse_us = None
        self._last = None

    def centre(self):
        self.pulse_us = float(self.centre_us)

    def current(self):
        return 0 if self.pulse_us is None else int(round(self.pulse_us))

    def step(self, speed, now):
        """Advance the target angle by `speed` since the last call."""
        last, self._last = self._last, now
        if self.pulse_us is None or last is None:
            return self.current()
        dt = max(0.0, min(0.1, now - last))
        speed = max(-1.0, min(1.0, float(speed)))
        if self.invert:
            speed = -speed
        self.pulse_us = max(self.min_us, min(
            self.max_us, self.pulse_us + speed * self.rate_us * dt))
        return self.current()

    def hold(self, now):
        """Keep the angle; just note the time, so no jump on the next step."""
        self._last = now
        return self.current()


def find_port(log=print):
    """The one serial port that looks like a microcontroller, or None.

    Picks by USB vendor, never by "first ttyACM": the camera board's VISCA
    channel is also a ttyACM, and on a laptop the order depends on what was
    plugged in first. More than one candidate is reported, not guessed at.
    """
    if list_ports is None:
        log("pyserial not installed -- pip3 install pyserial")
        return None
    candidates = [
        p for p in list_ports.comports()
        if p.vid is not None and p.vid in MCU_USB_VIDS
        and p.vid not in CAMERA_USB_VIDS
    ]
    if len(candidates) == 1:
        p = candidates[0]
        log(f"servo board: {p.device} ({p.vid:04x}:{p.pid:04x} "
            f"{p.description})")
        return p.device
    if not candidates:
        log("no Arduino-like serial port found (looked for USB vendors "
            + ", ".join(f"{v:04x}" for v in sorted(MCU_USB_VIDS))
            + ") -- is it plugged in? Pass --servo-port to name it")
    else:
        log("more than one possible servo board: "
            + ", ".join(p.device for p in candidates)
            + " -- pass --servo-port to pick one")
    return None


class PanTilt:
    """Speed commands to the Arduino, kept alive from a background thread."""

    def __init__(self, port, pan=None, tilt=None, log=print):
        self.positional_tilt = isinstance(tilt, PositionAxis)
        if serial is None:
            raise PanTiltError("pyserial not installed -- pip3 install pyserial")
        self.pan_axis = pan or Axis()
        self.tilt_axis = tilt or Axis()
        self._log = log
        self._lock = threading.Lock()
        self._command = (0, 0)
        self.killed = False
        self.hardware_killed = False
        self.port = port
        try:
            self._serial = serial.Serial(port, BAUD, timeout=0.1)
        except serial.SerialException as exc:
            raise PanTiltError(f"cannot open {port}: {exc}") from exc
        self._handshake()
        self._running = True
        self._thread = threading.Thread(target=self._keepalive, daemon=True)
        self._thread.start()

    def _handshake(self):
        # Opening the port resets an Uno (DTR), and its bootloader eats
        # anything sent in the first ~1.5 s, so wait for the sketch's banner
        # before asking who it is.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if IDENTITY in self._readline():
                break
        for _ in range(3):
            self._serial.reset_input_buffer()
            self._serial.write(b"?\n")
            reply_by = time.monotonic() + 0.5
            while time.monotonic() < reply_by:
                if self._readline() == IDENTITY:
                    self._log(f"servo board on {self.port} answered "
                              f"\"{IDENTITY}\"")
                    return
        self._serial.close()
        raise PanTiltError(
            f"{self.port} did not answer \"{IDENTITY}\" -- is "
            f"pantilt_servo/pantilt_servo.ino flashed onto it?")

    def _readline(self):
        return self._serial.readline().decode("ascii", "replace").strip()

    def _keepalive(self):
        pending = b""
        while self._running:
            with self._lock:
                pan_us, tilt_us = self._command
                if self.killed:
                    pan_us = 0
                    tilt_us = (self.tilt_axis.current()
                               if self.positional_tilt else 0)
            try:
                self._serial.write(f"S {pan_us} {tilt_us}\n".encode())
                if self._serial.in_waiting:
                    pending += self._serial.read(self._serial.in_waiting)
                    *lines, pending = pending.split(b"\n")
                    for line in lines:
                        self._board_said(line.decode("ascii", "replace").strip())
            except (serial.SerialException, OSError) as exc:
                # The Arduino's watchdog has already stopped the servos.
                self._log(f"servo board lost: {exc}")
                self._running = False
                return
            time.sleep(KEEPALIVE_INTERVAL)

    def _board_said(self, line):
        if line in ("KILLED", "ARMED"):
            killed = line == "KILLED"
            if killed != self.hardware_killed:
                self._log("hardware kill switch: " + line)
            self.hardware_killed = killed

    @property
    def alive(self):
        return self._running

    def kill(self):
        """Latch the servos off until arm(): pan stops, a 180 tilt holds.

        Sent at once rather than on the next keepalive. A 360-only head gets
        "X"; with a 180 tilt that would let it go limp for a moment, so it
        gets pan stopped and tilt held instead.
        """
        with self._lock:
            self.killed = True
            tilt_us = self.tilt_axis.current() if self.positional_tilt else 0
            self._command = (0, tilt_us)
        try:
            self._serial.write(f"S 0 {tilt_us}\n".encode()
                               if self.positional_tilt else b"X\n")
        except (serial.SerialException, OSError):
            pass

    def arm(self):
        with self._lock:
            self.killed = False
            if self.positional_tilt and self.tilt_axis.pulse_us is None:
                self.tilt_axis.centre()

    def set_speed(self, pan, tilt):
        """Speeds in [-1, 1]: +pan turns right, +tilt tilts up.

        A 180 tilt turns the speed into a moving angle (PositionAxis).
        Returns the pulses actually going out: pan 0 while killed, and tilt
        held where it was.
        """
        now = time.monotonic()
        with self._lock:
            stopped = self.killed or self.hardware_killed
            if self.positional_tilt:
                tilt_us = (self.tilt_axis.hold(now) if stopped
                           else self.tilt_axis.step(tilt, now))
            else:
                tilt_us = self.tilt_axis.pulse(tilt)
            command = (self.pan_axis.pulse(pan), tilt_us)
            self._command = command
            if stopped:
                return (0, tilt_us if self.positional_tilt else 0)
        return command

    def stop(self):
        return self.set_speed(0.0, 0.0)

    def close(self):
        self.stop()
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            self._serial.write(b"X\n")
            self._serial.flush()
        except (serial.SerialException, OSError):
            pass
        self._serial.close()


class DryRunPanTilt:
    """Same interface, no hardware: for testing tracking with nothing wired."""

    port = "dry-run"
    alive = True
    hardware_killed = False

    def __init__(self, pan=None, tilt=None, log=print):
        self.pan_axis = pan or Axis()
        self.tilt_axis = tilt or Axis()
        self.positional_tilt = isinstance(self.tilt_axis, PositionAxis)
        self.killed = False

    def kill(self):
        self.killed = True

    def arm(self):
        self.killed = False
        if self.positional_tilt and self.tilt_axis.pulse_us is None:
            self.tilt_axis.centre()

    def set_speed(self, pan, tilt):
        now = time.monotonic()
        if self.positional_tilt:
            tilt_us = (self.tilt_axis.hold(now) if self.killed
                       else self.tilt_axis.step(tilt, now))
        else:
            tilt_us = 0 if self.killed else self.tilt_axis.pulse(tilt)
        if self.killed:
            return (0, tilt_us)
        return (self.pan_axis.pulse(pan), tilt_us)

    def stop(self):
        return self.set_speed(0.0, 0.0)

    def close(self):
        pass
