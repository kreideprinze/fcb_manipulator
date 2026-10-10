"""Zoom in / out by hand over VISCA, from a background thread.

The UI only says which way it wants the lens to go (in, out, or stop);
this thread turns changes of that into CAM_Zoom Tele / Wide / Stop and
polls the lens position so the magnification can be shown and used.
Nothing in the UI waits on the serial link: a slow or lost reply costs
the zoom readout a moment, never a video frame.

Variable-speed Tele/Wide rather than absolute positions, for the same
reason zoom_servo.py gives in fcb_orin: a drive command is acknowledged
at once, while a stream of Direct commands fills the camera's two command
sockets and gets refused.

Only the camera board's own serial ports are probed. Sending VISCA bytes
to whatever else is plugged in -- the servo Arduino -- is not harmless.
"""
import threading
import time

from fcb_base_driver import devices, zoom_map
from fcb_base_driver.visca_link import ViscaLink

LOOP_INTERVAL = 0.03
POLL_MOVING = 0.2     # s between position reads while the lens moves
POLL_IDLE = 1.0       # ... and while it is parked


class ZoomDrive:

    def __init__(self, port=None, speed=4, log=print):
        """Raises RuntimeError if no camera answers VISCA."""
        self._log = log
        self.speed = max(0, min(7, int(speed)))
        if port:
            found_port, baud = devices.autodetect_visca(ports=[port])
        else:
            ports = devices.camera_serial_ports()
            if not ports:
                raise RuntimeError("the camera board has no serial port -- "
                                   "zoom control unavailable")
            found_port, baud = devices.autodetect_visca(ports=ports)
        if found_port is None:
            raise RuntimeError("the camera did not answer VISCA on "
                               f"{port or ', '.join(ports)}")
        self.port = found_port
        self._link = ViscaLink(found_port, baud)
        self._lock = threading.Lock()
        self._wanted = 0      # -1 out, 0 stop, +1 in
        self._goto = None     # absolute position, for "all the way out"
        self.position = None
        self.ratio = 1.0
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log(f"zoom: VISCA on {found_port} at {baud} baud")

    def drive(self, direction):
        """+1 zoom in, -1 zoom out, 0 stop. Cheap; call as often as you like."""
        with self._lock:
            self._wanted = (direction > 0) - (direction < 0)

    def go_wide(self):
        """Zoom all the way out -- the quickest way to find someone again."""
        with self._lock:
            self._wanted = 0
            self._goto = 0

    def _run(self):
        sent = 0
        next_poll = 0.0
        while self._running:
            with self._lock:
                wanted, goto = self._wanted, self._goto
                self._goto = None
            try:
                if goto is not None:
                    self._link.zoom_direct(goto)
                    sent = 0
                    next_poll = 0.0
                elif wanted != sent:
                    if wanted > 0:
                        self._link.zoom_tele(self.speed)
                    elif wanted < 0:
                        self._link.zoom_wide(self.speed)
                    else:
                        self._link.zoom_stop()
                    sent = wanted
                    next_poll = 0.0
                now = time.monotonic()
                if now >= next_poll:
                    self.position = self._link.zoom_position()
                    self.ratio = zoom_map.position_to_ratio(self.position)
                    next_poll = now + (POLL_MOVING if sent or goto is not None
                                       else POLL_IDLE)
            except OSError as exc:
                # The port itself is gone -- the camera dropped off USB.
                # Every later write fails the same way, so say it once.
                self._log(f"zoom: camera control lost ({exc}) -- zoom off "
                          f"until restart")
                self._running = False
                return
            except Exception as exc:   # a dropped reply must not end zoom
                self._log(f"zoom: {exc}")
                sent = None            # resend whatever is wanted next pass
                time.sleep(0.2)
            time.sleep(LOOP_INTERVAL)

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        try:
            self._link.zoom_stop()
        except Exception:
            pass
        self._link.close()
