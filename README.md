# fcb_manipulator: the FCB follows a person on a two-servo pan/tilt

    ./run.sh

The tracker opens **full screen** with the servos **KILLED**: you see the
feed and who it would follow, and nothing moves until you press `g`.

| Key / control | Does |
|---|---|
| `g` | arm the servos |
| `k`, or click the red KILL button | stop the servos, latched until `g` |
| pushbutton on Arduino D2 to GND | hardware kill. Works even if the laptop hangs. Press again to re-arm |
| `+` / `-` (hold), the on-screen + / − buttons (hold), mouse wheel | zoom in / out (the FCB's optical zoom, over VISCA) |
| `0` | zoom all the way out |
| `f` | full screen on / off (`./run.sh --windowed` starts in a window) |
| click a person | follow that person |
| `r` | drop the target and pick again |
| `space` | pause; `w a s d` then jog the head by hand |
| `q`, `Esc`, Ctrl+C | quit (servos stop) |

The Arduino also stops both servos by itself if it hears nothing from the
laptop for 300 ms.

**Setup, once:** `python3 -m venv .venv && .venv/bin/pip install ultralytics pyserial`
(`run.sh` uses `.venv` automatically).

**Pan = yaw** (turn left/right): a 360 servo on **D11**. **Tilt = pitch** (nod
up/down): a **180** servo on **D10**.

**Frame rate.** Detection runs on its own thread, so the picture runs at the
camera's 60 fps however long detection takes. The servos update once per
detection: about 20 a second on the CPU, and faster with the GPU `.venv`.

**Tests:** `python3 test/test_pantilt.py`. They need no hardware: the
Arduino is faked.

`fcb_track.py` points the camera at a person and keeps them centred:

    FCB video -> YOLO person detection + ByteTrack IDs -> pick one person
      -> error from frame centre -> PD speed per axis
      -> Arduino (pantilt_servo/) -> pan (yaw, 360) D11, tilt (pitch, 180) D10

It opens the camera itself, so it cannot run alongside `~/fcb_orin`'s
`fcb_record.py`. The camera code in `fcb_base_driver/` is copied from
`~/fcb_orin`. If you fix camera handling there, copy the fix here too.

**Wiring.** Connect the servo signal wires to D11 (pan, 360) and D10 (tilt, 180) on an
Uno or Nano. Power the servos from their own 5-6 V supply, not the
Arduino's 5 V pin, and join the grounds. Flash
`pantilt_servo/pantilt_servo.ino` with the Arduino IDE. For an optional
hardware kill switch, wire a pushbutton from D2 to GND.

**First run.** With the servos connected and the camera on the head:

    ./run.sh --servo-test      # each servo turns each way in turn
    ./run.sh --dry-run         # tracking only, no servos
    ./run.sh                   # the real thing: full screen, starts KILLED

Pan is reversed by default (`--pan-invert`) to match how it is mounted. The
180 tilt servo is not reversed (`--tilt-invert` to flip it). Check both with
`--servo-test`, and flip whichever moves the wrong way. Each servo has its own start and full-speed offsets. Pan starts at
`--pan-deadzone-us 80` and reaches full speed at `--pan-max-us 250`;
Pan used to turn left more weakly than right, so `--pan-left-boost` gives it
longer pulses turning left. It is 1.0 (off) now that pan turns on a bearing;
set it to 1.4 if left is weak again. The sketch accepts 500-2500 us. Raise the deadzone of a servo that does not move on
small corrections, and lower its max if full speed is too fast. A 360 servo commands speed,
not angle, so a stopped servo receives **no pulses at all**. That stops
an analogue 360 servo whatever its exact neutral is. If one of yours
still creeps (some digital ones hold their last pulse), it needs
`--pan-neutral`/`--tilt-neutral` set to its real stop point. The Arduino
stops both servos if the host goes quiet for 300 ms.

**Control: a PID per axis.** P turns towards the person in proportion to how
far off-centre they are. I keeps pushing harder while they stay off-centre,
which gets a stalled servo moving again. It is capped (`--i-limit`) and
cleared once they are centred. D brakes on the approach to stop overshoot.
The live P, I and D values are shown on screen.

**Centre box.** A fixed-size box is drawn round the person's aim point:
`--box-w 0.06` of the frame's width by `--box-h 0.12` of its height. It does
not grow or shrink with the person. It is orange while the crosshair is
outside it and the servos are driving, and green once the crosshair is
inside it. Each axis stops as soon as the crosshair is inside the box on that
axis, and its integral is cleared. It starts again only once the crosshair is
clearly outside (1.25x the box).

**Speed PID (inner loop).** The servos have no encoder, so the camera measures
its own turning speed from the picture. When the head pans, the whole scene
slides across the frame, and phase correlation between frames (about 2 ms)
measures how far. The position PID sets how fast the head should turn. The
speed PID compares that with how fast it really is turning and corrects the
servo command. A stuck servo is pushed harder until it moves, and one going
too fast is eased off. The correction is at most +-0.4, it never reverses
the direction asked for, and a stop is always a stop. The servo's real speed
per command is learnt while running (from moving moments only, never from
a stall), so a voltage change needs no retuning. The screen shows
`speed want/seen` for each axis, with the correction in brackets.
`--no-speed-pid` turns it off. Its gains are `--speed-kp 0.1` and
`--speed-ki 1.0`. The measured speed is smoothed over 0.15 s, and the
correction changes by at most 1.5 per second, so the noisy measurement
does not make the head jerky.

**Stopping early.** A 360 servo coasts after it is told to stop, and each
detection is slightly old. So while closing in, an axis stops once the
crosshair *will be* inside the box `--pan-lead 0.2` / `--tilt-lead 0.1`
seconds later, at the current approach speed. Raise the lead if it still
overshoots.

**Servo voltage.** These pan settings are for a **7.2 V** pan servo supply.
At 5 V the same settings run about 30% slower: use `--pan-gain 0.8
--pan-ki 0.85 --pan-max-us 350 --pan-deadzone-us 100`.

**Smoothness.** The person's position is smoothed (`--smooth 0.08` s) before
the PID sees it, so the wobble of their box frame to frame does not turn
into jerky servo commands. Speed builds up at `--accel 3` per second and
eases down 2.5x as fast while moving. Stopping at the box is instant.

Defaults: `--kp 0.9`, `--kd 0.25`, `--pan-gain 0.55`, `--tilt-gain 0.5`
(both scale P and D), `--pan-ki 0.6`, `--tilt-ki 0.25`, and pan top speed
`--pan-max-us 250`.

**Tuning.** `--kp` is the speed at full error, where the target is at the
edge of the frame. The FCB's field of view narrows as it zooms in, so the
same servo speed sweeps the picture faster. The tracker therefore divides
the gain by the current magnification automatically. `--aim 0.3` centres a point 30%
of the way down the person's box (the chest), not the waist.

**Tilt is a 180 (positional) servo.** Its pulse width is an angle, so the
tilt PID moves a target angle at up to `--tilt-rate-us 800` us per second,
kept between `--tilt-low-us 1000` and `--tilt-high-us 2000`. Narrow those if
the camera can hit the frame at either end. When armed, tilt goes to
`--tilt-centre-us 1500`. After that it **holds its angle**, even when
killed: kill stops pan and freezes tilt where it is, because releasing a
180 servo lets the camera's weight drag it down. Tilt is released only by
the hardware button, by quitting, or by the Arduino's watchdog if the
laptop goes quiet. The speed PID is not used on tilt, since a positional
servo does not stall. For a 360 tilt servo again, use `--tilt-servo 360`.

**On the Orin.** Install JetPack's PyTorch build first, then
`pip3 install ultralytics`. Export the model once to TensorRT and pass
the result to `--model`:
`yolo export model=yolo11n.pt format=engine half=True`.

