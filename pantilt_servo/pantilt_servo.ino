// Pan/tilt driver: a continuous-rotation (360) pan servo and a positional
// (180) tilt servo. The sketch treats both the same -- pulse widths in,
// servo pulses out; whether a pulse means a speed or an angle is the host's
// business.
//
// The host (fcb_track.py) owns all the control logic; this sketch only turns
// pulse widths into servo pulses and stops everything if the host goes
// quiet. A 360 servo has no position feedback -- its pulse width sets a
// *speed* -- so a host that crashes mid-move would otherwise leave the
// gimbal spinning until someone pulls power.
//
// Serial, 115200 baud, newline-terminated ASCII:
//
//   S <pan_us> <tilt_us>   set both pulse widths in microseconds.
//                          0 stops that servo by sending no pulses at all,
//                          which stops a 360 servo without having to know
//                          its exact neutral point.
//   X                      stop both
//   ?                      identify -> "PANTILT 1"
//   D                      self-check: watch the pan and tilt pins for 200 ms
//                          and report "DIAG pan=<pins> <high%> tilt=..." --
//                          ~7.5% high for a 1500 us pulse every 20 ms, 0 if
//                          the pin is not pulsing or is shorted to ground
//
// No command for WATCHDOG_MS and both servos stop.
//
// Kill switch: a pushbutton from KILL_PIN (D2) to GND. A press latches
// both servos off and S commands are ignored; press again to re-arm. It
// works in here, so it still stops the head when the host has hung. The
// sketch reports "KILLED" / "ARMED" on each change, and on "?" too. With
// no button fitted the pull-up holds the pin high: always armed.
//
// Wiring (Uno/Nano): pan (360) signal -> D11, tilt (180) signal -> D10.
// Power the
// servos from their own 5-6 V supply, NOT the Arduino's 5 V pin, and tie
// that supply's ground to the Arduino's GND.

#include <Servo.h>

const int PAN_PIN = 11;  // D9, then D6, then D11 (2026-10-08)
const int TILT_PIN = 10;
const int KILL_PIN = 2;
const unsigned long DEBOUNCE_MS = 50;
const unsigned long WATCHDOG_MS = 300;
// Wider than the usual 1000-2000: a 360 servo that is weak in one direction
// can need longer pulses that way. Most accept 500-2500 us.
const int MIN_US = 500;
const int MAX_US = 2500;

Servo pan, tilt;
unsigned long lastCommand = 0;
char line[40];
byte lineLen = 0;
bool killed = false;
bool lastButton = HIGH;
unsigned long buttonChanged = 0;

void reportKill() {
  Serial.println(killed ? "KILLED" : "ARMED");
}

void drive(Servo &servo, int pin, long us) {
  if (us == 0) {
    if (servo.attached()) servo.detach();
    digitalWrite(pin, LOW);
    return;
  }
  if (us < MIN_US) us = MIN_US;
  if (us > MAX_US) us = MAX_US;
  if (!servo.attached()) servo.attach(pin);
  servo.writeMicroseconds((int)us);
}

void stopAll() {
  drive(pan, PAN_PIN, 0);
  drive(tilt, TILT_PIN, 0);
}

void diagnose() {
  // digitalRead on an output pin reads the real pin level, so a pin that is
  // driven but shorted to ground reads low here too.
  unsigned long panHigh = 0, tiltHigh = 0, samples = 0;
  unsigned long start = millis();
  while (millis() - start < 200) {
    panHigh += digitalRead(PAN_PIN);
    tiltHigh += digitalRead(TILT_PIN);
    samples++;
  }
  Serial.print("DIAG pan=D"); Serial.print(PAN_PIN);
  Serial.print(pan.attached() ? " attached " : " detached ");
  Serial.print(100.0 * panHigh / samples, 1);
  Serial.print("% tilt=D"); Serial.print(TILT_PIN);
  Serial.print(tilt.attached() ? " attached " : " detached ");
  Serial.print(100.0 * tiltHigh / samples, 1);
  Serial.println("%");
}

void handle(char *cmd) {
  if (cmd[0] == 'S') {
    lastCommand = millis();
    if (killed) return;
    char *end;
    long p = strtol(cmd + 1, &end, 10);
    long t = strtol(end, NULL, 10);
    drive(pan, PAN_PIN, p);
    drive(tilt, TILT_PIN, t);
  } else if (cmd[0] == 'X') {
    stopAll();
    lastCommand = millis();
  } else if (cmd[0] == 'D') {
    diagnose();
    lastCommand = millis();
  } else if (cmd[0] == '?') {
    Serial.println("PANTILT 1");
    reportKill();
    lastCommand = millis();
  }
}

void setup() {
  pinMode(PAN_PIN, OUTPUT);
  pinMode(TILT_PIN, OUTPUT);
  pinMode(KILL_PIN, INPUT_PULLUP);
  stopAll();
  Serial.begin(115200);
  Serial.println("PANTILT 1 READY");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (lineLen) {
        line[lineLen] = 0;
        handle(line);
        lineLen = 0;
      }
    } else if (lineLen < sizeof(line) - 1) {
      line[lineLen++] = c;
    } else {
      lineLen = 0;  // overlong garbage: drop it
    }
  }
  bool button = digitalRead(KILL_PIN);
  if (button != lastButton && millis() - buttonChanged > DEBOUNCE_MS) {
    buttonChanged = millis();
    lastButton = button;
    if (button == LOW) {          // pressed
      killed = !killed;
      if (killed) stopAll();
      reportKill();
    }
  }
  if (killed || millis() - lastCommand > WATCHDOG_MS) stopAll();
}
