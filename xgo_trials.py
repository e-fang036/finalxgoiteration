#!/usr/bin/env python3
"""
XGO-mini2S object-retrieval trial runner
========================================

Runs the four-task trial from the research map:
    1. Recognition  - find and confirm the green ball
    2. Locomotion   - approach until the ball is close AND centred
    3. Pick-up      - grasp the ball with the arm
    4. Return       - carry it back to the starting position

Two control modes share identical vision and movement code, so they can be
compared fairly. Only the decision layer differs:
    neuromorphic : leaky integrate-and-fire (LIF) spiking neurons decide
    traditional  : plain threshold (if/else) logic decides

Every trial is appended to a CSV with its first failure stage (recognition,
locomotion, pickup, return), spike counts, and timing / CPU / memory figures.
A summary at the end checks the ">= 7 of 10 successful" criterion and names
the stage that failed most often (the one to focus the next iteration on).

Usage
-----
    Stop the factory demo (main.py) first, then:
    python3 xgo_trials.py --mode neuromorphic --trials 10
    python3 xgo_trials.py --mode traditional  --trials 10
    Add --no-display when running without a monitor / VNC.
"""

import argparse
import csv
import math
import os
import resource
import time
from datetime import datetime

import cv2
from xgolib import XGO

# ============================================================
# CONFIGURATION
# ============================================================

# -------------------- ROBOT ---------------------------------
SERIAL_PORT = '/dev/ttyAMA0'
# Must be 'xgolite' for this robot: with 'xgomini' the claw ignores
# commands (confirmed with claw_test.py). The version also sets the speed
# ranges, so re-check walking after changing it.
ROBOT_VERSION = 'xgolite'
MOTOR_SPEED = 100
GAIT_PACE = 'normal'          # 'slow', 'normal', 'high'
CAMERA_INDEX = 0

# -------------------- VISION --------------------------------
LOWER_GREEN = (35, 70, 70)
UPPER_GREEN = (85, 255, 255)
MIN_AREA = 500
MIN_CIRCULARITY = 0.50
STOP_AREA = 12000             # pixel area at which the ball is "near" (raise = stop closer)
CENTER_DEAD_ZONE = 0.15       # fraction of half-width counted as centred
CONFIRM_FRAMES = 3            # consecutive detections to confirm recognition
FLUSH_FRAMES = 4              # stale frames dropped after each movement
CAMERA_HFOV_DEG = 62.0        # approx. horizontal field of view

# Close-range cue: when the robot is right next to the ball, the ball sinks
# below the bottom edge of the image. These settings treat that as "near".
BOTTOM_ZONE = 0.12            # ball's lower edge within the bottom 12% of frame
CLIPPED_MIN_CIRCULARITY = 0.25  # a ball cut off by the edge looks less round
LOST_BELOW_MIN_AREA = 3000    # ball must already be fairly big for the cue
LOST_BELOW_WINDOW = 2.0       # s: how recently it must have been at the bottom
CUE_AREA = 15000              # area fed to the decision layer for the cue (> STOP_AREA)
BACKUP_PULSE = 1.0            # s: step back to re-find the ball if not centred

# -------------------- MOVEMENT ------------------------------
# The robot walks with a gait: a command must last long enough for at
# least one full step cycle, or the legs barely lift before stop() runs.
TURN_SPEED = 60
FORWARD_SPEED = 22            # far from the ball (xgolite forward max ~25)

MIN_TURN_PULSE = 0.8          # seconds
MAX_TURN_PULSE = 1.5
MIN_FORWARD_PULSE = 1.5       # used when returning to start
MAX_FORWARD_PULSE = 2.5
SEARCH_TURN_PULSE = 1.0
SETTLE_TIME = 0.40            # pause after each pulse before the next frame
RETURN_CHUNK = 2.0            # max seconds per forward pulse when returning

# Approach: the robot walks continuously and the camera decides when to
# stop, instead of fixed-length pulses that can carry it past the ball.
APPROACH_SPEED = 12           # slower final approach once the ball is big
SLOW_AREA = 5000              # ball area (px) at which it slows down
MAX_WALK_TIME = 4.0           # safety cap on one continuous walk (s)

# -------------------- ODOMETRY ------------------------------
FORWARD_CM_PER_SEC = 6.0      # at FORWARD_SPEED
TURN_DEG_PER_SEC = 45.0       # at TURN_SPEED
YAW_SIGN = 1                  # +1 if read_yaw() increases when turning left
USE_IMU = True                # use the robot's yaw sensor for turns
HEADING_TOLERANCE_DEG = 8.0
MAX_TURN_CORRECTIONS = 6

# -------------------- ARM / CLAW ----------------------------
CLAW_OPEN = 3                 # not 0: fully open can strain the claw servo
CLAW_CLOSED = 255
# Grasp sequence (see Robot.grasp): open -> [tilt] -> lower arm -> close
# -> [lift arm] -> [level body]. 
ARM_REACH_POLAR = (195, 155)  # arm_polar(theta, r): claw around the ball
ARM_LIFT_POLAR = None         # arm_polar(theta, r) after closing; None = stay put
# Tilt the body nose-down during the grasp so the claw reaches lower.
# Flip the sign if it tilts nose-up; 0 turns the tilt off.
GRASP_PITCH_DEG = 13          # about 15 is the servo limit
GRASP_SETTLE = 1.0            # stand-still time before the grasp starts (s)
# Part of the grasp: with the claw open, walk forward blind for this long at
# APPROACH_SPEED before lowering the arm, so the ball doesn't have to stay in
# view. Raise it to get closer; 0 = off; below ~0.5 s it only shuffles.
GRASP_FORWARD_TIME = 0
ARM_MOVE_TIME = 2.5           # time allowed for the arm to finish moving (s)
CLAW_MOVE_TIME = 1.5          # time allowed for the claw to finish moving (s)
# Servo speed used only while the claw closes (motor_speed range 1-255,
# higher = faster). Set back to MOTOR_SPEED straight afterwards.
CLAW_CLOSE_SPEED = 255

# -------------------- TRIAL LIMITS --------------------------
RECOGNITION_TIMEOUT = 60.0    # how long to search for the ball at the start
APPROACH_TIMEOUT = 90.0       # whole approach, including re-searching
BALL_LOST_TIMEOUT = 15.0      # how long to search if the ball is lost mid-approach

# -------------------- SPIKING NEURONS -----------------------
SIM_WINDOW = 0.100            # simulated seconds of neural activity per frame
SIM_DT = 0.001
NEURON_TAU = 0.020
NEURON_THRESHOLD = 1.0
NEURON_REFRACTORY = 0.004
STEER_GAIN = 5.0              # steering neurons fire when input > ~0.2
NEAR_GAIN = 1.0               # near neuron fires only when area > STOP_AREA
NEAR_MIN_SPIKES = 1

# -------------------- "BALL WENT BELOW THE CAMERA" MEMORY ---------
# Objects on the floor sit lower in the image the closer they are. A 'low'
# neuron fires when the ball's bottom edge nears the bottom of the frame,
# and its spikes feed a decaying trace (short-term memory). If the ball
# then disappears while that trace is still high, it went out of view
# underneath the camera, so the robot treats it as reached.
LOW_START = 0.75              # bottom edge below 75% of frame height starts the input
LOW_GAIN = 2.0                # low neuron fires when its input > 0.5
TRACE_TAU = 1.5               # seconds for the memory trace to decay by ~63%
LOST_NEAR_TRACE = 4.0         # trace needed at loss to infer "near"
# Equivalent rule for the traditional controller:
LOW_TRIGGER = 0.5             # same point at which the low neuron starts firing
LOST_NEAR_WINDOW = 1.0        # ball seen low within this many seconds -> near

# -------------------- ENERGY ESTIMATES ----------------------
# Fill in from your sources before analysis; left as None they are skipped.
ENERGY_PER_SPIKE_J = None     # joules per spike (e.g. from Lele et al., 2020)
CPU_ACTIVE_POWER_W = None     # measured CM4 active power, for CPU-time energy

LOG_FILE = 'trial_log.csv'
STAGES = ['recognition', 'locomotion', 'pickup', 'return']


# ============================================================
# HELPERS
# ============================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def ask_yes_no(prompt):
    while True:
        ans = input(f'{prompt} [y/n]: ').strip().lower()
        if ans in ('y', 'yes'):
            return True
        if ans in ('n', 'no'):
            return False


def ask_float(prompt):
    ans = input(prompt).strip()
    try:
        return float(ans)
    except ValueError:
        return None


class TrialAborted(Exception):
    pass


# ============================================================
# DECISION LAYERS
# ============================================================

class LIFNeuron:
    def __init__(self, gain):
        self.gain = gain
        self.reset()

    def reset(self):
        self.v = 0.0
        self.last_spike = -1e9

    def step(self, current, t, dt):
        if t - self.last_spike < NEURON_REFRACTORY:
            return False
        self.v += dt * (-self.v + self.gain * current) / NEURON_TAU
        if self.v >= NEURON_THRESHOLD:
            self.v = 0.0
            self.last_spike = t
            return True
        return False


class SpikingDecision:
    """
    Each frame, four LIF neurons are driven for SIM_WINDOW seconds by
    rate-coded inputs. The decision comes from their spike counts, so
    the spikes actually control the robot.
    """
    name = 'neuromorphic'
    neuron_names = ['turn_left', 'turn_right', 'centered', 'near', 'low']

    def __init__(self):
        self.neurons = {
            'turn_left': LIFNeuron(STEER_GAIN),
            'turn_right': LIFNeuron(STEER_GAIN),
            'centered': LIFNeuron(STEER_GAIN),
            'near': LIFNeuron(NEAR_GAIN),
            'low': LIFNeuron(LOW_GAIN),
        }
        self.steps = int(round(SIM_WINDOW / SIM_DT))
        self.reset_memory()

    # ---- short-term memory: decaying trace of 'low' spikes ----
    def reset_memory(self):
        self.trace = 0.0
        self.trace_t = time.monotonic()

    def _decay_trace(self):
        now = time.monotonic()
        self.trace *= math.exp(-(now - self.trace_t) / TRACE_TAU)
        self.trace_t = now

    def ball_lost(self):
        """Called when the ball vanishes: True if it went out below the camera."""
        self._decay_trace()
        return self.trace >= LOST_NEAR_TRACE

    def decide(self, error, area, low=0.0):
        dz2 = 2.0 * CENTER_DEAD_ZONE
        # Left/centred inputs cross over exactly at the dead-zone edge.
        inputs = {
            'turn_left': clamp(-error / dz2, 0.0, 1.0),
            'turn_right': clamp(error / dz2, 0.0, 1.0),
            'centered': clamp(1.0 - abs(error) / dz2, 0.0, 1.0),
            'near': min(area / STOP_AREA, 2.0),
            'low': low,
        }
        spikes = {k: 0 for k in self.neurons}
        for n in self.neurons.values():
            n.reset()

        for i in range(self.steps):
            t = i * SIM_DT
            for k, n in self.neurons.items():
                if n.step(inputs[k], t, SIM_DT):
                    spikes[k] += 1

        self._decay_trace()
        self.trace += spikes['low']

        lr = max(spikes['turn_left'], spikes['turn_right'])
        centred_wins = spikes['centered'] >= lr

        if spikes['near'] >= NEAR_MIN_SPIKES and centred_wins:
            return 'near', spikes
        if centred_wins:
            return 'centered', spikes
        if spikes['turn_left'] > spikes['turn_right']:
            return 'turn_left', spikes
        return 'turn_right', spikes


class ThresholdDecision:
    """Conventional if/else controller with the same behaviour."""
    name = 'traditional'
    neuron_names = []

    def __init__(self):
        self.reset_memory()

    def reset_memory(self):
        self.last_low_time = -1e9

    def ball_lost(self):
        return time.monotonic() - self.last_low_time <= LOST_NEAR_WINDOW

    def decide(self, error, area, low=0.0):
        if low >= LOW_TRIGGER:
            self.last_low_time = time.monotonic()
        centred = abs(error) <= CENTER_DEAD_ZONE
        if area >= STOP_AREA and centred:
            return 'near', {}
        if centred:
            return 'centered', {}
        return ('turn_left' if error < 0 else 'turn_right'), {}


# ============================================================
# VISION
# ============================================================

def detect_green_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_h = mask.shape[0]
    best, best_score = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        perim = cv2.arcLength(c, True)
        if perim <= 0:
            continue
        circ = 4.0 * math.pi * area / (perim * perim)
        bx, by, bw, bh = cv2.boundingRect(c)
        bottom = by + bh
        clipped = bottom >= frame_h - 2        # touching the bottom edge
        if circ < (CLIPPED_MIN_CIRCULARITY if clipped else MIN_CIRCULARITY):
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        score = area * circ
        if score > best_score:
            best_score = score
            best = {'x': int(x), 'y': int(y), 'radius': int(r),
                    'area': area, 'circularity': circ,
                    'bottom': bottom, 'clipped': clipped}
    return best, mask


class Camera:
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError('Could not open camera.')
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def read_fresh(self):
        # Drop buffered frames captured while the robot was moving.
        for _ in range(FLUSH_FRAMES):
            self.cap.grab()
        ok, frame = self.cap.read()
        return frame if ok else None

    def read_live(self):
        # Minimal flush, for use while the robot is walking continuously.
        self.cap.grab()
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


# ============================================================
# ROBOT
# ============================================================

class Robot:
    def __init__(self):
        self.dog = XGO(port=SERIAL_PORT, version=ROBOT_VERSION)
        time.sleep(0.5)
        self.dog.motor_speed(MOTOR_SPEED)
        try:
            self.dog.pace(GAIT_PACE)
        except Exception:
            pass
        self.imu_ok = False
        self.yaw0 = 0.0
        self.last_heading = 0.0
        self.est_heading = 0.0
        self.x = self.y = 0.0
        self.move_time = 0.0
        if USE_IMU:
            try:
                yaw = self.dog.read_yaw()
                self.imu_ok = isinstance(yaw, (int, float)) and math.isfinite(yaw)
            except Exception:
                self.imu_ok = False
        print('IMU yaw:', 'available' if self.imu_ok
              else 'unavailable - using timed turn estimates')

    # ---------- basics ----------
    def battery(self):
        try:
            return self.dog.read_battery()
        except Exception:
            return None

    def stop(self):
        try:
            self.dog.stop()
        except Exception:
            pass

    def stand(self):
        try:
            self.dog.reset()
        except Exception:
            pass
        time.sleep(1.5)

    def raw_yaw(self):
        return float(self.dog.read_yaw())

    def reset_pose(self):
        self.x = self.y = 0.0
        self.est_heading = self.last_heading = 0.0
        self.move_time = 0.0
        if self.imu_ok:
            try:
                self.yaw0 = self.raw_yaw()
            except Exception:
                self.imu_ok = False

    def heading(self):
        """Heading in degrees relative to the start; positive = left."""
        if self.imu_ok:
            try:
                self.last_heading = wrap180(YAW_SIGN * (self.raw_yaw() - self.yaw0))
                self.est_heading = self.last_heading
                return self.last_heading
            except Exception:
                self.imu_ok = False
        return self.est_heading

    def prepare_for_trial(self):
        self.stand()
        try:
            self.dog.claw(CLAW_OPEN)
        except Exception:
            pass
        time.sleep(0.5)
        self.reset_pose()

    # ---------- movement ----------
    def pulse(self, direction, speed, duration):
        """One movement long enough for the gait to step, then stop and settle."""
        h_before = self.heading()
        cmd = {'left': self.dog.turnleft, 'right': self.dog.turnright,
               'forward': self.dog.forward, 'back': self.dog.back}[direction]
        cmd(speed)
        time.sleep(duration)
        self.stop()
        time.sleep(SETTLE_TIME)
        self.move_time += duration

        if direction in ('forward', 'back'):
            d = FORWARD_CM_PER_SEC * duration * (speed / FORWARD_SPEED)
            if direction == 'back':
                d = -d
            h = math.radians(h_before)
            self.x += d * math.cos(h)
            self.y += d * math.sin(h)
        elif not self.imu_ok:
            dh = TURN_DEG_PER_SEC * duration * (speed / TURN_SPEED)
            self.est_heading = wrap180(
                self.est_heading + (dh if direction == 'left' else -dh))

    def start_walk(self, speed):
        """Begin walking forward until end_walk() is called."""
        self._walk_heading = self.heading()
        self._walk_speed = speed
        self._walk_t0 = time.monotonic()
        self.dog.forward(speed)

    def end_walk(self):
        """Stop a continuous walk and add the distance to the odometry."""
        self.stop()
        dur = time.monotonic() - self._walk_t0
        d = FORWARD_CM_PER_SEC * dur * (self._walk_speed / FORWARD_SPEED)
        h = math.radians(self._walk_heading)
        self.x += d * math.cos(h)
        self.y += d * math.sin(h)
        self.move_time += dur
        time.sleep(SETTLE_TIME)

    def turn_to(self, target):
        """Turn to an absolute heading (IMU closed-loop when available)."""
        for _ in range(MAX_TURN_CORRECTIONS):
            err = wrap180(target - self.heading())
            if abs(err) <= HEADING_TOLERANCE_DEG:
                return
            dur = clamp(abs(err) / TURN_DEG_PER_SEC, MIN_TURN_PULSE, 1.5)
            self.pulse('left' if err > 0 else 'right', TURN_SPEED, dur)

    def walk(self, distance_cm):
        remaining = distance_cm / FORWARD_CM_PER_SEC
        while remaining > 0.05:
            chunk = min(RETURN_CHUNK, remaining)
            self.pulse('forward', FORWARD_SPEED, max(chunk, MIN_FORWARD_PULSE))
            remaining -= chunk

    def return_home(self):
        """Face the start point, walk back, then restore original heading."""
        dist = math.hypot(self.x, self.y)
        print(f'Returning: estimated {dist:.0f} cm from start, '
              f'heading {self.heading():.0f} deg')
        if dist > 3.0:
            bearing = math.degrees(math.atan2(-self.y, -self.x))
            self.turn_to(bearing)
            self.walk(dist)
        self.turn_to(0.0)
        self.stop()

    # ---------- arm ----------
    def grasp(self):
        """
        Pick up the ball as a fixed sequence of single commands, one at a
        time, each given time to finish before the next is sent.
        """
        def step(label, action, wait):
            print(f'  [grasp] {label}')
            action()
            time.sleep(wait)

        print('Attempting pick-up...')
        step('stop walking and stand still', self.stop, GRASP_SETTLE)
        step(f'open claw -> claw({CLAW_OPEN})',
             lambda: self.dog.claw(CLAW_OPEN), CLAW_MOVE_TIME)
        if GRASP_FORWARD_TIME > 0:
            def step_forward():
                self.start_walk(APPROACH_SPEED)
                time.sleep(GRASP_FORWARD_TIME)
                self.end_walk()          # stops and records the distance
            step(f'step forward {GRASP_FORWARD_TIME} s', step_forward, 0.5)
        if GRASP_PITCH_DEG:
            step(f'tilt body -> attitude("p", {GRASP_PITCH_DEG})',
                 lambda: self.dog.attitude('p', GRASP_PITCH_DEG), 1.0)
        step(f'lower arm -> arm_polar{ARM_REACH_POLAR}',
             lambda: self.dog.arm_polar(*ARM_REACH_POLAR), ARM_MOVE_TIME)
        def fast_close():
            self.dog.motor_speed(CLAW_CLOSE_SPEED)   # speed up the servos
            time.sleep(0.1)
            self.dog.claw(CLAW_CLOSED)
        step(f'close claw fast -> motor_speed({CLAW_CLOSE_SPEED}), '
             f'claw({CLAW_CLOSED})', fast_close, CLAW_MOVE_TIME)
        step(f'restore servo speed -> motor_speed({MOTOR_SPEED})',
             lambda: self.dog.motor_speed(MOTOR_SPEED), 0.2)
        if ARM_LIFT_POLAR is not None:
            step(f'lift arm -> arm_polar{ARM_LIFT_POLAR}',
                 lambda: self.dog.arm_polar(*ARM_LIFT_POLAR), ARM_MOVE_TIME)
        if GRASP_PITCH_DEG:
            step('level body -> attitude("p", 0)',
                 lambda: self.dog.attitude('p', 0), 1.0)
        print('  [grasp] done')


# ============================================================
# TRIAL PHASES
# ============================================================

def new_stats(neuron_names):
    return {'frames': 0, 'decisions': 0, 'vision_s': 0.0, 'decision_s': 0.0,
            'spikes': {k: 0 for k in neuron_names}, 'total_spikes': 0}


def timed_detect(frame, stats):
    t = time.perf_counter()
    ball, mask = detect_green_ball(frame)
    stats['vision_s'] += time.perf_counter() - t
    stats['frames'] += 1
    return ball, mask


def low_signal(ball, frame_height):
    """0..1: how close the ball's bottom edge is to the bottom of the frame."""
    bottom = (ball['y'] + ball['radius']) / frame_height
    return clamp((bottom - LOW_START) / (1.0 - LOW_START), 0.0, 1.0)


def timed_lost(decider, stats):
    t = time.perf_counter()
    near = decider.ball_lost()
    stats['decision_s'] += time.perf_counter() - t
    return near


def timed_decide(decider, error, area, stats, low=0.0):
    t = time.perf_counter()
    decision, spikes = decider.decide(error, area, low)
    stats['decision_s'] += time.perf_counter() - t
    stats['decisions'] += 1
    for k, v in spikes.items():
        stats['spikes'][k] += v
        stats['total_spikes'] += v
    return decision


def show(frame, mask, ball, text, display):
    if not display:
        return
    h, w = frame.shape[:2]
    cx = w // 2
    dz = int(w / 2 * CENTER_DEAD_ZONE)
    cv2.line(frame, (cx, 0), (cx, h), (255, 255, 255), 1)
    cv2.rectangle(frame, (cx - dz, 0), (cx + dz, h), (255, 255, 255), 1)
    if ball:
        cv2.circle(frame, (ball['x'], ball['y']), ball['radius'], (0, 255, 0), 2)
        cv2.circle(frame, (ball['x'], ball['y']), 4, (0, 0, 255), -1)
    cv2.putText(frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0) if ball else (0, 0, 255), 2)
    cv2.imshow('XGO Tracking', frame)
    cv2.imshow('Green Mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        raise TrialAborted()


def recognize(robot, cam, stats, display):
    start = time.monotonic()
    consecutive = 0
    while time.monotonic() - start < RECOGNITION_TIMEOUT:
        frame = cam.read_fresh()
        if frame is None:
            time.sleep(0.05)
            continue
        ball, mask = timed_detect(frame, stats)
        if ball:
            consecutive += 1
            show(frame, mask, ball, f'Confirming {consecutive}/{CONFIRM_FRAMES}', display)
            if consecutive >= CONFIRM_FRAMES:
                print('Recognition: ball confirmed.')
                return True
        else:
            consecutive = 0
            show(frame, mask, None, 'Searching...', display)
            robot.pulse('left', TURN_SPEED, SEARCH_TURN_PULSE)
    return False


def approach(robot, cam, decider, stats, display):
    start = time.monotonic()
    last_seen = start
    last_error = 0.0
    decider.reset_memory()
    while time.monotonic() - start < APPROACH_TIMEOUT:
        frame = cam.read_fresh()
        if frame is None:
            time.sleep(0.05)
            continue
        ball, mask = timed_detect(frame, stats)
        h, w = frame.shape[:2]

        if ball is None:
            if timed_lost(decider, stats):
                print('Locomotion: ball passed below the camera - reached.')
                return True
            if time.monotonic() - last_seen > BALL_LOST_TIMEOUT:
                print('Locomotion: ball lost.')
                return False
            show(frame, mask, None, 'Ball lost - turning toward last side', display)
            robot.pulse('left' if last_error < 0 else 'right',
                        TURN_SPEED, SEARCH_TURN_PULSE)
            continue

        last_seen = time.monotonic()
        error = (ball['x'] - w / 2) / (w / 2)
        last_error = error
        low = low_signal(ball, h)
        decision = timed_decide(decider, error, ball['area'], stats, low)
        show(frame, mask, ball,
             f'err {error:+.2f} area {ball["area"]:.0f} low {low:.2f} -> {decision}',
             display)

        if decision == 'near':
            print('Locomotion: reached ball.')
            return True
        if decision in ('turn_left', 'turn_right'):
            deg = abs(error) * CAMERA_HFOV_DEG / 2
            dur = clamp(deg / TURN_DEG_PER_SEC, MIN_TURN_PULSE, MAX_TURN_PULSE)
            robot.pulse('left' if decision == 'turn_left' else 'right',
                        TURN_SPEED, dur)
        else:
            # Centred: walk continuously, checking every frame, and stop the
            # moment the decision layer says 'near' or the ball drifts.
            fast = ball['area'] < SLOW_AREA
            robot.start_walk(FORWARD_SPEED if fast else APPROACH_SPEED)
            walk_start = time.monotonic()
            reached = False
            try:
                while time.monotonic() - walk_start < MAX_WALK_TIME:
                    frame = cam.read_live()
                    if frame is None:
                        continue
                    ball, mask = timed_detect(frame, stats)
                    if ball is None:
                        # Vanished mid-walk: did it go out under the camera?
                        reached = timed_lost(decider, stats)
                        if reached:
                            print('Ball passed below the camera.')
                        break
                    last_seen = time.monotonic()
                    error = (ball['x'] - w / 2) / (w / 2)
                    last_error = error
                    low = low_signal(ball, h)
                    decision = timed_decide(decider, error, ball['area'], stats, low)
                    show(frame, mask, ball,
                         f'walking area {ball["area"]:.0f} low {low:.2f} -> {decision}',
                         display)
                    if decision != 'centered':
                        reached = decision == 'near'
                        break
                    if fast and ball['area'] >= SLOW_AREA:
                        break   # restart at the slower approach speed
            finally:
                robot.end_walk()
            if reached:
                print('Locomotion: reached ball.')
                return True

    print('Locomotion: timed out.')
    return False


def run_trial(trial_no, robot, cam, decider, display):
    stats = new_stats(decider.neuron_names)
    flags = {'recognized': False, 'reached': False,
             'picked_up': False, 'returned': False}
    failure = ''
    return_error = None
    battery = robot.battery()

    robot.prepare_for_trial()
    cpu0 = cpu_seconds()
    t0 = time.monotonic()

    try:
        flags['recognized'] = recognize(robot, cam, stats, display)
        if not flags['recognized']:
            failure = 'recognition'
        else:
            flags['reached'] = approach(robot, cam, decider, stats, display)
            if not flags['reached']:
                failure = 'locomotion'
            else:
                robot.grasp()
                flags['picked_up'] = ask_yes_no('Is the ball held in the claw?')
                if not flags['picked_up']:
                    failure = 'pickup'
                else:
                    robot.return_home()
                    flags['returned'] = ask_yes_no(
                        'Is the robot back at the start AND still holding the ball?')
                    return_error = ask_float(
                        'Distance from start marker in cm (Enter to skip): ')
                    if not flags['returned']:
                        failure = 'return'
    except TrialAborted:
        failure = 'aborted'
    finally:
        robot.stop()

    duration = time.monotonic() - t0
    cpu_used = cpu_seconds() - cpu0

    if failure == '':
        input('SUCCESS. Press Enter to open the claw and release the ball...')
        try:
            robot.dog.claw(CLAW_OPEN)
        except Exception:
            pass

    spike_energy = (stats['total_spikes'] * ENERGY_PER_SPIKE_J
                    if ENERGY_PER_SPIKE_J is not None else '')
    cpu_energy = (stats['decision_s'] * CPU_ACTIVE_POWER_W
                  if CPU_ACTIVE_POWER_W is not None else '')

    row = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'trial': trial_no,
        'mode': decider.name,
        'success': failure == '',
        'failure_stage': failure,
        **flags,
        'duration_s': round(duration, 2),
        'move_time_s': round(robot.move_time, 2),
        'frames': stats['frames'],
        'decisions': stats['decisions'],
        'vision_s': round(stats['vision_s'], 4),
        'decision_s': round(stats['decision_s'], 6),
        'process_cpu_s': round(cpu_used, 3),
        'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'total_spikes': stats['total_spikes'],
        'spikes_left': stats['spikes'].get('turn_left', ''),
        'spikes_right': stats['spikes'].get('turn_right', ''),
        'spikes_centered': stats['spikes'].get('centered', ''),
        'spikes_near': stats['spikes'].get('near', ''),
        'spikes_low': stats['spikes'].get('low', ''),
        'est_spike_energy_J': spike_energy,
        'est_decision_cpu_energy_J': cpu_energy,
        'return_error_cm': return_error if return_error is not None else '',
        'battery': battery if battery is not None else '',
    }
    return row


def append_log(row, path):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            writer.writeheader()
        writer.writerow(row)


def print_summary(rows):
    counted = [r for r in rows if r['failure_stage'] != 'aborted']
    if not counted:
        print('No completed trials.')
        return
    wins = sum(1 for r in counted if r['success'])
    fails = {s: sum(1 for r in counted if r['failure_stage'] == s) for s in STAGES}
    n = len(counted)
    print('\n================ SESSION SUMMARY ================')
    print(f'Mode: {counted[0]["mode"]}   Successful: {wins}/{n}')
    if n >= 10:
        passed = wins / n >= 0.7
        print('Criterion (>= 7 of 10):', 'MET' if passed else 'NOT MET')
    for s in STAGES:
        print(f'  failed at {s:<12}: {fails[s]}')
    if any(fails.values()):
        worst = max(fails, key=fails.get)
        print(f'Most frequent failure stage: {worst} -> focus next iteration here')
    avg_spikes = sum(r['total_spikes'] for r in counted) / n
    avg_dec = sum(r['decision_s'] for r in counted) / n
    print(f'Avg spikes/trial: {avg_spikes:.0f}   Avg decision time/trial: {avg_dec:.4f} s')
    print('=================================================')


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser(description='XGO retrieval trials')
    p.add_argument('--mode', choices=['neuromorphic', 'traditional'],
                   default='neuromorphic')
    p.add_argument('--trials', type=int, default=10)
    p.add_argument('--start', type=int, default=1, help='first trial number')
    p.add_argument('--log', default=LOG_FILE)
    p.add_argument('--no-display', action='store_true')
    args = p.parse_args()

    robot = Robot()
    try:
        if ENERGY_PER_SPIKE_J is None:
            print('Note: ENERGY_PER_SPIKE_J not set; energy column left blank.')

        decider = SpikingDecision() if args.mode == 'neuromorphic' else ThresholdDecision()
        cam = Camera(CAMERA_INDEX)
        display = not args.no_display
        rows = []
        try:
            for i in range(args.start, args.start + args.trials):
                input(f'\nTrial {i} ({decider.name}): place robot on START '
                      'and the ball in the arena, then press Enter...')
                row = run_trial(i, robot, cam, decider, display)
                append_log(row, args.log)
                rows.append(row)
                print(f'Trial {i}: '
                      f'{"SUCCESS" if row["success"] else "FAILED at " + row["failure_stage"]}'
                      f'  | spikes {row["total_spikes"]}  | {row["duration_s"]} s')
        except KeyboardInterrupt:
            print('\nSession stopped by user.')
        finally:
            cam.release()
            cv2.destroyAllWindows()
            print_summary(rows)
            print(f'Results appended to {args.log}')
    finally:
        robot.stop()


if __name__ == '__main__':
    main()
