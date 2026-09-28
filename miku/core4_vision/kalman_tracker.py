"""
Object Tracking with 2D Kalman Filter.
Core 4: Visual Perception Daemon (Eyes - v1.2.0)
Architecture §4.3: Radar-grade smooth object state estimation across occlusions.
State vector: x = [pos_x, pos_y, vel_x, vel_y]^T
"""
import numpy as np
from typing import Tuple, Optional

class KalmanObjectTracker:
    def __init__(self, dt: float = 1.0, q_var: float = 1e-4, r_var: float = 1e-2):
        self.dt = dt
        # State vector: [px, py, vx, vy]
        self.x = np.zeros((4, 1), dtype=np.float64)

        # Transition matrix F: constant velocity kinematics
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Measurement matrix H: observe position only
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)

        # Process noise covariance Q
        self.Q = np.eye(4, dtype=np.float64) * q_var
        # Measurement noise covariance R
        self.R = np.eye(2, dtype=np.float64) * r_var
        # Error covariance P
        self.P = np.eye(4, dtype=np.float64) * 1.0

        self.initialized = False

    def initialize(self, init_pos: Tuple[float, float]):
        self.x[0, 0] = init_pos[0]
        self.x[1, 0] = init_pos[1]
        self.x[2, 0] = 0.0
        self.x[3, 0] = 0.0
        self.P = np.eye(4, dtype=np.float64) * 1.0
        self.initialized = True

    def predict(self) -> Tuple[float, float]:
        """
        Predict step:
        x_hat_minus = F * x
        P_minus = F * P * F^T + Q
        """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        """
        Update step with new measurement z = [pos_x, pos_y]^T:
        K = P_minus * H^T * inv(H * P_minus * H^T + R)
        x = x_hat_minus + K * (z - H * x_hat_minus)
        P = (I - K * H) * P_minus
        """
        if not self.initialized:
            self.initialize(measurement)
            return measurement

        z = np.array([[measurement[0]], [measurement[1]]], dtype=np.float64)
        y = z - np.dot(self.H, self.x)  # Innovation / residual

        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)
        I = np.eye(4, dtype=np.float64)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def position(self) -> Tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.x[2, 0]), float(self.x[3, 0])
