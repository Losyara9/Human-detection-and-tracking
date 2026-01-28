import numpy as np
from typing import Tuple, List, Optional, Dict
from collections import deque

class CoordinateSmoother:

    def __init__(
        self,
        method: str = 'exponential',
        alpha: float = 0.3,
        window_size: int = 5
    ):

        self.method = method
        self.alpha = alpha
        self.window_size = window_size

        # предыдущее сглаженное значение для каждого track_id
        self.previous_smoothed: Dict[int, Tuple[float, float]] ={}

        # история точек для скользящего среднего по каждому track_id
        self.history: Dict[int, deque] = {}

    # сглаживает одну точку с учетом истории по track_id
    def smooth_point(
        self,
        point: Tuple[float, float],
        track_id: int
    ) -> Tuple[float, float]:

        if self.method == 'exponential':
            return self.exponential_smoothing(point, track_id)
        elif self.method == 'moving_average':
            return self.moving_average(point, track_id)
        else:
            return point

    def exponential_smoothing(
        self,
        point: Tuple[float, float],
        track_id: int
    ) -> Tuple[float, float]:

        if track_id not in self.previous_smoothed:
            self.previous_smoothed[track_id] = point # первая точка трека
            return point

        prev_x, prev_y = self.previous_smoothed[track_id]
        curr_x, curr_y = point

        smoothed_x = self.alpha * curr_x + (1 - self.alpha) * prev_x
        smoothed_y = self.alpha * curr_y + (1 - self.alpha) * prev_y

        smoothed = (smoothed_x, smoothed_y)
        self.previous_smoothed[track_id] = smoothed

        return smoothed

    def moving_average(
        self,
        point: Tuple[float, float],
        track_id: int
    ) -> Tuple[float, float]:

        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.window_size)

        self.history[track_id].append(point)

        # среднее значение по окну
        points_list = list(self.history[track_id])
        avg_x = np.mean([p[0] for p in points_list])
        avg_y = np.mean([p[1] for p in points_list])

        return (avg_x, avg_y)

    def reset_track(self, track_id: int):
        if track_id in self.previous_smoothed:
            del self.previous_smoothed[track_id]
        if track_id in self.history:
            del self.history[track_id]

    def reset_all(self):
        self.previous_smoothed.clear()
        self.history.clear()