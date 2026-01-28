import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import config

class TrajectoryVisualizer:

    def __init__(
        self,
        trajectory_length: int = None,
        bbox_color: Tuple[int, int, int] = None,
        trajectory_color: Tuple[int, int, int] = None,
        text_color: Tuple[int, int, int] = None
    ):

        self.trajectory_length = trajectory_length or config.TRAJECTORY_LENGTH
        self.bbox_color = bbox_color or config.BBOX_COLOR
        self.trajectory_color = trajectory_color or config.TRAJECTORY_COLOR
        self.text_color = text_color or config.TEXT_COLOR

    # рисует ббокс + подпись на кадре
    def draw_detection(
        self,
        frame: np.ndarray,
        bbox: List[float],
        track_id: Optional[int] = None,
        confidence: Optional[float] = None
    ) -> np.ndarray:

        x1, y1, x2, y2 = map(int, bbox)

        cv2.rectangle(frame, (x1, y1), (x2, y2), self.bbox_color, 2)

        label_parts = []
        if track_id is not None:
            label_parts.append(f"ID: {track_id}")
        if confidence is not None:
            label_parts.append(f"{confidence:.2f}")

        if label_parts:
            label = " | ".join(label_parts)
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            label_y = max(y1 - 10, label_size[1])

            cv2.rectangle(
                frame,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0], label_y + 5),
                self.bbox_color,
                -1
            )

            cv2.putText(
                frame,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.text_color,
                2
            )

        return frame

    # траектория по bbox на кадре
    def draw_trajectory(
        self,
        frame: np.ndarray,
        trajectory: List[Tuple[int, List[float]]],
        current_frame: int
    ) -> np.ndarray:

        if not trajectory:
            return frame

        # берем последние точки
        recent_trajectory = trajectory[-self.trajectory_length:]

        # точки стопы (нижний центр bbox)
        points = []
        for frame_id, bbox in recent_trajectory:
            x1, y1, x2, y2 = bbox
            center_x = int((x1 + x2) / 2)
            center_y = int(y2)
            points.append((center_x, center_y))

        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(
                    frame,
                    points[i -1],
                    points[i],
                    self.trajectory_color,
                    2
                )

        for point in points:
            cv2.circle(frame, point, 3, self.trajectory_color, -1)

        return frame

    # рисование трека на кадре
    def draw_track(
        self,
            frame: np.ndarray,
            track: Dict,
            current_frame: int
    ) -> np.ndarray:

        if 'trajectory' in track:
            frame = self.draw_trajectory(
                frame,
                track['trajectory'],
                current_frame
            )

        frame = self.draw_detection(
            frame,
            track['bbox'],
            current_frame
        )

        frame = self.draw_detection(
            frame,
            track['bbox'],
            track.get('track_id'),
            track.get('confidence')
        )

        return frame

    # трек на видео (bbox и нижняя точка)
    def draw_track_video_feed(
        self,
        frame: np.ndarray,
        track: Dict
    ) -> np.ndarray:

        bbox = track['bbox']
        track_id = track.get('track_id')

        # bbox
        frame = self.draw_detection(frame, bbox, track_id, None)

        # координаты нижней точки
        x1, y1, x2, y2 = bbox
        foot_x = int((x1 + x2) / 2)
        foot_y = int(y2)

        cv2.circle(frame, (foot_x, foot_y), 5, (0, 255, 255), -1)
        cv2.circle(frame, (foot_x, foot_y), 7, (0, 255, 255), -1)

        return frame

    # отрисовка траектории видом сверху
    def draw_trajectory_on_map(
        self,
        floor_map: np.ndarray,
        world_trajectory: List[Tuple[int, Tuple[float, float]]],
        track_id: Optional[int] = None,
        current_position: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:

        if not world_trajectory:
            return floor_map

        result = floor_map

        if track_id is not None:
            hue = (track_id * 137) % 180
            color_bgr = cv2.cvtColor(
                np.uint8([[[hue, 255, 255]]]),
                cv2.COLOR_HSV2RGB
            )[0][0]
            color = tuple(map(int, color_bgr))
        else:
            color = self.trajectory_color

        # перевод нормализованных координат в пиксели карты
        map_height, map_width = result.shape[:2]

        points = []

        for frame_id, (x_norm, y_norm) in world_trajectory:
            x_px = int(x_norm * map_width)
            y_px = int(y_norm * map_height)
            points.append((x_px, y_px))

        # линия траектории
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(
                    result,
                    points[i - 1],
                    points[i],
                    color,
                    2
                )

        # точки траектории
        for point in points:
            cv2.circle(result, point, 3, color, -1)

        # стартовая точка
        if points:
            cv2.circle(result, points[0], 6, (0, 255, 0), -1)
            cv2.circle(result, points[0], 8, (0, 255, 0), 2)

        # текущая позиция
        if current_position:
            x_norm, y_norm = current_position
            x_px = int(x_norm * map_width)
            y_px = int(y_norm * map_height)
            cv2.circle(result, (x_px, y_px), 8, (255, 0, 255), -1)
            cv2.circle(result, (x_px, y_px), 10, (255, 0, 255), 2)

        # подпись id у старта

        if track_id is not None and points:
            label = f"ID: {track_id}"
            cv2.putText(
                result,
                label,
                (points[0][0] + 10, points[0][1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        return result