import numpy as np
from typing import List, Dict, Tuple
from narwhals.selectors import matches
import config

class Track:
    # Аргументы: id объекта, bbox, номер кадра
    def __init__(self, track_id: int, bbox: List[float], frame_id: int):
        self.track_id = track_id
        self.bbox = bbox
        self.frame_id = frame_id
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.trajectory = [(frame_id, bbox)]
        self.center = self.calculate_center(bbox)

    # центр bbox
    def calculate_center(self, bbox: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    # обновление трека новой детекцией
    def update(self, bbox: List[float], frame_id: int):
        self.bbox = bbox
        self.frame_id = frame_id
        self.hits += 1
        self.age = 0
        self.time_since_update = 0
        self.center = self.calculate_center(bbox)
        self.trajectory.append((frame_id, bbox))

class HumanTracker:
    # Аргументы: кол-во кадров для потерянных треков, кол-во подтверждений для валидного трека
    def __init__(self, max_age: int = None, min_hits: int = None):
        self.max_age = max_age or config.MAX_AGE
        self.min_hits = min_hits or config.MIN_HITS
        self.tracks: List[Track] = []
        self.next_id = 1
        self.frame_id = 0
        self.completed_tracks: Dict[int, Dict] = {}

    # обновление трекера новыми детекциями, возвращает список треков
    def update(self, detections: List[Dict]) -> List[Dict]:
        self.frame_id += 1

        # увеличиваем возраст треков
        for track in self.tracks:
            track.age += 1
            track.time_since_update += 1

        # сопоставление детекции с текущими треками
        matched_tracks, unmatched_detections = self.match(detections)

        # обновляем сопоставленные треки
        for track_idx, det_idx in matched_tracks:
            bbox = detections[det_idx]['bbox']
            self.tracks[track_idx].update(bbox, self.frame_id)

        # создаем новые треки для несопоставленных детекций
        for det_idx in unmatched_detections:
            bbox = detections[det_idx]['bbox']
            new_track = Track(self.next_id, bbox, self.frame_id)
            self.tracks.append(new_track)
            self.next_id += 1

        tracks_to_remove = []
        for track in self.tracks:
            if track.time_since_update >= self.max_age:
                tracks_to_remove.append(track)

                # Сохраняем в историю перед удалением
                self.completed_tracks[track.track_id] = {
                    "track_id": track.track_id,
                    "bbox": track.bbox,
                    "center": track.center,
                    "trajectory": track.trajectory.copy(),
                    "hits": track.hits,
                    "age": track.age,
                    "start_frame": track.trajectory[0][0] if track.trajectory else 0,
                    "end_frame": track.trajectory[-1][0] if track.trajectory else 0
                }

        # удаление устаревших треков
        self.tracks = [
            track for track in self.tracks
            if track.time_since_update < self.max_age
        ]

        # возвращение подтвержденных треков
        confirmed_tracks = []
        for track in self.tracks:
            if track.hits >= self.min_hits:
                confirmed_tracks.append({
                    'track_id': track.track_id,
                    'bbox': track.bbox,
                    'center': track.center,
                    'trajectory': track.trajectory,
                    'hits': track.hits,
                    'age': track.age
                })
        return  confirmed_tracks

    # сопоставление детекций и треков о IoU
    def match(self, detections: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int]]:\

        if not self.tracks or not detections:
            return [], list(range(len(detections)))

        # матрица IoU
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self.calculate_iou(track.bbox, det['bbox'])

        # матчинг
        matched = []
        unmatched_detections = list(range(len(detections)))
        used_tracks = set()

        # сортируем кандидатов по IoU в порядке убывания
        matches = []
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                if iou_matrix[i, j] > 0.3:  # Порог IoU для сопоставления
                    matches.append((iou_matrix[i, j], i, j))

        matches.sort(reverse=True, key=lambda x: x[0])

        for iou, track_idx, det_idx in matches:
            if track_idx not in used_tracks and det_idx in unmatched_detections:
                matched.append((track_idx, det_idx))
                used_tracks.add(track_idx)
                unmatched_detections.remove(det_idx)
        return matched, unmatched_detections

    def calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Пересечение
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # объединение
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 -x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0  else 0.0

    def get_all_tracks(self) -> List[Dict]:
        return [
            {
                'track_id': track.track_id,
                'bbox': track.bbox,
                'center': track.center,
                'trajectory': track.trajectory,
                'hits': track.hits,
                'age': track.age
            }
            for track in self.tracks
            if track.hits >= self.min_hits
        ]

    def get_all_tracks_history(self) -> List[Dict]:
        # Собираем активные треки
        active_tracks = self.get_all_tracks()

        # Собираем завершённые треки из истории
        completed_tracks = list(self.completed_tracks.values())

        # Объединяем, убирая дубликаты (если трек ещё активен)
        all_tracks = []
        track_ids_seen = set()

        # Сначала активные
        for track in active_tracks:
            all_tracks.append(track)
            track_ids_seen.add(track["track_id"])

        # Затем завершённые (которые ещё не в списке)
        for track in completed_tracks:
            if track["track_id"] not in track_ids_seen:
                all_tracks.append(track)

        return all_tracks

    # сброс состояния трекера
    def reset(self):
        self.tracks = []
        self.next_id = 1
        self.frame_id = 0