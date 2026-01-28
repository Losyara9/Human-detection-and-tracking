import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional
import config

class HumanDetector:
    def __init__(
            self,
            model_name: str = None,
            confidence_threshold: float = None,
            iou_threshold: float = None,
            roi_points: Optional[List[Tuple[float, float]]] = None,
            debug: bool = None
    ):
    # Аргументы: имя/путь к модели
    # Порог уверенности детекции, iou порог для nms
    # roi-полигон (4 точки), отладка для фильтрации

        self.model_name = model_name or config.MODEL_NAME
        self.confidence_threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD
        self.iou_threshold = iou_threshold or config.IOU_THRESHOLD
        self.roi_points = roi_points
        self.debug = debug if debug is not None else config.DEBUG_DETECTIONS

        # Счетчик кадров для статистики
        self.frame_count = 0

        # Загрузка модели
        self.model = YOLO(self.model_name)

        print('Фильтры детекции:')
        print(f"  - Размер bbox: {config.MIN_BBOX_HEIGHT_RATIO * 100:.1f}% высоты кадра, {config.MIN_BBOX_WIDTH_RATIO * 100:.1f}% ширины кадра")
        print(f"  - Отношение сторон: {config.MIN_ASPECT_RATIO:.1f} – {config.MAX_ASPECT_RATIO:.1f}")
        print(f"  - ROI: {'включен' if self.roi_points else 'выключен'} (запас: {config.ROI_MARGIN_PIXELS}px)")
        if self.debug:
            print("  - Отладка: ВКЛ")

    # Основная функция детекции
    # Передается кадр в качестве аргумента, возвращает координаты ббокса
    # Уверенность и класс
    def detect(self, frame: np.ndarray) -> List[Dict]:
        self.frame_count += 1
        # Размер кадра
        frame_height, frame_width = frame.shape[:2]

        # Запуск YOLO
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=config.CLASS_IDS,
            verbose=False
        )

        raw_detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                # Фильтрация только по классу person
                if class_id != 0:
                    continue
                # Сохраняем сырые детекции
                raw_detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': confidence,
                    'class_id': class_id
                })

        # Фильтры после детекции
        filtered_detections = []
        for det in raw_detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1

            # фильтра размера bboxa (относительно кадра)
            if not self.size_filter(width, height, frame_width, frame_height, det):
                continue
            # фильтр отношения сторон bboxa
            if not self.aspect_ratio_filter(width, height, det):
                continue
            # фильтр roi
            if not self.roi_filter(bbox, det):
                continue

            filtered_detections.append(det)
        # фильтр nms (убираем перекрывающие детекции)
        final_detections = self.apply_nms(filtered_detections)

        return final_detections

    def size_filter(self, width: float, height: float, frame_width: int, frame_height: int, det: Dict) -> bool:
        min_width = max(
            frame_width * config.MIN_BBOX_WIDTH_RATIO,
            config.MIN_BBOX_WIDTH_ABS
        )
        min_height = max(
            frame_height * config.MIN_BBOX_HEIGHT_RATIO,
            config.MIN_BBOX_HEIGHT_ABS
        )

        if width < min_width or height < min_height:
            if self.debug:
                print(f"  Отброшено (размер): {int(width)}x{int(height)} < {int(min_width)}x{int(min_height)}")
            return False
        return True

    def aspect_ratio_filter(self, width: float, height: float, det: Dict) -> bool:
        if width == 0:
            return False
        aspect_ratio = height / width

        if aspect_ratio < config.MIN_ASPECT_RATIO or aspect_ratio > config.MAX_ASPECT_RATIO:
            if self.debug:
                print(f"  Отброшено (ratio): {aspect_ratio:.2f} не в [{config.MIN_ASPECT_RATIO}, {config.MAX_ASPECT_RATIO}]")
            return False
        return True

    # Оставляем детекции внутри Roi с запасом
    def roi_filter(self, bbox: List[float], det: Dict) -> bool:
        if not config.ENABLE_ROI_FILTER or self.roi_points is None:
            return True

        confidence = det.get('confidence', 1.0)

        # если настроено, применяем roi только к низкой уверенности
        if config.ROI_ONLY_LOW_CONFIDENCE and confidence >= config.ROI_CONFIDENCE_THRESHOLD:
            return True

        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y =(y2) # нижний центр (отслеживаем по ногам)

        # расширяем roi-полигон наружу
        roi_array = np.array(self.roi_points, dtype=np.float32)

        centroid = np.mean(roi_array, axis=0)

        # Расширение полигона
        expanded_roi = roi_array.copy()
        for i in range(len(expanded_roi)):
            direction = expanded_roi[i] - centroid
            norm = np.linalg.norm(direction)
            if norm > 0:
                expanded_roi[i] = expanded_roi[i] + (direction / norm) * config.ROI_MARGIN_PIXELS

        expanded_roi_int = expanded_roi.astype(np.int32)
        result = cv2.pointPolygonTest(expanded_roi_int, (center_x, center_y), False)

        if result < 0: # снаружи
            if self.debug:
                print((f"  Отброшено (ROI): точка ({int(center_x)}, {int(center_y)}) вне ROI (+{config.ROI_MARGIN_PIXELS}px)"))
            return False
        return True

    def apply_nms(self, detections: List[Dict]) -> List[Dict]:
        if len(detections) == 0:
            return detections

        boxes_xywh: List[List[int]] = []
        scores: List[float] = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            boxes_xywh.append([int(x1), int(y1), int(w), int(h)])
            scores.append(float(det.get('confidence', 0.0)))

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            scores,
            self.confidence_threshold,
            self.iou_threshold
        )

        if len(indices) == 0:
            return []

        if isinstance(indices, np.ndarray):
            idxs = indices.flatten().tolist()
        else:
            idxs = [int(i[0]) if isinstance(i, (list, tuple)) else int(i) for i in indices]

        return [detections[i] for i in idxs]

    def set_roi(self, roi_points: List[Tuple[float, float]]):
        if len(roi_points) != 4:
            raise ValueError('ROI должен содержать ровно 4 точки')
        self.roi_points = roi_points
        print(f"ROI‑фильтр включен, точек: {len(roi_points)}")

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        all_detections = []
        for frame in frames:
            detections = self.detect(frame)
            all_detections.append(detections)
        return all_detections