import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple

class CameraCalibrator:
    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibrated = False

    # калибровка камеры по шахматной доске
    # аргументы: список точек на избб соответствующие точки в реальности, размер изображения
    def calibrate(
            self,
            image_points: List[List[Tuple[float, float]]],
            object_points: List[List[Tuple[float, float, float]]],
            image_size: Tuple[int, int]
    ) -> bool:

        try:
            obj_pts = np.array(object_points, dtype=np.float32)
            img_pts = np.array(image_points, dtype=np.float32)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                obj_pts,
                img_pts,
                image_size,
                None,
                None
            )

            if ret:
                self.camera_matrix = mtx
                self.dist_coeffs = dist
                self.calibrated = True
                return True
            return False
        except Exception as e:
            print(f"Ошибка калибровки: {e}")
            return False

    # убирает дисторсию с изображения по параметрам калибровки
    def undistort(self, image: np.ndarray) -> np.ndarray:
        if not self.calibrated:
            return image

        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeffs,
            (w, h),
            1,
            (w, h)
        )

        undistorted = cv2.undistort(
            image,
            self.camera_matrix,
            self.dist_coeffs,
            None,
            new_camera_matrix
        )

        return undistorted

    # сохранение параметров калибровки в файл
    def save(self, filepath: Path):
        if not self.calibrated:
            raise ValueError('Камера еще не откалибрована')

        data = {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: Path):
        with open(filepath, 'r') as f:
            data= json.load(f)

        self.camera_matrix = np.array(data['camera_matrix'])
        self.dist_coeffs = np.array(data['dist_coeffs'])
        self.calibrated = True