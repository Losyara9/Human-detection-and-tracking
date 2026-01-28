import cv2
import numpy as np
from typing import Tuple, List, Optional
import config

class FloorMapGenerator:

    def __init__(
        self,
        width_meters: float = None, # ширина расширенной карты
        height_meters: float = None, # высота расширенной карты
        pixels_per_meter: int = None, # разрешение карты
        calibration_width: float = None, # ширина зоны калибровки
        calibration_height: float = None # высота зоны калибровки
    ):

        self.width_meters = width_meters or config.FLOOR_MAP_WIDTH_METERS
        self.height_meters = height_meters or config.FLOOR_MAP_HEIGHT_METERS
        self.pixels_per_meter = pixels_per_meter or config.FLOOR_MAP_PIXELS_PER_METER
        self.calibration_width = calibration_width or config.CALIBRATION_AREA_WIDTH
        self.calibration_height = calibration_height or config.CALIBRATION_AREA_HEIGHT

        self.map_width = int(self.width_meters * self.pixels_per_meter)
        self.map_height = int(self.height_meters * self.pixels_per_meter)

        # смещение, чтобы зона калибровки была по центру расширенной карты
        self.offset_x = (self.width_meters - self.calibration_width) / 2
        self.offset_y = (self.height_meters - self.calibration_height) / 2

        # точки калибровки в 'мировых' координатах
        tl_x = self.offset_x / self.width_meters
        tl_y = self.offset_y / self.height_meters
        tr_x = (self.offset_x + self.calibration_width) / self.width_meters
        tr_y = self.offset_y / self.height_meters
        br_x = (self.offset_x + self.calibration_width) / self.width_meters
        br_y = (self.offset_y + self.calibration_height) / self.height_meters
        bl_x = self.offset_x / self.width_meters
        bl_y = (self.offset_y + self.calibration_height) / self.height_meters

        self.calibration_points_world = [
            (tl_x, tl_y),
            (tr_x, tr_y),
            (br_x, br_y),
            (bl_x, bl_y)
        ]

        self.calibration_points_image: List[Tuple[float, float]] = []

    # генерация карты пола с сеткой и подписями
    def generate_map(self) -> np.ndarray:

        # белый фон
        floor_map = np.ones((self.map_height, self.map_width, 3), dtype=np.uint8) * 255

        # сетка
        grid_spacing_01m = int(0.1 * self.pixels_per_meter)
        grid_spacing_05m = int(0.5 * self.pixels_per_meter)

        # вертикальные линии
        for x in range(0, self.map_width + 1, grid_spacing_01m):
            thickness = 2 if x % grid_spacing_05m == 0 else 1
            color = (200, 200, 200) if x % grid_spacing_05m == 0 else (230, 230, 230)
            cv2.line(floor_map, (x, 0), (x, self.map_height), color, thickness)

        # горизонтальные линии
        for y in range(0, self.map_height + 1, grid_spacing_01m):
            thickness = 2 if y % grid_spacing_05m == 0 else 1
            color = (200, 200, 200) if y % grid_spacing_05m == 0 else (230, 230, 230)
            cv2.line(floor_map, (0, y), (self.map_width, y), color, thickness)

        # подписи осей (в метрах)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        label_color = (100, 100, 100)

        # подписи x (внизу)
        for x_m in np.arange(0, self.width_meters + 0.1, 0.1):
            x_px = int(x_m * self.pixels_per_meter)
            if x_m % 0.5 == 0:
                label = f"{x_m:.1f}m"
                text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                cv2.putText(
                    floor_map,
                    label,
                    (x_px - text_size[0] // 2, self.map_height -5),
                    font,
                    font_scale,
                    label_color,
                    font_thickness
                )

        # подписи y (слева)
        for y_m in np.arange(0, self.height_meters + 0.1, 0.1):
            y_px = int(y_m * self.pixels_per_meter)
            if y_m % 0.5 == 0:
                label = f"{y_m:.1f}m"
                text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                cv2.putText(
                    floor_map,
                    label,
                    (5, self.map_height - y_px + text_size[1] // 2),
                    font,
                    font_scale,
                    label_color,
                    font_thickness
                )

        # отрисовка прямоугольника зоны калибровки (если точки заданы)
        if len(self.calibration_points_image) == 4:
            # контур зоны калибровки
            tl_px = int(self.offset_x * self.pixels_per_meter)
            tl_py = int(self.offset_y * self.pixels_per_meter)
            br_px = int((self.offset_x + self.calibration_width) * self.pixels_per_meter)
            br_py = int((self.offset_y + self.calibration_height) * self.pixels_per_meter)

            # прямоугольник калибровки
            cv2.rectangle(floor_map, (tl_px, tl_py), (br_px, br_py), (0, 0, 255), 2)

            # маркеры точек калибровки
            for i, (wx, wy) in enumerate(self.calibration_points_world):
                px = int(wx * self.map_width)
                py = int(wy * self.map_height)

                cv2.circle(floor_map, (px, py), 8, (0, 0, 255), -1)
                cv2.circle(floor_map, (px, py), 10, (0, 0, 255), 2)

                labels = ['TL', 'TR', 'BR', 'BL']
                cv2.putText(
                    floor_map,
                    labels[i],
                    (px + 12, py),
                    font,
                    0.5,
                    (0, 0, 255),
                    2
                )

        return floor_map

    # перевод метров в пиксели
    def world_to_pixel(self, x_meters: float, y_meters: float) -> Tuple[int, int]:

        x_px = int(x_meters * self.pixels_per_meter)
        y_px = int(y_meters * self.pixels_per_meter)

        return x_px, y_px

    # перевод нормализованных координат в пиксели
    def normalized_to_pixel(self, x_norm: float, y_norm: float) -> Tuple[int, int]:

        x_px = int(x_norm * self.map_width)
        y_px = int(y_norm * self.map_height)

        return x_px, y_px

    def set_calibration_points(self, image_points: List[Tuple[float, float]]):

        if len(image_points) != 4:
            raise ValueError('Должно быть 4 точки для калибровки')
        self.calibration_points_image = image_points

# интерактивная калибровка (клик по 4 маркерам на первом кадре)
def interactive_calibration(frame: np.ndarray) -> Optional[List[Tuple[float, float]]]:

    points = []
    current_point = 0
    point_labels = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']

    display_frame = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal  current_point, points, display_frame

        if event == cv2.EVENT_LBUTTONDOWN:
            if current_point < 4:
                points.append((float(x), float(y)))

                # отрисовка точек
                cv2.circle(display_frame, (x, y), 10, (0, 255, 0), -1)
                cv2.circle(display_frame, (x, y), 12, (0, 255, 0), 2)

                # названия точек
                label = f"{point_labels[current_point]} ({current_point + 1}/4)"
                cv2.putText(
                    display_frame,
                    label,
                    (x + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                current_point += 1

                # отрисовка соединительных линий
                if len(points) > 1:
                    cv2.line(display_frame,
                             (int(points[-2][0]), int(points[-2][1])),
                             (int(points[-1][0]), int(points[-1][1])),
                             (0, 255, 0), 2)

                # закрытие прямоугольника если выбрано 4 точки
                if len(points) == 4:
                    cv2.line(display_frame,
                             (int(points[0][0]), int(points[0][1])),
                             (int(points[-1][0]), int(points[-1][1])),
                             (0, 255, 0), 2)

    window_name = 'Calibration - Click 4 White Markers'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    instructions = [
        "Click on 4 markers in order:",
        "1. Top-left (TL)",
        "2. Top-right (TR)",
        "3. Bottom-right (BR)",
        "4. Bottom-left (BL)",
        "",
        "R — reset, ESC — cancel, ENTER — done"
    ]

    print("\n" + "=" * 50)
    print("ИНТЕРАКТИВНАЯ КАЛИБРОВКА")
    print("=" * 50)
    print("Кликните 4 маркера в порядке:")
    print("  1. TL (верх‑лево)")
    print("  2. TR (верх‑право)")
    print("  3. BR (низ‑право)")
    print("  4. BL (низ‑лево)")
    print("\nR — сброс, ESC — отмена, ENTER — готово")
    print("=" * 50 + "\n")

    while True:
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("Калибровка отменена")
            cv2.destroyWindow(window_name)
            return None

        if key == ord('r') or key == ord('R'):  # Reset
            points = []
            current_point = 0
            display_frame = frame.copy()
            # Redraw instructions
            y_offset = 30
            for i, instruction in enumerate(instructions):
                cv2.putText(
                    display_frame,
                    instruction,
                    (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    display_frame,
                    instruction,
                    (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1
                )
            print("Сброс калибровки. Кликните точки заново.")

        if key == 13 or key == 10:  # ENTER
            if len(points) == 4:
                print(f"Калибровка завершена! Точки: {points}")
                cv2.destroyWindow(window_name)
                return points
            else:
                print(f"Нужно 4 точки. Сейчас: {len(points)}/4")

    cv2.destroyWindow(window_name)
    return None


