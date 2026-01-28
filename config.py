import os
from pathlib import Path

# Базовые пути
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
FLOOR_PLANS_DIR = DATA_DIR / "floor_plans"
OUTPUT_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

# Создаём директории, если их нет
for directory in [DATA_DIR, VIDEOS_DIR, FLOOR_PLANS_DIR, OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Настройки модели
_DEFAULT_YOLO = DATA_DIR / "yolo" / "yolov8n.pt"
MODEL_PATH = os.environ.get("MODEL_PATH") or (str(_DEFAULT_YOLO) if _DEFAULT_YOLO.exists() else None)
MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
CLASS_IDS = [0]  # Класс person

# Пост‑фильтры детекции
# Фильтр размера (относительно размера кадра)
MIN_BBOX_HEIGHT_RATIO = 0.03  # Мин. высота как доля высоты кадра (3%)
MIN_BBOX_WIDTH_RATIO = 0.02   # Мин. ширина как доля ширины кадра (2%)
MIN_BBOX_HEIGHT_ABS = 50      # Абсолютный минимум высоты (пикс)
MIN_BBOX_WIDTH_ABS = 30       # Абсолютный минимум ширины (пикс)

# Фильтр отношения сторон bbox
MIN_ASPECT_RATIO = 0.7   # Мин. (height/width)
MAX_ASPECT_RATIO = 5.0   # Макс. (height/width)

# Фильтр ROI вокруг калибровочного прямоугольника
ENABLE_ROI_FILTER = True        # Включить ROI‑фильтр
ROI_MARGIN_PIXELS = 20          # Запас (пикс) вокруг ROI‑полигона
ROI_ONLY_LOW_CONFIDENCE = True  # Применять ROI только к низкой уверенности
ROI_CONFIDENCE_THRESHOLD = 0.7  # Порог уверенности для ROI‑фильтра


TEMPORAL_MIN_FRAMES = 1  # В детекторе не используется

# Отладка
DEBUG_DETECTIONS = False   # Печатать причины отбраковки детекций
VISUALIZE_FILTERS = False

# Настройки трекинга
TRACKING_METHOD = "bytetrack"
MAX_AGE = 30  # Сколько кадров хранить "потерянный" трек
MIN_HITS = 3  # Сколько подтверждений нужно для "валидного" трека

# Настройки видео (дефолты)
VIDEO_FPS = 30
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

# Калибровка/гомография
CALIBRATION_FILE = DATA_DIR / "calibration.json"
HOMOGRAPHY_FILE = DATA_DIR / "homography.json"

# Настройки API
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Human Detection and Tracking API"
API_VERSION = "1.0.0"

# Визуализация
TRAJECTORY_COLOR = (0, 255, 0)     # зелёный
BBOX_COLOR = (255, 0, 0)           # синий (BGR)
TEXT_COLOR = (255, 255, 255)       # белый
TRAJECTORY_LENGTH = 50             # сколько точек траектории показывать

# Сглаживание координат точки стопы
SMOOTHING_METHOD = "exponential"  # варианты: "exponential", "moving_average", "none"
SMOOTHING_ALPHA = 0.3             # коэффициент экспоненциального сглаживания (чем меньше — тем плавнее)
SMOOTHING_WINDOW_SIZE = 5         # окно для скользящего среднего

# Настройки 2D‑карты пола (вид сверху)
FLOOR_MAP_WIDTH_METERS = 2.0    # расширенная ширина (1.0м + 50% слева/справа)
FLOOR_MAP_HEIGHT_METERS = 2.25  # расширенная высота (1.5м + 50% сверху/снизу)
FLOOR_MAP_PIXELS_PER_METER = 200
CALIBRATION_AREA_WIDTH = 1.0    # исходная ширина зоны калибровки
CALIBRATION_AREA_HEIGHT = 1.5   # исходная высота зоны калибровки

# Экспорт
EXPORT_FORMATS = ["xlsx", "json", "pdf"]