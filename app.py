from __future__ import annotations

import asyncio
import base64
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from contourpy import contour_generator
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from core.detector import HumanDetector
from core.tracker import HumanTracker
from utils.calibration import CameraCalibrator
from utils.export import TrajectoryExporter
from utils.floor_map import FloorMapGenerator
from utils.homography import HomographyTransformer
from utils.smoothing import CoordinateSmoother
from utils.visualization import TrajectoryVisualizer

camera_calibrator: Optional[Any] = None

app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description='API for human detection and tracking with 2D trajectory visualization'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

web_dir = Path(__file__).parent / 'web'
if web_dir.exists():
    app.mount('/static', StaticFiles(directory=str(web_dir)), name='static')

# история обработок (SQLite)

DB_PATH = config.DATA_DIR / 'web_history.sqlite3'

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def db_init() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                filename TEXT,
                status TEXT,
                created_at TEXT,
                started_at TEXT,
                finished_at TEXT,
                error TEXT,
                frames_processed INTEGER,
                video_fps REAL,
                processing_fps REAL,
                people_max INTEGER,
                calibration_points_json TEXT,
                calibration_has_real_scale BOOLEAN DEFAULT 0,
                calibration_real_width REAL,
                calibration_real_height REAL,
                input_path TEXT,
                output_feed_path TEXT,
                output_map_path TEXT,
                tracks_json_path TEXT,
                tracks_xlsx_path TEXT,
                report_pdf_path TEXT
            )
            """
        )

def db_upsert(job: "JobRuntime") -> None:
    with db() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, filename, status, created_at, started_at, finished_at, error,
                frames_processed, video_fps, processing_fps, people_max,
                calibration_points_json,
                calibration_has_real_scale, calibration_real_width, calibration_real_height,
                input_path, output_feed_path, output_map_path,
                tracks_json_path, tracks_xlsx_path, report_pdf_path
            ) VALUES (
                :job_id, :filename, :status, :created_at, :started_at, :finished_at, :error,
                :frames_processed, :video_fps, :processing_fps, :people_max,
                :calibration_points_json,
                :calibration_has_real_scale, :calibration_real_width, :calibration_real_height,
                :input_path, :output_feed_path, :output_map_path,
                :tracks_json_path, :tracks_xlsx_path, :report_pdf_path
            )
            ON CONFLICT(job_id) DO UPDATE SET
                filename = excluded.filename,
                status = excluded.status,
                started_at = excluded.started_at,
                finished_at = excluded.finished_at,
                error = excluded.error,
                frames_processed = excluded.frames_processed,
                video_fps = excluded.video_fps,
                processing_fps = excluded.processing_fps,
                people_max = excluded.people_max,
                calibration_points_json = excluded.calibration_points_json,
                calibration_has_real_scale = excluded.calibration_has_real_scale,
                calibration_real_width = excluded.calibration_real_width,
                calibration_real_height = excluded.calibration_real_height,
                input_path = excluded.input_path,
                output_feed_path = excluded.output_feed_path,
                output_map_path = excluded.output_map_path,
                tracks_json_path = excluded.tracks_json_path,
                tracks_xlsx_path = excluded.tracks_xlsx_path,
                report_pdf_path = excluded.report_pdf_path
            """,
            {
                "job_id": job.job_id,
                "filename": job.filename,
                "status": job.status,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "error": job.error,
                "frames_processed": job.frames_processed,
                "video_fps": job.video_fps,
                "processing_fps": job.processing_fps,
                "people_max": job.people_max,
                "calibration_points_json": json.dumps(job.calibration_points) if job.calibration_points else None,
                "calibration_has_real_scale": getattr(job, 'calibration_has_real_scale', False),
                "calibration_real_width": getattr(job, 'calibration_real_width', None),
                "calibration_real_height": getattr(job, 'calibration_real_height', None),
                "input_path": str(job.input_path) if job.input_path else None,
                "output_feed_path": str(job.output_feed_path) if job.output_feed_path else None,
                "output_map_path": str(job.output_map_path) if job.output_map_path else None,
                "tracks_json_path": str(job.tracks_json_path) if job.tracks_json_path else None,
                "tracks_xlsx_path": str(job.tracks_xlsx_path) if job.tracks_xlsx_path else None,
                "report_pdf_path": str(job.report_pdf_path) if job.report_pdf_path else None,
            },
        )

def db_list_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]

def db_get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return dict(row) if row else None

# менеджер job'ов

@dataclass
class JobRuntime:
    job_id: str
    filename: str
    created_at: str
    status: str = "uploaded"  # uploaded|calibrated|processing|done|error (состояния job)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None

    input_path: Optional[Path] = None
    job_dir: Optional[Path] = None
    frame0_path: Optional[Path] = None

    calibration_points: Optional[List[Tuple[float, float]]] = None
    # поля для хранения информации о реальных размерах
    calibration_has_real_scale: bool = False
    calibration_real_width: Optional[float] = None
    calibration_real_height: Optional[float] = None
    calibration_scale_info: Optional[Dict[str, Any]] = None

    # Выходные файлы
    output_feed_path: Optional[Path] = None
    output_map_path: Optional[Path] = None
    tracks_json_path: Optional[Path] = None
    tracks_xlsx_path: Optional[Path] = None
    report_pdf_path: Optional[Path] = None

    # Статистика
    frames_processed: int = 0
    video_fps: float = 0.0
    processing_fps: float = 0.0
    people_max: int = 0

    # Подключённые WebSocket‑клиенты
    clients: List[WebSocket] = field(default_factory=list)

jobs: Dict[str, JobRuntime] = {}
jobs_lock = asyncio.Lock()

async def broadcast(job: JobRuntime, payload: Dict[str, Any]) -> None:
    if not job.clients:
        return
    dead: List[WebSocket] = []
    for ws in job.clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    if dead:
        for ws in dead:
            try:
                job.clients.remove(ws)
            except ValueError:
                pass

def encode_jpeg_b64(img_bgr: np.ndarray, quality: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")

class CalibrateRequest(BaseModel):
    points: List[Dict[str, float]]  # [{x:..., y:...} x4]
    knows_real_dimensions: bool = False  # знает ли пользователь реальные размеры
    real_width_m: Optional[float] = None  # реальная ширина в метрах
    real_height_m: Optional[float] = None  # реальная высота в метрах

@app.on_event("startup")
async def startup_event() -> None:
    global camera_calibrator
    db_init()
    app.state.loop = asyncio.get_running_loop()
    # Опциональная калибровка камеры для устранения дисторсии (по шахматной доске).
    # Файл data/calibration.json создаётся отдельно (скрипт или вручную по OpenCV).
    if config.CALIBRATION_FILE.exists():
        try:
            cal = CameraCalibrator()
            cal.load(config.CALIBRATION_FILE)
            camera_calibrator = cal
            print(f"[Calibration] Загружена калибровка дисторсии из {config.CALIBRATION_FILE}")
        except Exception as e:
            print(f"[Calibration] Не удалось загрузить {config.CALIBRATION_FILE}: {e}")
    else:
        camera_calibrator = None
        print("[Calibration] Файл калибровки дисторсии не найден — обработка без устранения дисторсии")

# базовые endpoin'ы

@app.get("/", response_class=JSONResponse)
async def root() -> Dict[str, Any]:
    return {
        "message": "Human Detection and Tracking API",
        "version": config.API_VERSION,
        "ui": "/static/index.html",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health", response_class=JSONResponse)
async def health_check() -> Dict[str, Any]:
    return {"status": "healthy"}

@app.get("/ui", response_class=HTMLResponse)
async def ui_redirect() -> str:
    # Простой редирект на страницу панели
    return '<meta http-equiv="refresh" content="0; url=/static/index.html" />'

# детекция

@app.post("/api/v1/detect", response_class=JSONResponse)
async def detect_humans(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        contents = await file.read()
        arr = np.frombuffer(contents, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        det = HumanDetector(
            model_name=(config.MODEL_PATH or config.MODEL_NAME),
            confidence_threshold=config.CONFIDENCE_THRESHOLD,
            iou_threshold=config.IOU_THRESHOLD,
            roi_points=None,
            debug=config.DEBUG_DETECTIONS,
        )
        detections = det.detect(img)
        return {"status": "success", "detections": detections}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# job api

@app.post("/api/v1/jobs", response_class=JSONResponse)
async def create_job(file: UploadFile = File(...)) -> Dict[str, Any]:
    job_id = uuid.uuid4().hex
    job_dir = config.OUTPUT_DIR / "web_jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # сохранение загруженного видео
    suffix = Path(file.filename).suffix or ".mp4"
    input_path = job_dir / f"input{suffix}"
    contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(contents)

    # достаем первый кадр для калибровки
    cap = cv2.VideoCapture(str(input_path))
    ok, frame0 = cap.read()
    cap.release()
    if not ok or frame0 is None:
        raise HTTPException(status_code=400, detail="Could not read video / first frame")
    if camera_calibrator is not None:
        frame0 = camera_calibrator.undistort(frame0)

    frame0_path = job_dir / "frame0.jpg"
    cv2.imwrite(str(frame0_path), frame0)

    job = JobRuntime(
        job_id=job_id,
        filename=file.filename,
        created_at=utc_now_iso(),
        status="uploaded",
        input_path=input_path,
        job_dir=job_dir,
        frame0_path=frame0_path,
    )

    async with jobs_lock:
        jobs[job_id] = job

    db_upsert(job)

    return {
        "job_id": job_id,
        "status": job.status,
        "frame0_url": f"/api/v1/jobs/{job_id}/frame0",
        "status_url": f"/api/v1/jobs/{job_id}",
        "ws_url": f"/ws/jobs/{job_id}",
    }

@app.post("/upload", response_class=JSONResponse)
async def upload_alias(file: UploadFile = File(...)) -> Dict[str, Any]:
    return await create_job(file)


@app.post("/api/v1/upload", response_class=JSONResponse)
async def api_upload_alias(file: UploadFile = File(...)) -> Dict[str, Any]:
    return await create_job(file)

@app.post("/api/v1/track", response_class=JSONResponse)
async def api_track_alias(file: UploadFile = File(...)) -> Dict[str, Any]:
    return await create_job(file)

@app.get("/api/v1/jobs/{job_id}", response_class=JSONResponse)
async def get_job(job_id: str) -> Dict[str, Any]:
    async with jobs_lock:
        job = jobs.get(job_id)
    if job:
        return {
            "job_id": job.job_id,
            "filename": job.filename,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "error": job.error,
            "frames_processed": job.frames_processed,
            "people_max": job.people_max,
            "downloads": job_downloads(job),
        }

    row = db_get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    row["downloads"] = job_downloads_from_row(row)
    return row

@app.get("/api/v1/jobs/{job_id}/frame0")
async def get_frame0(job_id: str) -> FileResponse:
    async with jobs_lock:
        job = jobs.get(job_id)
    if not job or not job.frame0_path or not job.frame0_path.exists():
        raise HTTPException(status_code=404, detail="Frame0 not found")
    return FileResponse(str(job.frame0_path), media_type="image/jpeg")

@app.post("/api/v1/jobs/{job_id}/calibrate", response_class=JSONResponse)
async def calibrate_job(job_id: str, req: CalibrateRequest) -> Dict[str, Any]:
    pts = [(float(p["x"]), float(p["y"])) for p in req.points]
    if len(pts) != 4:
        raise HTTPException(status_code=400, detail="Need exactly 4 points")

    # проверка введенных размеров
    knows_real = req.knows_real_dimensions
    real_width = req.real_width_m
    real_height = req.real_height_m

    if knows_real:
        if not real_width or not real_height:
            raise HTTPException(
                status_code=400,
                detail="Real dimensions required when knows_real_dimensions is True"
            )
        if real_width <= 0 or real_height <= 0:
            raise HTTPException(
                status_code=400,
                detail="Dimensions must be positive numbers"
            )

    async with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # сохранение калибровочных точек
        job.calibration_points = pts
        job.status = "calibrated"

        # сохранение информации о реальных размерах в job
        job.calibration_real_width = real_width
        job.calibration_real_height = real_height
        job.calibration_has_real_scale = knows_real

        db_upsert(job)

        # обновление информации о масштабе в job для использования при обработке
        job.calibration_scale_info = {
            'has_real_scale': knows_real,
            'real_width_m': real_width,
            'real_height_m': real_height
        }

    return {
        "status": "ok",
        "job_id": job_id,
        "calibration_points": pts,
        "knows_real_dimensions": knows_real,
        "real_width_m": real_width,
        "real_height_m": real_height
    }

@app.post("/api/v1/jobs/{job_id}/start", response_class=JSONResponse)
async def start_job(job_id: str) -> Dict[str, Any]:
    async with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status in ("processing", "done"):
            return {"status": job.status, "job_id": job_id}
        if not job.calibration_points:
            raise HTTPException(status_code=400, detail="Job is not calibrated yet")

        job.status = "processing"
        job.started_at = utc_now_iso()
        db_upsert(job)

    asyncio.create_task(run_job(job_id))
    return {"status": "processing", "job_id": job_id, "ws_url": f"/ws/jobs/{job_id}"}

@app.get("/api/v1/history", response_class=JSONResponse)
async def history(limit: int = 50) -> Dict[str, Any]:
    rows = db_list_jobs(limit=limit)
    for r in rows:
        r["downloads"] = job_downloads_from_row(r)
    return {"items": rows}

def job_downloads(job: JobRuntime) -> Dict[str, Optional[str]]:
    base = f"/api/v1/jobs/{job.job_id}/download"
    return {
        "feed": f"{base}/feed" if job.output_feed_path else None,
        "map": f"{base}/map" if job.output_map_path else None,
        "tracks_json": f"{base}/tracks_json" if job.tracks_json_path else None,
        "tracks_xlsx": f"{base}/tracks_xlsx" if job.tracks_xlsx_path else None,
        "report_pdf": f"{base}/report_pdf" if job.report_pdf_path else None,
    }

def job_downloads_from_row(row: Dict[str, Any]) -> Dict[str, Optional[str]]:
    job_id = row["job_id"]
    base = f"/api/v1/jobs/{job_id}/download"
    return {
        "feed": f"{base}/feed" if row.get("output_feed_path") else None,
        "map": f"{base}/map" if row.get("output_map_path") else None,
        "tracks_json": f"{base}/tracks_json" if row.get("tracks_json_path") else None,
        "tracks_xlsx": f"{base}/tracks_xlsx" if row.get("tracks_xlsx_path") else None,
        "report_pdf": f"{base}/report_pdf" if row.get("report_pdf_path") else None,
    }

@app.get("/api/v1/jobs/{job_id}/download/{kind}")
async def download(job_id: str, kind: str) -> FileResponse:
    # сначала берём данные из runtime (если job ещё в памяти), иначе читаем из SQLite
    async with jobs_lock:
        job = jobs.get(job_id)
    paths: Dict[str, Optional[Path]] = {}
    if job:
        paths = {
            "feed": job.output_feed_path,
            "map": job.output_map_path,
            "tracks_json": job.tracks_json_path,
            "tracks_xlsx": job.tracks_xlsx_path,
            "report_pdf": job.report_pdf_path,
        }
    else:
        row = db_get_job(job_id)
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        paths = {
            "feed": Path(row["output_feed_path"]) if row.get("output_feed_path") else None,
            "map": Path(row["output_map_path"]) if row.get("output_map_path") else None,
            "tracks_json": Path(row["tracks_json_path"]) if row.get("tracks_json_path") else None,
            "tracks_xlsx": Path(row["tracks_xlsx_path"]) if row.get("tracks_xlsx_path") else None,
            "report_pdf": Path(row["report_pdf_path"]) if row.get("report_pdf_path") else None,
        }

    path = paths.get(kind)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    media_type = "application/octet-stream"
    if kind.endswith("pdf"):
        media_type = "application/pdf"
    elif kind.endswith("xlsx"):
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif str(path).lower().endswith(".mp4"):
        media_type = "video/mp4"

    return FileResponse(str(path), media_type=media_type, filename=path.name)

# web-socket стриминг

@app.websocket("/ws/jobs/{job_id}")
async def ws_job(job_id: str, websocket: WebSocket) -> None:
    await websocket.accept()
    async with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            await websocket.send_json({"type": "error", "message": "Job not found"})
            await websocket.close()
            return
        job.clients.append(websocket)

    # отправляем начальный статус
    await websocket.send_json({"type": "status", "job_id": job_id, "status": job.status})

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with jobs_lock:
            job = jobs.get(job_id)
            if job and websocket in job.clients:
                job.clients.remove(websocket)

# фоновая обработка

async def run_job(job_id: str) -> None:
    loop = asyncio.get_running_loop()
    await asyncio.to_thread(process_job_sync, job_id, loop)

def process_job_sync(job_id: str, loop: asyncio.AbstractEventLoop) -> None:
    # Берём job
    job = jobs.get(job_id)
    if not job:
        return

    try:
        if not job.input_path or not job.input_path.exists():
            raise RuntimeError("Missing input video")
        if not job.calibration_points:
            raise RuntimeError("Job is not calibrated")

        # Инициализируем пайплайн под конкретный job
        detector = HumanDetector(
            model_name=(config.MODEL_PATH or config.MODEL_NAME),
            confidence_threshold=config.CONFIDENCE_THRESHOLD,
            iou_threshold=config.IOU_THRESHOLD,
            roi_points=job.calibration_points,
            debug=config.DEBUG_DETECTIONS,
        )
        tracker = HumanTracker(max_age=config.MAX_AGE, min_hits=config.MIN_HITS)
        visualizer = TrajectoryVisualizer(
            trajectory_length=config.TRAJECTORY_LENGTH,
            bbox_color=config.BBOX_COLOR,
            trajectory_color=config.TRAJECTORY_COLOR,
        )
        exporter = TrajectoryExporter()

        smoother = None
        if config.SMOOTHING_METHOD != "none":
            smoother = CoordinateSmoother(
                method=config.SMOOTHING_METHOD,
                alpha=config.SMOOTHING_ALPHA,
                window_size=config.SMOOTHING_WINDOW_SIZE,
            )

        # Находим информацию о реальных размерах из job
        calibration_has_real_scale = getattr(job, 'calibration_has_real_scale', False)
        calibration_real_width = getattr(job, 'calibration_real_width', 1.0)
        calibration_real_height = getattr(job, 'calibration_real_height', 1.5)

        # Если знаем реальные размеры, используем их
        if calibration_has_real_scale and calibration_real_width and calibration_real_height:
            map_width = calibration_real_width
            map_height = calibration_real_height
        else:
            # Используем значения по умолчанию из config
            map_width = config.FLOOR_MAP_WIDTH_METERS
            map_height = config.FLOOR_MAP_HEIGHT_METERS

        floor_map_generator = FloorMapGenerator(
            width_meters=map_width,  # ИЗМЕНЕНО: используем реальные размеры если есть
            height_meters=map_height,  # ИЗМЕНЕНО
            pixels_per_meter=config.FLOOR_MAP_PIXELS_PER_METER,
            calibration_width=config.CALIBRATION_AREA_WIDTH,
            calibration_height=config.CALIBRATION_AREA_HEIGHT,
        )
        floor_map_generator.set_calibration_points(job.calibration_points)

        homography = HomographyTransformer()
        ok = homography.calibrate(
            job.calibration_points,
            floor_map_generator.calibration_points_world,
            real_width_m=calibration_real_width if calibration_has_real_scale else None,
            real_height_m=calibration_real_height if calibration_has_real_scale else None,
            knows_real_dimensions=calibration_has_real_scale
        )
        if not ok:
            raise RuntimeError("Failed to calibrate homography")

        # Чтение/запись видео
        cap = cv2.VideoCapture(str(job.input_path))
        if not cap.isOpened():
            raise RuntimeError("Could not open video")

        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or config.VIDEO_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        job.video_fps = video_fps

        out_dir = job.job_dir or (config.OUTPUT_DIR / "web_jobs" / job_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        job.output_feed_path = out_dir / "video_feed.mp4"
        job.output_map_path = out_dir / "floor_map.mp4"
        job.tracks_json_path = out_dir / "tracks.json"
        job.tracks_xlsx_path = out_dir / "tracks.xlsx"
        job.report_pdf_path = out_dir / "report.pdf"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_feed = cv2.VideoWriter(str(job.output_feed_path), fourcc, video_fps, (width, height))
        writer_map = cv2.VideoWriter(
            str(job.output_map_path),
            fourcc,
            video_fps,
            (floor_map_generator.map_width, floor_map_generator.map_height),
        )

        # Траектории в "мировых" координатах (нормализованные 0..1 в расширенной карте)
        world_trajs: Dict[int, List[Tuple[int, Tuple[float, float]]]] = {}
        completed_world_trajs: Dict[int, List[Tuple[int, Tuple[float, float]]]] = {}

        start_t = time.time()
        last_stats_t = start_t
        people_max = 0

        frame_idx = 0
        stream_every_n = 2  # уменьшаем трафик (не каждый кадр стримим)

        while True:
            ok_read, frame = cap.read()
            if not ok_read or frame is None:
                break
            if camera_calibrator is not None:
                frame = camera_calibrator.undistort(frame)
            frame_idx += 1
            job.frames_processed = frame_idx

            detections = detector.detect(frame)
            tracks = tracker.update(detections)

            # Рендер Video Feed
            video_feed = frame.copy()
            for tr in tracks:
                video_feed = visualizer.draw_track_video_feed(video_feed, tr)

            # Рендер карты (сетка пересоздаётся каждый кадр)
            map_frame = floor_map_generator.generate_map()

            # Обновляем траектории в координатах карты (для подтверждённых треков)
            for tr in tracks:
                tid = int(tr.get("track_id", -1))
                if tid < 0:
                    continue
                x1, y1, x2, y2 = tr["bbox"]
                foot_x = (x1 + x2) / 2.0
                foot_y = y2
                if smoother:
                    foot_x, foot_y = smoother.smooth_point((foot_x, foot_y), tid)
                wp = homography.transform_point((foot_x, foot_y))
                if wp is None:
                    continue
                world_trajs.setdefault(tid, []).append((frame_idx, wp))
                # Также сохраняем в completed_world_trajs (для всех треков)
                completed_world_trajs.setdefault(tid, []).append((frame_idx, wp))

            # Рисуем все траектории на карте (цвет по track_id)
            for tid, traj in world_trajs.items():
                current_pos = traj[-1][1] if traj else None
                map_frame = visualizer.draw_trajectory_on_map(
                    map_frame,
                    traj,
                    track_id=tid,
                    current_position=current_pos,
                )

            people_max = max(people_max, len(tracks))

            # Пишем выходные видео
            writer_feed.write(video_feed)
            writer_map.write(map_frame)

            # Отправка кадров по WebSocket
            if (frame_idx % stream_every_n) == 0:
                payload_video = {
                    "type": "video_frame",
                    "frame": frame_idx,
                    "total_frames": total_frames,
                    "jpeg_b64": encode_jpeg_b64(video_feed, quality=80),
                }
                payload_map = {
                    "type": "map_frame",
                    "frame": frame_idx,
                    "total_frames": total_frames,
                    "jpeg_b64": encode_jpeg_b64(map_frame, quality=80),
                }
                asyncio.run_coroutine_threadsafe(broadcast(job, payload_video), loop)
                asyncio.run_coroutine_threadsafe(broadcast(job, payload_map), loop)

            # Обновление статистики (примерно 4 раза в секунду)
            now = time.time()
            if now - last_stats_t >= 0.25:
                elapsed = now - start_t
                proc_fps = (frame_idx / elapsed) if elapsed > 0 else 0.0
                payload_stats = {
                    "type": "stats",
                    "frame": frame_idx,
                    "total_frames": total_frames,
                    "processing_fps": proc_fps,
                    "people": len(tracks),
                    "people_max": people_max,
                    "status": "processing",
                }
                asyncio.run_coroutine_threadsafe(broadcast(job, payload_stats), loop)
                last_stats_t = now

        cap.release()
        writer_feed.release()
        writer_map.release()

        total_t = time.time() - start_t
        job.processing_fps = (job.frames_processed / total_t) if total_t > 0 else 0.0
        job.people_max = people_max

        # Экспорт треков + отчёт (с метриками, если есть world_trajs)
        all_tracks = tracker.get_all_tracks_history()
        with open(job.tracks_json_path, "w", encoding="utf-8") as f:
            json.dump(all_tracks, f, ensure_ascii=False, indent=2)
        exporter.export_excel(
            all_tracks,
            job.tracks_xlsx_path,
            world_trajectories=completed_world_trajs,
            video_fps=video_fps,
            map_width_m=floor_map_generator.width_meters,
            map_height_m=floor_map_generator.height_meters,
            offset_x_m=floor_map_generator.offset_x,
            offset_y_m=floor_map_generator.offset_y,
            knows_real_dimensions=calibration_real_height
        )
        exporter.export_pdf(
            all_tracks,
            job.report_pdf_path,
            world_trajectories=completed_world_trajs,
            video_fps=video_fps,
            map_width_m=floor_map_generator.width_meters,
            map_height_m=floor_map_generator.height_meters,
            offset_x_m=floor_map_generator.offset_x,
            offset_y_m=floor_map_generator.offset_y,
            knows_real_dimensions=calibration_real_height
        )

        job.status = "done"
        job.finished_at = utc_now_iso()
        db_upsert(job)

        done_payload = {
            "type": "done",
            "job_id": job.job_id,
            "status": job.status,
            "frames_processed": job.frames_processed,
            "processing_fps": job.processing_fps,
            "people_max": job.people_max,
            "downloads": job_downloads(job),
        }
        asyncio.run_coroutine_threadsafe(broadcast(job, done_payload), loop)

    except Exception as e:
        job.status = "error"
        job.error = str(e)
        job.finished_at = utc_now_iso()
        db_upsert(job)
        asyncio.run_coroutine_threadsafe(
            broadcast(job, {"type": "error", "message": str(e), "job_id": job.job_id}),
            loop,
        )
