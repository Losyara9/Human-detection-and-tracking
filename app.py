from __future__ import annotations

import asyncio
import base64
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from idlelib.help_about import version
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

DB_PATH = config.DATA_DIR