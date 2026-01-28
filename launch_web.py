import argparse
import sys
import subprocess
import time
import webbrowser
from pathlib import Path
import config


def launch_fastapi(host: str = None, port: int = None, reload: bool = False):
    host = host or "127.0.0.1"
    port = port or config.API_PORT

    print("=" * 60)
    print("Запуск FastAPI‑бэкенда")
    print("=" * 60)
    print(f"Сервер будет доступен по адресу: http://{host}:{port}")
    print(f"Документация API (Swagger): http://{host}:{port}/docs")
    print(f"Веб‑панель: http://{host}:{port}/static/index.html")
    print("\nНажмите Ctrl+C для остановки сервера")
    print("=" * 60 + "\n")

    try:
        import uvicorn
        uvicorn.run(
            "app:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\nСервер остановлен пользователем")
    except Exception as e:
        print(f"\nОшибка запуска сервера: {e}")
        sys.exit(1)


def launch_streamlit(port: int = 8501):
    streamlit_file = Path(__file__).parent / "web" / "streamlit_app.py"

    if not streamlit_file.exists():
        print(f"Ошибка: Streamlit‑приложение не найдено: {streamlit_file}")
        print("Создаю базовое Streamlit‑приложение")
        create_streamlit_app(streamlit_file)

    print("=" * 60)
    print("Запуск Streamlit‑фронтенда")
    print("=" * 60)
    print(f"Streamlit будет доступен по адресу: http://localhost:{port}")
    print("\nНажмите Ctrl+C для остановки")
    print("=" * 60 + "\n")

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(streamlit_file),
            "--server.port", str(port),
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n\nStreamlit остановлен пользователем")
    except Exception as e:
        print(f"\nОшибка запуска Streamlit: {e}")
        print("\nУбедитесь, что Streamlit установлен: pip install streamlit")
        sys.exit(1)


def create_streamlit_app(filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)

    content = '''"""
Streamlit web interface for human detection and tracking.
"""
import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
from pathlib import Path

st.set_page_config(
    page_title="Human Detection & Tracking",
    page_icon="👤",
    layout="wide"
)

st.title("👤 Human Detection and Tracking System")
st.markdown("Upload a video to detect and track humans with 2D trajectory visualization")

# Configuration
API_URL = st.sidebar.text_input(
    "API URL",
    value="http://localhost:8000",
    help="URL of the FastAPI backend server"
)

# File upload
uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov", "mkv"],
    help="Upload a video file for processing"
)

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Process Video", type="primary"):
        with st.spinner("Processing video... This may take a while."):
            try:
                # Upload video to API
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(
                    f"{API_URL}/api/v1/track",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "video/mp4")}
                )

                if response.status_code == 200:
                    result = response.json()
                    st.success("Video processed successfully!")
                    st.json(result)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error processing video: {e}")

# Image detection
st.header("Image Detection")
uploaded_image = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"],
    help="Upload an image to detect humans"
)

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Humans"):
        with st.spinner("Detecting humans..."):
            try:
                files = {"file": uploaded_image.getvalue()}
                response = requests.post(
                    f"{API_URL}/api/v1/detect",
                    files={"file": (uploaded_image.name, uploaded_image.getvalue(), "image/jpeg")}
                )

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Found {len(result.get('detections', []))} humans!")
                    st.json(result)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error detecting humans: {e}")

# Health check
st.sidebar.header("System Status")
if st.sidebar.button("Check API Health"):
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            st.sidebar.success("✅ API is healthy")
            st.sidebar.json(response.json())
        else:
            st.sidebar.error("API is not responding")
    except Exception as e:
        st.sidebar.error(f"Cannot connect to API: {e}")
'''

    with open(filepath, "w") as f:
        f.write(content)
    print(f"Создан Streamlit‑файл: {filepath}")


def launch_both(fastapi_port: int = None, streamlit_port: int = 8501):
    import threading

    fastapi_port = fastapi_port or config.API_PORT

    print("=" * 60)
    print("Запуск обоих сервисов")
    print("=" * 60)
    print(f"FastAPI: http://localhost:{fastapi_port}")
    print(f"Streamlit: http://localhost:{streamlit_port}")
    print("\nНажмите Ctrl+C для остановки обоих сервисов")
    print("=" * 60 + "\n")

    def run_fastapi():
        import uvicorn
        uvicorn.run("app:app", host="127.0.0.1", port=fastapi_port, log_level="info")

    def run_streamlit():
        streamlit_file = Path(__file__).parent / "web" / "streamlit_app.py"
        if not streamlit_file.exists():
            create_streamlit_app(streamlit_file)
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(streamlit_file),
            "--server.port", str(streamlit_port),
            "--server.headless", "true"
        ])

    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)

    fastapi_thread.start()
    time.sleep(2)
    streamlit_thread.start()

    try:
        fastapi_thread.join()
        streamlit_thread.join()
    except KeyboardInterrupt:
        print("\n\nОстановка серверов…")


def main():
    parser = argparse.ArgumentParser(
        description="Запуск веб‑интерфейса (FastAPI/Streamlit)"
    )
    parser.add_argument(
        "--mode",
        choices=["fastapi", "streamlit", "both"],
        default="fastapi",
        help="Что запускать (по умолчанию: fastapi)"
    )
    parser.add_argument(
        "--fastapi-port",
        type=int,
        default=None,
        help=f"Порт FastAPI (по умолчанию: {config.API_PORT})"
    )
    parser.add_argument(
        "--streamlit-port",
        type=int,
        default=8501,
        help="Порт Streamlit (по умолчанию: 8501)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Хост FastAPI (например 127.0.0.1 или 0.0.0.0)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Авто‑перезагрузка FastAPI (режим разработки)"
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Автоматически открыть браузер"
    )

    args = parser.parse_args()

    if args.mode == "fastapi":
        if args.open_browser:
            time.sleep(2)  # Ждём запуска сервера
            browser_host = args.host or "127.0.0.1"
            if browser_host == "0.0.0.0":
                browser_host = "127.0.0.1"
            webbrowser.open(f"http://{browser_host}:{args.fastapi_port or config.API_PORT}/docs")
        launch_fastapi(args.host, args.fastapi_port, args.reload)
    elif args.mode == "streamlit":
        if args.open_browser:
            time.sleep(2)
            webbrowser.open(f"http://localhost:{args.streamlit_port}")
        launch_streamlit(args.streamlit_port)
    elif args.mode == "both":
        if args.open_browser:
            time.sleep(3)
            webbrowser.open(f"http://localhost:{args.streamlit_port}")
        launch_both(args.fastapi_port, args.streamlit_port)


if __name__ == "__main__":
    main()
