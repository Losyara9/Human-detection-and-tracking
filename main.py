import argparse
import sys
import time
import cv2
import numpy as np
from pathlib import Path

from mpl_toolkits.mplot3d.proj3d import world_transformation
from openpyxl.styles.builtins import output

from core.detector import HumanDetector
from core.tracker import HumanTracker
from utils.visualization import TrajectoryVisualizer
from utils.homography import HomographyTransformer
from utils.export import TrajectoryExporter
from utils.floor_map import FloorMapGenerator, interactive_calibration
from utils.smoothing import CoordinateSmoother
import config

def main():
    parser = argparse.ArgumentParser(
        description='Система детекции и трекинга людей'
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Путь к входному видео'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Путь к выходному видуо (опционально)'
    )
    parser.add_argument(
        '--calibration',
        type=str,
        default=None,
        help='Путь к файлу калибровки/гомографии (опционально)'
    )
    parser.add_argument(
        "--export",
        type=str,
        choices=config.EXPORT_FORMATS,
        default=None,
        help="Экспорт траекторий в формат (xlsx/json/pdf)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Показывать 2 окна: Video Feed и 2D Floor Map"
    )
    parser.add_argument(
        "--no-2d",
        action="store_true",
        help="Отключить 2D‑визуализацию"
    )

    args = parser.parse_args()

    # проверка входного видео
    video_path = Path(args.video)
    if not video_path.exists():
        print('Ошибка: видео не найдено: ', video_path)
        sys.exit(1)

    print('Инициализация детектора')
    detector = HumanDetector(
        model_name=config.MODEL_NAME,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        iou_threshold=config.IOU_THRESHOLD,
        debug=config.DEBUG_DETECTIONS
    )

    print('Инициализация трекера')
    tracker = HumanTracker(
        max_age=config.MAX_AGE,
        min_hits=config.MIN_HITS
    )

    #Гомография/д2 карта
    homography_transformer = None
    floor_map_generator = None

    # Если есть файл калибровки, то используем, иначе интерактивная калибровка
    if args.calibration:
        calib_path = Path(args.calibration)
        if calib_path.exists():
            print('Загрузка калибровки гомографии из файла')
            homography_transformer = HomographyTransformer()
            homography_transformer.load(calib_path)
        else:
            print('Предупреждение: файл калибровки не найден: ', calib_path)
            print('Будет выполнена интерактивная калибровка')

    visualizer = TrajectoryVisualizer(
        trajectory_length=config.TRAJECTORY_LENGTH,
        bbox_color=config.BBOX_COLOR,
        trajectory_color=config.TRAJECTORY_COLOR
    )

    print('Обработка видео: ', video_path)

    # Открываем видео
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print('Ошибка: не удалось открыть видео: ', video_path)
        sys.exit(1)

    # параметры видео
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or config.VIDEO_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0:
        fps = config.VIDEO_FPS
        print('Предупреждение: FPS не определен, используется по умолчанию: ', fps)

    print(f'Параметры видео: {width}x{height} @ {fps} FPS, количество кадров: {total_frames}')

    # интерактивная калибровка
    if not homography_transformer and not args.no_2d:
        print('\n' + '='*50)
        print('Старт интерактивной калибровки')
        print('='*50)

        # Чтение первого кадра
        ret, first_frame = cap.read()
        if not ret:
            print('Ошибка: не удалось прочитать первый кадр для калибровки')
            sys.exit(1)

        calibration_points = interactive_calibration(first_frame)

        if calibration_points is None:
            print('Калибровка отменена. Выход.')
            cap.release()
            sys.exit(0)

        # Инициализация генератора карты
        floor_map_generator = FloorMapGenerator(
            width_meters=config.FLOOR_MAP_WIDTH_METERS,
            height_meters=config.FLOOR_MAP_HEIGHT_METERS,
            pixels_per_meter=config.FLOOR_MAP_PIXELS_PER_METER,
            calibration_width=config.CALIBRATION_AREA_WIDTH,
            calibration_height=config.CALIBRATION_AREA_HEIGHT
        )
        floor_map_generator.set_calibration_points(calibration_points)

        # установка ROI для фильтрации детектора
        detector.set_roi(calibration_points)

        homography_transformer = HomographyTransformer()

        world_points = floor_map_generator.calibration_points_world

        success = homography_transformer.calibrate(calibration_points, world_points)
        if not success:
            print('Ошибка: не удалось откалибровать гомографию')
            sys.exit(1)

        print('Калибровка прошла успешно')

        # сохранение калибровки
        if args.calibration:
            calib_path = Path(args.calibration)
            calib_path.parent.mkdir(parents=True, exist_ok=True)
            homography_transformer.save(calib_path)
            print(f'Калибровка сохранена: {calib_path}')

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if homography_transformer and not args.no_2d and floor_map_generator is None:
        floor_map_generator = FloorMapGenerator(
            width_meters=config.FLOOR_MAP_WIDTH_METERS,
            height_meters=config.FLOOR_MAP_HEIGHT_METERS,
            pixels_per_meter=config.FLOOR_MAP_PIXELS_PER_METER,
            calibration_width=config.CALIBRATION_AREA_WIDTH,
            calibration_height=config.CALIBRATION_AREA_HEIGHT
        )

    # инициализация сглаживания координат
    coordinate_smoother = None
    if homography_transformer and not args.no_2d:
        if config.SMOOTHING_METHOD != 'none':
            coordinate_smoother = CoordinateSmoother(
                method=config.SMOOTHING_METHOD,
                alpha=config.SMOOTHING_ALPHA,
                window_size=config.SMOOTHING_WINDOW_SIZE
            )
            print(f'Сглаживание координат включено: {config.SMOOTHING_METHOD} (alpha={config.SMOOTHING_ALPHA})')

    # инициализация окон для стриминга
    video_writer_feed = None
    video_writer_map = None

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # видео с детекцией
        feed_output_path = output_path.parent / f'{output_path.stem}_feed{output_path.suffix}'
        video_writer_feed = cv2.VideoWriter(
            str(feed_output_path),
            fourcc,
            fps,
            (width, height)
        )
        print('Готовое видео будет сохранено в: ', feed_output_path)

        # 2д карта
        if homography_transformer and not args.no_2d and floor_map_generator:
            map_width = floor_map_generator.map_width
            map_height = floor_map_generator.map_height
            map_output_path = output_path.parent / f'{output_path.stem}_map{output_path.suffix}'
            video_writer_map = cv2.VideoWriter(
                str(feed_output_path),
                fourcc,
                fps,
                (width, height)
            )
            print('Готовая 2Д карта будет сохранена в: ', map_output_path)

    exporter = None
    if args.export:
        exporter = TrajectoryExporter()

    # отслеживание fps
    frame_count = 0
    start_time = time.time()
    fps_times = []

    floor_map = None
    if homography_transformer and not args.no_2d and floor_map_generator:
        floor_map = floor_map_generator.generate_map()

    # основной цикл

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_start = time.time()

            video_feed_frame = frame.copy()

            detections = detector.detect(frame)

            tracks = tracker.update(detections)

            # отрисовка bbox и нижней точки на исходном видео
            for track in tracks:
                video_feed_frame = visualizer.draw_track_video_feed(video_feed_frame, track)

            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            fps_times.append(time.time() - frame_start)

            total_frames_str = str(total_frames) if total_frames > 0 else '?'
            info_text = f'Frame: {frame_count}/{total_frames_str} | FPS: {current_fps:.1f} | Tracks: {len(tracks)}'
            cv2.putText(
                video_feed_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                config.TEXT_COLOR,
                2
            )

            # обновление 2Д карты с траекторией
            map_frame = None
            if homography_transformer and not args.no_2d and floor_map_generator:
                # создание новой карты
                map_frame = floor_map_generator.generate_map()

                for track in tracks:
                    bbox = track['bbox']
                    x1, y1, x2, y2 = bbox
                    foot_x = (x1 + x2) / 2
                    foot_y = y2

                    if coordinate_smoother:
                        track_id = track.get('track_id', -1)
                        foot_x, foot_y = coordinate_smoother.smooth_point((foot_x, foot_y), track_id)

                    # трансформация на координаты "мира"
                    world_point = homography_transformer.transform_point((foot_x, foot_y))

                    if world_point:
                        trajectory = track.get('trajectory', [])
                        world_trajectory = []
                        track_id = track.get('track_id', -1)

                        for frame_id, bbox_traj in trajectory:
                            x1_t, y1_t, x2_t, y2_t = bbox_traj
                            foot_x_t = (x1_t + x2_t) / 2
                            foot_y_t = y2_t

                            if coordinate_smoother:
                                foot_x_t, foot_y_t = coordinate_smoother.smooth_point(
                                    (foot_x_t, foot_y_t), track_id
                                )

                            world_pt = homography_transformer.transform_point((foot_x_t, foot_y_t))
                            if world_pt:
                                world_trajectory.append((frame_id, world_pt))

                        # отрисовка траектории на карте
                        map_frame = visualizer.draw_trajectory_on_map(
                            map_frame,
                            world_trajectory,
                            track.get('track_id'),
                            world_point # текущая позиция
                        )

                # текстовая информация на карте
                cv2.putText(
                    map_frame,
                    f"2D Floor Map | Tracks: {len(tracks)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2
                )

            if args.show:
                cv2.imshow('Video feed', video_feed_frame)

                if map_frame is not None:
                    cv2.imshow('2D floor map', map_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27: # ESC
                    print('\nОбработка прервана пользователем (ESC)')
                    break

            if video_writer_feed:
                video_writer_feed.write(video_feed_frame)

            if video_writer_map and map_frame is not None:
                video_writer_map.write(map_frame)

            # Запись прогресса каждые 30 кадров
            if frame_count % 30 == 0:
                if total_frames > 0:
                    progress = (frame_count / total_frames) * 100
                else:
                    progress = 0
                recent_times = fps_times[-30:] if len(fps_times) >= 30 else fps_times
                valid_times = [t for t in recent_times if t > 0.001]
                if valid_times:
                    avg_time = np.mean(valid_times)
                    processing_fps = 1.0 / avg_time if avg_time > 0 else 0
                else:
                    processing_fps = 0
                if total_frames > 0:
                    print(f'Прогресс: {progress:.1f}% | FPS обработки: {processing_fps:.1f}')
                else:
                    print(f'Кадров обработано: {frame_count} | FPS обработки: {processing_fps:.1f}')

    except KeyboardInterrupt:
        print('\nОбработка прервана пользователем (Ctrl+C)')

    finally:
        cap.release()
        if video_writer_feed:
            video_writer_feed.release()
        if video_writer_map:
            video_writer_map.release()
        if args.show:
            cv2.destroyAllWindows()

    # Подсчет итоговой статистики
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    if fps_times:
        valid_times = [t for t in fps_times if t > 0.001]
        if valid_times:
            processing_fps = np.mean(1.0 / np.array(valid_times))
        else:
            processing_fps = 0
    else:
        processing_fps = 0

    print('\n' + '=' * 50)
    print('Статистика обработки:')
    print('  Кадров обработано: ', frame_count)
    print(f'  Время: {total_time:.2f} сек')
    print(f'  Средний FPS: {avg_fps:.2f}')
    print(f'  FPS обработки: {processing_fps:.2f}')
    print(f'  Всего треков: {len(tracker.get_all_tracks())}')
    print('=' * 50)

    # Экспорт статистики в отчеты
    if args.export and exporter:
        print(f'\nЭкспорт данных траектории в {args.export} формате')
        if hasattr(tracker, 'get_all_tracks_history'):
            all_tracks = tracker.get_all_tracks_history()
        else:
            all_tracks = tracker.get_all_tracks()

        if not all_tracks:
            print('Предупреждение: нет треков для экспорта')
        else:
            output_name = video_path.stem
            output_path = config.OUTPUT_DIR / f'{output_name}_tracks.{args.export}'

            try:
                if args.export == 'xlsx':
                    exporter.export_excel(all_tracks, output_path)
                elif args.export == 'json':
                    exporter.export_json(all_tracks, output_path)
                elif args.export == 'pdf':
                    exporter.export_pdf(all_tracks, output_path)

                print(f'Экспортировано {len(all_tracks)} треков в: {output_path}')

            except Exception as e:
                print('Ошибка экспорта: ', e)

    # генерация итоговой 2д траектории
    if homography_transformer and not args.no_2d and floor_map_generator:
        print('\nГенерация финальной 2D-визуализации траекторий')

        # используем историю треков
        if hasattr(tracker, 'get_all_tracks_history'):
            all_tracks = tracker.get_all_tracks_history()
        else:
            all_tracks = tracker.get_all_tracks()

        if all_tracks:
            final_map = floor_map_generator.generate_map()

            for track in all_tracks:
                trajectory = track.get('trajectory', [])
                world_trajectory = []
                track_id = track.get('track_id', -1)

                for frame_id, bbox in trajectory:
                    x1, y1, x2, y2 = bbox
                    foot_x = (x1 + x2) / 2
                    foot_y = y2

                    if coordinate_smoother:
                        foot_x, foot_y = coordinate_smoother.smooth_point((foot_x, foot_y), track_id)

                    world_pt = homography_transformer.transform_point((foot_x, foot_y))
                    if world_pt:
                        world_trajectory.append((frame_id, world_pt))

                if world_trajectory:
                    final_map = visualizer.draw_trajectory_on_map(
                        final_map,
                        world_trajectory,
                        track.get('track_id'),
                        None
                    )

            output_2d_path = config.OUTPUT_DIR / f'{video_path.stem}_2d_trajectories.png'
            cv2.imwrite(str(output_2d_path), final_map)
            print('Готовая 2Д визуализация траектории сохранена в: ', output_2d_path)

            if args.show:
                cv2.imshow('Final 2D trajectory visualization', final_map)
                print('Нажмите любую клавишу, чтобы закрыть окно финальной 2D визуализации')
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    print('\nГотово')

if __name__ == '__main__':
    main()
