import pandas as pd
import json
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import config
from datetime import datetime

class TrajectoryExporter:


    def export_csv(
            self,
            tracks: List[Dict],
            output_path: Path,
            world_trajectories: Optional[Dict[int, List[Tuple[int, Tuple[float, float]]]]] = None,
            video_fps: Optional[float] = None,
            map_width_m: Optional[float] = None,
            map_height_m: Optional[float] = None,
            offset_x_m: float = 0.0,
            offset_y_m: float = 0.0,
            knows_real_dimensions: bool = False
    ):

        data = []
        world_trajectories = world_trajectories or {}
        video_fps = float(video_fps or 0.0)
        map_width_m = float(map_width_m or config.FLOOR_MAP_WIDTH_METERS)
        map_height_m = float(map_height_m or config.FLOOR_MAP_HEIGHT_METERS)

        for track in tracks:
            track_id = track.get('track_id', -1)
            trajectory = track.get('trajectory', [])
            wtraj = world_trajectories.get(int(track_id), [])
            w_by_frame = {int(fid): (float(p[0]), float(p[1])) for fid, p in wtraj}

            for frame_id, bbox in trajectory:
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                x_norm = None
                y_norm = None
                x_m = None
                y_m = None
                x_cal_m = None
                y_cal_m = None

                if int(frame_id) in w_by_frame:
                    x_norm, y_norm = w_by_frame[int(frame_id)]
                    x_m = x_norm * map_width_m
                    y_m = y_norm * map_height_m
                    x_cal_m = x_m - offset_x_m
                    y_cal_m = y_m - offset_y_m

                data.append({
                    "track_id": track_id,
                    "frame_id": frame_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "center_x": center_x,
                    "center_y": center_y,
                    "x_norm": x_norm,
                    "y_norm": y_norm,
                    "x_m": x_m if knows_real_dimensions else None,  # ТОЛЬКО если знаем реальные размеры
                    "y_m": y_m if knows_real_dimensions else None,
                    "x_cal_m": x_cal_m if knows_real_dimensions else None,
                    "y_cal_m": y_cal_m if knows_real_dimensions else None,
                    "units": "meters" if knows_real_dimensions else "pixels",
                })

        df = pd.DataFrame(data)
        df.to_csv(output_path, index = False)

    def export_json(self, tracks: List[Dict], output_path: Path):
        with open(output_path, 'w') as f:
            json.dump(tracks, f, indent=2)

    def export_excel(
            self,
            output_path: Path,
            world_trajectories: Optional[Dict[int, List[Tuple[int, Tuple[float, float]]]]] = None,
            video_fps: Optional[float] = None,
            map_width_m: Optional[float] = None,
            map_height_m: Optional[float] = None,
            offset_x_m: float = 0.0,
            offset_y_m: float = 0.0,
            knows_real_dimensions: bool = False,
    ):

        world_trajectories = world_trajectories or {}
        video_fps = float(video_fps or 0.0)
        map_width_m = float(map_width_m or config.FLOOR_MAP_WIDTH_METERS)
        map_height_m = float(map_height_m or config.FLOOR_MAP_HEIGHT_METERS)

        points_rows: List[Dict[str, Any]] = []
        for track in tracks:
            track_id = track.get('track_id', -1)
            trajectory = track.get('trajectory', [])
            wtraj = world_trajectories.get(int(track_id), [])
            w_by_frame = {int(fid): (float(p[0]), float(p[1])) for fid, p in wtraj}

            for frame_id, bbox in trajectory:
                x1, y1, x2, y2 = bbox
                foot_x = (x1 + x2) / 2
                foot_y = y2
                x_norm = None
                y_norm = None
                x_m = None
                y_m = None
                x_cal_m = None
                y_cal_m = None

                if int(frame_id) in w_by_frame:
                    x_norm, y_norm = w_by_frame[int(frame_id)]
                    if knows_real_dimensions:
                        x_m = x_norm * map_width_m
                        y_m = y_norm * map_height_m
                        x_cal_m = x_m - offset_x_m
                        y_cal_m = y_m - offset_y_m

                points_rows.append(
                    {
                        "track_id": track_id,
                        "frame_id": frame_id,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "foot_x": foot_x,
                        "foot_y": foot_y,
                        "x_norm": x_norm,
                        "y_norm": y_norm,
                        "x_m": x_m if knows_real_dimensions else None,  # ТОЛЬКО если знаем реальные размеры
                        "y_m": y_m if knows_real_dimensions else None,
                        "x_cal_m": x_cal_m if knows_real_dimensions else None,
                        "y_cal_m": y_cal_m if knows_real_dimensions else None,
                        "units": "meters" if knows_real_dimensions else "pixels"
                    }
                )

        df_points = pd.DataFrame(points_rows)

        summary_rows = self.build_summary_rows(
            tracks=tracks,
            world_trajectories=world_trajectories,
            video_frps=video_fps,
            map_width_m=map_width_m,
            map_height_m=map_height_m,
            offset_x_m=offset_x_m,
            offset_y_m=offset_y_m,
            knows_real_dimensions=knows_real_dimensions
        )
        df_summary = pd.DataFrame(summary_rows)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_summary.to_excel(writer, index=False, sheet_name='summary')
            df_points.to_excel(writer, index=False, sheet_name='points')

    def export_pdf(
            self,
            tracks: List[Dict],
            output_path: Path,
            world_trajectories: Optional[Dict[int, List[Tuple[int, Tuple[float, float]]]]] = None,
            video_fps: Optional[float] = None,
            map_width_m: Optional[float] = None,
            map_height_m: Optional[float] = None,
            offset_x_m: float = 0.0,
            offset_y_m: float = 0.0,
            knows_real_dimensions: bool = False
    ):

        doc = SimpleDocTemplate(str(output_path), pagesize=A4)
        story = []
        styles = getSampleStyleSheet()

        styles['Normal'].fontsize = 10
        styles['Title'].fontsize = 16

        if 'Heading2' not in styles:
            styles['Heading2'] = styles['Heading2']
        styles['Heading2'].fontSize = 14

        if 'Heading3' not in styles:
            styles['Heading3'] = styles['Heading2']
        styles['Heading3'].fontSize = 12

        title = Paragraph("Human Tracking Analysis Report", styles["Title"])
        story.append(title)
        story.append(Spacer(1, 0.2 * inch))

        # Сводная информация
        summary_text = f"""
                <b>Report Summary</b><br/>
                Total tracks: {len(tracks)}<br/>
                Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
                """

        if knows_real_dimensions and map_width_m and map_height_m:
            summary_text += f"Scale: {map_width_m:.2f}m × {map_height_m:.2f}m<br/>"
        else:
            summary_text += "Scale: Pixel-based (relative units)<br/>"

        summary = Paragraph(summary_text, styles["Normal"])
        story.append(summary)
        story.append(Spacer(1, 0.3 * inch))

        # Таблица метрик по трекам
        world_trajectories = world_trajectories or {}
        video_fps = float(video_fps or 0.0)
        map_width_m = float(map_width_m or config.FLOOR_MAP_WIDTH_METERS)
        map_height_m = float(map_height_m or config.FLOOR_MAP_HEIGHT_METERS)

        summary_rows = self._build_summary_rows(
            tracks=tracks,
            world_trajectories=world_trajectories,
            video_fps=video_fps,
            map_width_m=map_width_m,
            map_height_m=map_height_m,
            offset_x_m=offset_x_m,
            offset_y_m=offset_y_m,
            knows_real_dimensions=knows_real_dimensions
        )

        story.append(Spacer(1, 0.2 * inch))

        # Заголовок секции с треками
        if summary_rows:
            tracks_header = Paragraph("<b>Track Analysis Details</b>", styles["Heading2"])
            story.append(tracks_header)
            story.append(Spacer(1, 0.15 * inch))

        # Для каждого трека создаем отдельную вертикальную таблицу
        for i, track_data in enumerate(summary_rows):
            # Заголовок для трека
            track_header = Paragraph(f"<b>Track #{i + 1} (ID: {track_data.get('track_id', 'N/A')})</b>",
                                     styles["Heading3"])
            story.append(track_header)
            story.append(Spacer(1, 0.1 * inch))

            # Создаем вертикальную таблицу с метриками
            metrics_data = [
                ["Metric", "Value"],
                ["Frames processed", str(track_data.get("frames", 0))],
                ["Hits", str(track_data.get("hits", 0))],
                ["Age", str(track_data.get("age", 0))],
                ["Duration", track_data.get("duration_s", "0.0 s")],
                ["Total distance", track_data.get("distance", "0.0")],
                ["Average speed", track_data.get("avg_speed", "0.0")],
                ["Maximum speed", track_data.get("max_speed", "0.0")],
                ["Movement pattern", track_data.get("pattern", "no data")],
                ["Most frequent zone", track_data.get("top_zone", "—")],
            ]

            # Добавляем зональное время если есть
            zone_nw = track_data.get("zone_NW_s", "0.0")
            zone_ne = track_data.get("zone_NE_s", "0.0")
            zone_sw = track_data.get("zone_SW_s", "0.0")
            zone_se = track_data.get("zone_SE_s", "0.0")

            if zone_nw or zone_ne or zone_sw or zone_se:
                metrics_data.append(["Time in NW zone", f"{zone_nw} s"])
                metrics_data.append(["Time in NE zone", f"{zone_ne} s"])
                metrics_data.append(["Time in SW zone", f"{zone_sw} s"])
                metrics_data.append(["Time in SE zone", f"{zone_se} s"])

            metrics_table = Table(metrics_data, colWidths=[2.2 * inch, 2.8 * inch])
            metrics_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
            ]))

            story.append(metrics_table)
            story.append(Spacer(1, 0.3 * inch))

            # Если треков нет
            if not summary_rows:
                no_data = Paragraph("<i>No tracks detected in the video</i>", styles["Normal"])
                story.append(no_data)
                story.append(Spacer(1, 0.2 * inch))

            story.append(Spacer(1, 0.2 * inch))

            notes_text = """
                <b>Notes:</b><br/>
                1. Distance and speed are calculated based on trajectory points.<br/>
                2. Pattern classification: linear (straight), chaotic (random), mixed.<br/>
                3. Zones are defined as quadrants of the tracking area.<br/>
                """

            if not knows_real_dimensions:
                notes_text += """
                    4. Measurements are in pixel units (relative scale).<br/>
                    5. For real-world metrics, provide area dimensions during calibration.<br/>
                    """

            notes = Paragraph(notes_text, styles["Normal"])
            story.append(notes)

            doc.build(story)

    def build_summary_rows(
            self,
            tracks: List[Dict],
            world_trajectories: Dict[int, List[Tuple[int, Tuple[float, float]]]],
            video_fps: float,
            map_width_m: float,
            map_height_m: float,
            offset_x_m: float,
            offset_y_m: float,
            knows_real_dimensions: bool = False,
    ) -> List[Dict[str, Any]]:
        # сводные метрики по трекам
        rows: List[Dict[str, Any]] = []
        fps = float(video_fps or 0.0)

        for tr in tracks:
            track_id = int(tr.get("track_id", -1))
            bbox_traj = tr.get("trajectory", [])
            frames = len(bbox_traj)
            hits = int(tr.get("hits", 0) or 0)
            age = int(tr.get("age", 0) or 0)

            wtraj = world_trajectories.get(track_id, [])
            # Перевод в метры на расширенной карте, затем в «метры калибровки» вычитанием смещений
            pts_m: List[Tuple[int, float, float]] = []
            for fid, (x_norm, y_norm) in wtraj:
                x_m = float(x_norm) * map_width_m
                y_m = float(y_norm) * map_height_m
                x_cal = x_m - offset_x_m
                y_cal = y_m - offset_y_m
                pts_m.append((int(fid), x_cal, y_cal))

            distance_m = 0.0
            max_speed_mps = 0.0
            if len(pts_m) >= 2 and fps > 0:
                for (f0, x0, y0), (f1, x1, y1) in zip(pts_m, pts_m[1:]):
                    df = max(1, f1 - f0)
                    d = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                    distance_m += d
                    v = d * fps / df
                    if v > max_speed_mps:
                        max_speed_mps = v

            duration_s = 0.0
            if len(pts_m) >= 2 and fps > 0:
                duration_s = (pts_m[-1][0] - pts_m[0][0]) / fps

            # В зависимости от типа измерений
            if knows_real_dimensions:
                avg_speed = (distance_m / duration_s) if duration_s > 0 else 0.0
                # Форматируем числа
                distance_str = f"{distance_m:.2f} m"
                avg_speed_str = f"{avg_speed:.2f} m/s"
                max_speed_str = f"{max_speed_mps:.2f} m/s"
                units = "m"
            else:
                # Для пиксельных единиц - конвертируем в пиксели
                scale_factor = 100  # примерный масштаб
                distance_px = distance_m * scale_factor
                avg_speed_pps = (distance_px / duration_s) if duration_s > 0 else 0.0
                max_speed_pps = max_speed_mps * scale_factor

                # Форматируем числа
                distance_str = f"{distance_px:.0f} px"
                avg_speed_str = f"{avg_speed_pps:.1f} px/s"
                max_speed_str = f"{max_speed_pps:.1f} px/s"
                units = "px"

            # Время в зонах (4 квадранта области калибровки 1.0 x 1.5 по умолчанию)
            zones = {"NW": 0, "NE": 0, "SW": 0, "SE": 0}
            if pts_m and fps > 0:
                for _, x, y in pts_m:
                    # X: 0..1.0 (слева/справа), Y: 0..1.5 (сверху/снизу)
                    left = x < (config.CALIBRATION_AREA_WIDTH / 2)
                    top = y < (config.CALIBRATION_AREA_HEIGHT / 2)
                    if left and top:
                        zones["NW"] += 1
                    elif (not left) and top:
                        zones["NE"] += 1
                    elif left and (not top):
                        zones["SW"] += 1
                    else:
                        zones["SE"] += 1

            zones_s = {k: (v / fps if fps > 0 else 0.0) for k, v in zones.items()}
            top_zone = max(zones_s.items(), key=lambda kv: kv[1])[0] if zones_s else ""

            # Анализ типа движения (линейный/хаотичный/смешанный)
            pattern = "no data"
            if (distance_m < 0.3 if knows_real_dimensions else distance_m * 100 < 30):
                pattern = "almost stationary"
            elif len(pts_m) >= 2:
                dx = pts_m[-1][1] - pts_m[0][1]
                dy = pts_m[-1][2] - pts_m[0][2]
                displacement = (dx * dx + dy * dy) ** 0.5
                straightness = (displacement / distance_m) if distance_m > 0 else 0.0
                if straightness > 0.85:
                    pattern = "linear"
                elif straightness < 0.5:
                    pattern = "chaotic"
                else:
                    pattern = "mixed"

            rows.append(
                {
                    "track_id": track_id,
                    "frames": frames,
                    "hits": hits,
                    "age": age,
                    "distance": distance_str,
                    "duration_s": f"{duration_s:.1f} s",
                    "avg_speed": avg_speed_str,
                    "max_speed": max_speed_str,
                    "zone_NW_s": f"{zones_s.get('NW', 0.0):.1f}",
                    "zone_NE_s": f"{zones_s.get('NE', 0.0):.1f}",
                    "zone_SW_s": f"{zones_s.get('SW', 0.0):.1f}",
                    "zone_SE_s": f"{zones_s.get('SE', 0.0):.1f}",
                    "top_zone": top_zone,
                    "pattern": pattern,
                    "scale_type": "real" if knows_real_dimensions else "pixel"
                }
            )

        return rows
