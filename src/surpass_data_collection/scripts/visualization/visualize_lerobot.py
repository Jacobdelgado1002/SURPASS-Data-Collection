#!/usr/bin/env python3
"""
LeRobot Dataset Visualizer for SURPASS DVRK Surgical Robot Data
===============================================================

Interactive GUI for exploring LeRobot v2.1 format datasets with synchronized
video playback, 3D end-effector trajectory, and time-series joint plots.

Usage:
    python visualize_lerobot.py --dataset-path <path_to_dataset>
    python visualize_lerobot.py  # uses default SURPASS path

Dependencies:
    pip install PyQt5 pyqtgraph PyOpenGL opencv-python pyarrow numpy

Controls:
    Space       - Play/Pause
    Left/Right  - Step forward/backward
    Home/End    - Jump to start/end
    +/-         - Adjust playback speed
"""

import sys
import os
import json
import argparse
import numpy as np
import pyarrow.parquet as pq
import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QComboBox, QPushButton, QSlider, QSplitter,
    QGroupBox, QCheckBox, QStatusBar, QSizePolicy, QAction
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeySequence
import pyqtgraph as pg

# Try OpenGL 3D support; fall back to matplotlib if unavailable
try:
    import pyqtgraph.opengl as gl
    HAS_GL = True
except Exception:
    HAS_GL = False

if not HAS_GL:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ─── Color Palette (Catppuccin Mocha) ────────────────────────────────────────
C = {
    'base': '#1e1e2e', 'mantle': '#181825', 'crust': '#11111b',
    'surface0': '#313244', 'surface1': '#45475a', 'surface2': '#585b70',
    'text': '#cdd6f4', 'subtext': '#a6adc8',
    'blue': '#89b4fa', 'sky': '#89dceb', 'teal': '#94e2d5',
    'lavender': '#b4befe',
    'peach': '#fab387', 'yellow': '#f9e2af', 'flamingo': '#f2cdcd',
    'pink': '#f5c2e7',
    'green': '#a6e3a1', 'red': '#f38ba8', 'mauve': '#cba6f7',
    'rosewater': '#f5e0dc',
}

# Per-component colors for PSM1 (cool) and PSM2 (warm)
PSM1_POS_C = [C['blue'], C['sky'], C['teal']]
PSM1_ORI_C = [C['blue'], C['sky'], C['teal'], C['lavender']]
PSM2_POS_C = [C['peach'], C['yellow'], C['flamingo']]
PSM2_ORI_C = [C['peach'], C['yellow'], C['flamingo'], C['pink']]

POS_LABELS = ['X', 'Y', 'Z']
ORI_LABELS = ['qX', 'qY', 'qZ', 'qW']

SPEED_OPTIONS = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0]

VIDEO_DISPLAY_NAMES = {
    'observation.images.endoscope.left': 'Endoscope L',
    'observation.images.endoscope.right': 'Endoscope R',
    'observation.images.wrist.left': 'Wrist L',
    'observation.images.wrist.right': 'Wrist R',
}

# ─── Utilities ────────────────────────────────────────────────────────────────


def hex_to_rgba(h, a=1.0):
    h = h.lstrip('#')
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255,
            int(h[4:6], 16) / 255, a)


def hex_to_gl_color(h, a=1.0):
    """Return (r, g, b, a) floats for pyqtgraph GL items."""
    return hex_to_rgba(h, a)


def np_to_qpixmap(frame_rgb, target_w, target_h):
    """Convert an RGB numpy array to a QPixmap, resized efficiently."""
    h, w = frame_rgb.shape[:2]
    # Resize in OpenCV (faster than Qt scaling)
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    if nw != w or nh != h:
        frame_rgb = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    qimg = QImage(frame_rgb.data, nw, nh, 3 * nw, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ─── Data Loading ─────────────────────────────────────────────────────────────

class DataLoader:
    """Loads and caches LeRobot v2.1 metadata and episode data."""

    def __init__(self, dataset_path):
        self.path = dataset_path
        meta = os.path.join(dataset_path, 'meta')

        with open(os.path.join(meta, 'info.json')) as f:
            self.info = json.load(f)

        self.episodes = []
        with open(os.path.join(meta, 'episodes.jsonl')) as f:
            for line in f:
                if line.strip():
                    self.episodes.append(json.loads(line))

        self.fps = self.info.get('fps', 30)
        self.num_episodes = len(self.episodes)
        self.chunk_size = self.info.get('chunks_size', 1000)
        self.video_keys = sorted(
            k for k, v in self.info['features'].items()
            if v.get('dtype') == 'video'
        )
        state_feat = self.info['features'].get('observation.state', {})
        self.state_names = state_feat.get('names', [[]])[0]

        self._cache_idx = -1
        self._cache = None

    @staticmethod
    def _arrow_list_col_to_np(table, col_name):
        """Convert a pyarrow list column to a 2D numpy array (no pandas needed)."""
        col = table.column(col_name)
        rows = []
        for val in col:
            rows.append(val.as_py())
        return np.array(rows, dtype=np.float32)

    def load_episode(self, idx):
        if self._cache_idx == idx:
            return self._cache
        chunk = idx // self.chunk_size
        path = os.path.join(
            self.path, 'data', f'chunk-{chunk:03d}',
            f'episode_{idx:06d}.parquet'
        )
        table = pq.ParquetFile(path).read()

        states = self._arrow_list_col_to_np(table, 'observation.state')
        actions = None
        if 'action' in table.column_names:
            actions = self._arrow_list_col_to_np(table, 'action')

        ts_col = table.column('timestamp')
        ts = np.array([v.as_py() for v in ts_col], dtype=np.float64)

        self._cache = {
            'states': states,
            'actions': actions,
            'timestamps': ts,
            'n_frames': table.num_rows,
            'episode_idx': idx,
        }
        self._cache_idx = idx
        return self._cache

    def video_path(self, idx, key):
        chunk = idx // self.chunk_size
        return os.path.join(
            self.path, 'videos', f'chunk-{chunk:03d}',
            key, f'episode_{idx:06d}.mp4'
        )


# ─── Video Manager ────────────────────────────────────────────────────────────

class VideoManager:
    """Manages OpenCV captures for multiple camera streams."""

    def __init__(self):
        self.caps = {}

    def open_episode(self, loader: DataLoader, ep_idx: int):
        self.close()
        for key in loader.video_keys:
            p = loader.video_path(ep_idx, key)
            if os.path.isfile(p):
                cap = cv2.VideoCapture(p)
                if cap.isOpened():
                    self.caps[key] = cap

    def read_frame(self, key, frame_idx):
        if key not in self.caps:
            return None
        cap = self.caps[key]
        cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if cur != frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def close(self):
        for cap in self.caps.values():
            cap.release()
        self.caps.clear()

    def __del__(self):
        self.close()


# ─── 3D Trajectory Widget ────────────────────────────────────────────────────

class GLTrajectoryWidget(gl.GLViewWidget if HAS_GL else QWidget):
    """OpenGL-based 3D EE trajectory viewer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        if not HAS_GL:
            return
        self.setBackgroundColor(C['crust'])

        # Reference grid
        grid = gl.GLGridItem()
        grid.setSize(0.3, 0.3)
        grid.setSpacing(0.03, 0.03)
        grid.setColor((100, 100, 100, 40))
        self.addItem(grid)

        # Axis indicator
        ax = gl.GLAxisItem()
        ax.setSize(0.04, 0.04, 0.04)
        self.addItem(ax)

        self._items = []
        self._psm1_pos = self._psm2_pos = None
        self.setCameraPosition(distance=0.3, elevation=30, azimuth=45)

    def _clear_items(self):
        for item in self._items:
            self.removeItem(item)
        self._items.clear()

    def set_trajectory(self, psm1, psm2, act_psm1=None, act_psm2=None):
        self._clear_items()
        self._psm1_pos = psm1
        self._psm2_pos = psm2
        n1, n2 = len(psm1), len(psm2)

        # Gradient colors: dim->bright to show time direction
        def grad(base_hex, n, max_alpha=1.0):
            r, g, b, _ = hex_to_gl_color(base_hex)
            c = np.zeros((n, 4), dtype=np.float32)
            c[:, 0], c[:, 1], c[:, 2] = r, g, b
            c[:, 3] = np.linspace(0.15, max_alpha, n)
            return c

        # --- State trajectories (solid, thick) ---
        line1 = gl.GLLinePlotItem(pos=psm1, color=grad(C['blue'], n1),
                                  width=2.5, antialias=True)
        self.addItem(line1)
        self._items.append(line1)

        line2 = gl.GLLinePlotItem(pos=psm2, color=grad(C['peach'], n2),
                                  width=2.5, antialias=True)
        self.addItem(line2)
        self._items.append(line2)

        # --- Action trajectories (thin, semi-transparent) ---
        if act_psm1 is not None:
            na1 = len(act_psm1)
            aline1 = gl.GLLinePlotItem(pos=act_psm1,
                                       color=grad(C['sky'], na1, max_alpha=0.6),
                                       width=1.2, antialias=True)
            self.addItem(aline1)
            self._items.append(aline1)

        if act_psm2 is not None:
            na2 = len(act_psm2)
            aline2 = gl.GLLinePlotItem(pos=act_psm2,
                                       color=grad(C['yellow'], na2, max_alpha=0.6),
                                       width=1.2, antialias=True)
            self.addItem(aline2)
            self._items.append(aline2)

        # Start markers
        start1 = gl.GLScatterPlotItem(pos=psm1[:1], color=hex_to_gl_color(C['green']),
                                       size=10)
        start2 = gl.GLScatterPlotItem(pos=psm2[:1], color=hex_to_gl_color(C['green']),
                                       size=10)
        self.addItem(start1)
        self.addItem(start2)
        self._items.extend([start1, start2])

        # End markers
        end1 = gl.GLScatterPlotItem(pos=psm1[-1:], color=hex_to_gl_color(C['red']),
                                     size=10)
        end2 = gl.GLScatterPlotItem(pos=psm2[-1:], color=hex_to_gl_color(C['red']),
                                     size=10)
        self.addItem(end1)
        self.addItem(end2)
        self._items.extend([end1, end2])

        # Current position markers (large, white-ish)
        self._cur1 = gl.GLScatterPlotItem(
            pos=psm1[:1], color=hex_to_gl_color(C['rosewater']), size=14)
        self._cur2 = gl.GLScatterPlotItem(
            pos=psm2[:1], color=hex_to_gl_color(C['rosewater']), size=14)
        self.addItem(self._cur1)
        self.addItem(self._cur2)
        self._items.extend([self._cur1, self._cur2])

        # Center camera on data (include action pts if available)
        pts_list = [psm1, psm2]
        if act_psm1 is not None:
            pts_list.append(act_psm1)
        if act_psm2 is not None:
            pts_list.append(act_psm2)
        all_pts = np.vstack(pts_list)
        center = all_pts.mean(axis=0)
        span = np.ptp(all_pts, axis=0).max()
        self.opts['center'] = pg.Vector(center[0], center[1], center[2])
        self.setCameraPosition(distance=span * 3.0)

    def update_cursor(self, idx):
        if self._psm1_pos is None:
            return
        idx = min(idx, len(self._psm1_pos) - 1)
        self._cur1.setData(pos=self._psm1_pos[idx:idx + 1])
        self._cur2.setData(pos=self._psm2_pos[idx:idx + 1])


class MplTrajectoryWidget(FigureCanvasQTAgg if not HAS_GL else QWidget):
    """Matplotlib fallback 3D trajectory viewer."""

    def __init__(self, parent=None):
        if HAS_GL:
            super().__init__(parent)
            return
        self.fig = Figure(facecolor=C['crust'], dpi=90)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self._style_ax()
        self._psm1_pos = self._psm2_pos = None
        self._cur1 = self._cur2 = None

    def _style_ax(self):
        self.ax.set_facecolor(C['crust'])
        for axis in ['x', 'y', 'z']:
            getattr(self.ax, f'set_{axis}label')(axis.upper(), color=C['subtext'])
            getattr(self.ax, f'tick_params')(axis=axis, colors=C['surface2'])
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

    def set_trajectory(self, psm1, psm2, act_psm1=None, act_psm2=None):
        self.ax.clear()
        self._style_ax()
        self._psm1_pos, self._psm2_pos = psm1, psm2
        # State trajectories (solid)
        self.ax.plot(psm1[:, 0], psm1[:, 1], psm1[:, 2],
                     color=C['blue'], lw=1.5, label='PSM1 State', alpha=0.8)
        self.ax.plot(psm2[:, 0], psm2[:, 1], psm2[:, 2],
                     color=C['peach'], lw=1.5, label='PSM2 State', alpha=0.8)
        # Action trajectories (thin, dashed)
        if act_psm1 is not None:
            self.ax.plot(act_psm1[:, 0], act_psm1[:, 1], act_psm1[:, 2],
                         color=C['sky'], lw=1.0, ls='--', label='PSM1 Action', alpha=0.5)
        if act_psm2 is not None:
            self.ax.plot(act_psm2[:, 0], act_psm2[:, 1], act_psm2[:, 2],
                         color=C['yellow'], lw=1.0, ls='--', label='PSM2 Action', alpha=0.5)
        # Start/end markers
        self.ax.scatter(*psm1[0], c=C['green'], s=60, zorder=5)
        self.ax.scatter(*psm2[0], c=C['green'], s=60, zorder=5)
        self.ax.scatter(*psm1[-1], c=C['red'], s=60, zorder=5)
        self.ax.scatter(*psm2[-1], c=C['red'], s=60, zorder=5)
        self._cur1 = self.ax.scatter(*psm1[0], c=C['rosewater'], s=100, zorder=10)
        self._cur2 = self.ax.scatter(*psm2[0], c=C['rosewater'], s=100, zorder=10)
        self.ax.legend(loc='upper right', fontsize=8, facecolor=C['surface0'],
                       edgecolor=C['surface1'], labelcolor=C['text'])
        self.draw()

    def update_cursor(self, idx):
        if self._psm1_pos is None:
            return
        idx = min(idx, len(self._psm1_pos) - 1)
        self._cur1._offsets3d = ([self._psm1_pos[idx, 0]],
                                  [self._psm1_pos[idx, 1]],
                                  [self._psm1_pos[idx, 2]])
        self._cur2._offsets3d = ([self._psm2_pos[idx, 0]],
                                  [self._psm2_pos[idx, 1]],
                                  [self._psm2_pos[idx, 2]])
        self.draw_idle()


# ─── Joint Plots Widget ──────────────────────────────────────────────────────

class JointPlotsWidget(pg.GraphicsLayoutWidget):
    """Time-series plots: position, orientation, jaw for PSM1 + PSM2."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground(C['base'])
        self._build_plots()
        self._cursor_lines = []
        self._data_items = []

    def _build_plots(self):
        label_style = {'color': C['subtext'], 'font-size': '10px'}

        # Row 0: Position
        self.p_pos = self.addPlot(row=0, col=0, title='Position (m)')
        self.p_pos.setLabel('left', 'pos', **label_style)
        self.p_pos.showGrid(x=True, y=True, alpha=0.15)
        self.p_pos.addLegend(offset=(10, 10), labelTextSize='9pt',
                             brush=pg.mkBrush(C['surface0']))

        # Row 1: Orientation
        self.p_ori = self.addPlot(row=1, col=0, title='Orientation (quat)')
        self.p_ori.setLabel('left', 'quat', **label_style)
        self.p_ori.showGrid(x=True, y=True, alpha=0.15)
        self.p_ori.addLegend(offset=(10, 10), labelTextSize='9pt',
                             brush=pg.mkBrush(C['surface0']))
        self.p_ori.setXLink(self.p_pos)

        # Row 2: Jaw
        self.p_jaw = self.addPlot(row=2, col=0, title='Jaw Angle')
        self.p_jaw.setLabel('left', 'jaw', **label_style)
        self.p_jaw.setLabel('bottom', 'Frame', **label_style)
        self.p_jaw.showGrid(x=True, y=True, alpha=0.15)
        self.p_jaw.addLegend(offset=(10, 10), labelTextSize='9pt',
                             brush=pg.mkBrush(C['surface0']))
        self.p_jaw.setXLink(self.p_pos)

        self._plots = [self.p_pos, self.p_ori, self.p_jaw]
        for p in self._plots:
            p.getAxis('bottom').setPen(pg.mkPen(C['surface2']))
            p.getAxis('left').setPen(pg.mkPen(C['surface2']))
            title_item = p.titleLabel
            title_item.setText(p.titleLabel.text, color=C['text'], size='11pt')

    def set_data(self, states, actions=None):
        """Set episode data. states shape: (N, 16)."""
        for p in self._plots:
            p.clear()
            # Re-add legend after clear
        self._cursor_lines.clear()
        self._data_items.clear()

        # Re-add legends
        for p in self._plots:
            if p.legend:
                p.legend.scene().removeItem(p.legend)
                p.legend = None
            p.addLegend(offset=(10, 10), labelTextSize='9pt',
                        brush=pg.mkBrush(C['surface0']))

        n = len(states)
        x = np.arange(n)

        # Position: PSM1 (0:3), PSM2 (8:11)
        for i, lbl in enumerate(POS_LABELS):
            self.p_pos.plot(x, states[:, i], pen=pg.mkPen(PSM1_POS_C[i], width=1.5),
                           name=f'P1 {lbl}')
            self.p_pos.plot(x, states[:, 8 + i],
                           pen=pg.mkPen(PSM2_POS_C[i], width=1.5, style=Qt.DashLine),
                           name=f'P2 {lbl}')

        # Orientation: PSM1 (3:7), PSM2 (11:15)
        for i, lbl in enumerate(ORI_LABELS):
            self.p_ori.plot(x, states[:, 3 + i], pen=pg.mkPen(PSM1_ORI_C[i], width=1.5),
                           name=f'P1 {lbl}')
            self.p_ori.plot(x, states[:, 11 + i],
                           pen=pg.mkPen(PSM2_ORI_C[i], width=1.5, style=Qt.DashLine),
                           name=f'P2 {lbl}')

        # Jaw: PSM1 (7), PSM2 (15)
        self.p_jaw.plot(x, states[:, 7], pen=pg.mkPen(C['blue'], width=2),
                        name='PSM1 Jaw')
        self.p_jaw.plot(x, states[:, 15], pen=pg.mkPen(C['peach'], width=2),
                        name='PSM2 Jaw')

        # Cursor lines
        pen = pg.mkPen(C['rosewater'], width=1.5, style=Qt.DashLine)
        for p in self._plots:
            vline = pg.InfiniteLine(pos=0, angle=90, pen=pen)
            p.addItem(vline)
            self._cursor_lines.append(vline)

        # Fit view
        self.p_pos.setXRange(0, n - 1, padding=0.02)

    def update_cursor(self, idx):
        for vl in self._cursor_lines:
            vl.setValue(idx)


# ─── Main Window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self, dataset_path):
        super().__init__()
        self.setWindowTitle('SURPASS LeRobot Visualizer')
        self.setMinimumSize(1280, 800)
        self.resize(1600, 950)

        self.loader = DataLoader(dataset_path)
        self.video_mgr = VideoManager()
        self.playing = False
        self.speed_idx = SPEED_OPTIONS.index(1.0)
        self.current_frame = 0
        self.ep_data = None

        self._build_ui()
        self._build_menu()
        self._setup_timer()
        self._apply_style()

        # Load first episode
        self._on_episode_changed(0)

    # ── UI Construction ──────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # --- Control Bar ---
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)

        ctrl.addWidget(QLabel('Episode:'))
        self.ep_combo = QComboBox()
        for ep in self.loader.episodes:
            idx = ep['episode_index']
            length = ep.get('length', '?')
            self.ep_combo.addItem(f'Ep {idx:03d}  ({length} frames)', idx)
        self.ep_combo.currentIndexChanged.connect(self._on_episode_changed)
        ctrl.addWidget(self.ep_combo)

        ctrl.addSpacing(12)
        self.play_btn = QPushButton('▶  Play')
        self.play_btn.setObjectName('playBtn')
        self.play_btn.clicked.connect(self._toggle_play)
        ctrl.addWidget(self.play_btn)

        self.step_back_btn = QPushButton('◀')
        self.step_back_btn.setFixedWidth(36)
        self.step_back_btn.clicked.connect(lambda: self._step(-1))
        ctrl.addWidget(self.step_back_btn)

        self.step_fwd_btn = QPushButton('▶')
        self.step_fwd_btn.setFixedWidth(36)
        self.step_fwd_btn.clicked.connect(lambda: self._step(1))
        ctrl.addWidget(self.step_fwd_btn)

        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel('Speed:'))
        self.speed_combo = QComboBox()
        for s in SPEED_OPTIONS:
            self.speed_combo.addItem(f'{s:.2g}x')
        self.speed_combo.setCurrentIndex(self.speed_idx)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        ctrl.addWidget(self.speed_combo)

        ctrl.addStretch()
        self.frame_lbl = QLabel('Frame: 0 / 0')
        self.frame_lbl.setObjectName('frameLabel')
        ctrl.addWidget(self.frame_lbl)

        root.addLayout(ctrl)

        # --- Main Splitter (vertical) ---
        main_split = QSplitter(Qt.Vertical)
        main_split.setHandleWidth(3)

        # Upper half: videos + 3D trajectory
        upper_split = QSplitter(Qt.Horizontal)
        upper_split.setHandleWidth(3)

        # Video grid
        vid_group = QGroupBox('Camera Feeds')
        vid_layout = QGridLayout(vid_group)
        vid_layout.setSpacing(3)
        vid_layout.setContentsMargins(4, 16, 4, 4)
        self.vid_labels = {}
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for i, key in enumerate(sorted(VIDEO_DISPLAY_NAMES.keys())):
            name = VIDEO_DISPLAY_NAMES.get(key, key.split('.')[-1])
            container = QWidget()
            vl = QVBoxLayout(container)
            vl.setSpacing(1)
            vl.setContentsMargins(0, 0, 0, 0)
            title = QLabel(name)
            title.setObjectName('videoTitle')
            title.setAlignment(Qt.AlignCenter)
            vl.addWidget(title)
            lbl = QLabel()
            lbl.setObjectName('videoLabel')
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(240, 135)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            vl.addWidget(lbl)
            self.vid_labels[key] = lbl
            r, c_ = positions[i]
            vid_layout.addWidget(container, r, c_)
        upper_split.addWidget(vid_group)

        # 3D trajectory
        traj_group = QGroupBox('3D EE Trajectory')
        traj_layout = QVBoxLayout(traj_group)
        traj_layout.setContentsMargins(4, 16, 4, 4)
        if HAS_GL:
            self.traj_widget = GLTrajectoryWidget()
        else:
            self.traj_widget = MplTrajectoryWidget()
        traj_layout.addWidget(self.traj_widget)

        # Legend for 3D
        legend_layout = QHBoxLayout()
        for name, color in [('PSM1 State', C['blue']), ('PSM2 State', C['peach']),
                            ('PSM1 Action', C['sky']), ('PSM2 Action', C['yellow']),
                            ('Start', C['green']), ('End', C['red']),
                            ('Current', C['rosewater'])]:
            dot = QLabel(f'\u25cf {name}')
            dot.setStyleSheet(f'color: {color}; font-size: 11px; font-weight: bold;')
            legend_layout.addWidget(dot)
        legend_layout.addStretch()
        traj_layout.addLayout(legend_layout)
        upper_split.addWidget(traj_group)
        upper_split.setSizes([600, 500])

        main_split.addWidget(upper_split)

        # Lower half: joint plots
        plots_group = QGroupBox('Joint State Time-Series')
        plots_layout = QVBoxLayout(plots_group)
        plots_layout.setContentsMargins(4, 16, 4, 4)
        self.joint_plots = JointPlotsWidget()
        plots_layout.addWidget(self.joint_plots)

        # Show action overlay toggle
        action_bar = QHBoxLayout()
        info_lbl = QLabel('PSM1: solid lines  |  PSM2: dashed lines')
        info_lbl.setStyleSheet(f'color: {C["subtext"]}; font-size: 10px;')
        action_bar.addWidget(info_lbl)
        action_bar.addStretch()
        plots_layout.addLayout(action_bar)

        main_split.addWidget(plots_group)
        main_split.setSizes([500, 350])

        root.addWidget(main_split, 1)

        # --- Frame Slider ---
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self._on_slider)
        root.addWidget(self.slider)

        # --- Status Bar ---
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage('Ready')

    def _build_menu(self):
        menu = self.menuBar()
        view_menu = menu.addMenu('View')
        self.action_show_actions = QAction('Show Action Overlay', self, checkable=True)
        view_menu.addAction(self.action_show_actions)
        self.action_show_actions.triggered.connect(
            lambda: self._reload_current_episode())

    def _setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer_tick)

    def _apply_style(self):
        self.setStyleSheet(f"""
        QMainWindow, QWidget {{
            background-color: {C['base']}; color: {C['text']};
            font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 13px;
        }}
        QGroupBox {{
            border: 1px solid {C['surface1']}; border-radius: 6px;
            margin-top: 8px; padding-top: 14px; font-weight: bold;
            color: {C['subtext']};
        }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
        QComboBox {{
            background-color: {C['surface0']}; border: 1px solid {C['surface1']};
            border-radius: 4px; padding: 4px 8px; color: {C['text']}; min-width: 100px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {C['surface0']}; color: {C['text']};
            selection-background-color: {C['surface1']};
        }}
        QPushButton {{
            background-color: {C['surface0']}; border: 1px solid {C['surface1']};
            border-radius: 4px; padding: 5px 12px; color: {C['text']}; font-weight: bold;
        }}
        QPushButton:hover {{ background-color: {C['surface1']}; }}
        QPushButton:pressed {{ background-color: {C['surface2']}; }}
        QPushButton#playBtn {{
            background-color: {C['green']}; color: {C['crust']};
            font-size: 13px; min-width: 80px;
        }}
        QPushButton#playBtn:hover {{ background-color: {C['teal']}; }}
        QSlider::groove:horizontal {{
            height: 6px; background: {C['surface1']}; border-radius: 3px;
        }}
        QSlider::handle:horizontal {{
            background: {C['blue']}; width: 16px; height: 16px;
            margin: -5px 0; border-radius: 8px;
        }}
        QSlider::sub-page:horizontal {{
            background: {C['blue']}; border-radius: 3px;
        }}
        QLabel#frameLabel {{
            font-family: 'Consolas', monospace; font-size: 13px; color: {C['subtext']};
        }}
        QLabel#videoLabel {{
            background-color: {C['crust']}; border: 1px solid {C['surface0']};
            border-radius: 4px;
        }}
        QLabel#videoTitle {{
            font-size: 11px; font-weight: bold; color: {C['subtext']}; padding: 2px;
        }}
        QStatusBar {{
            background-color: {C['mantle']}; color: {C['subtext']}; font-size: 12px;
        }}
        QMenuBar {{
            background-color: {C['mantle']}; color: {C['text']};
        }}
        QMenuBar::item:selected {{ background-color: {C['surface0']}; }}
        QMenu {{
            background-color: {C['surface0']}; color: {C['text']};
            border: 1px solid {C['surface1']};
        }}
        QMenu::item:selected {{ background-color: {C['surface1']}; }}
        QSplitter::handle {{ background-color: {C['surface1']}; }}
        """)

    # ── Episode / Frame Logic ────────────────────────────────────────

    def _on_episode_changed(self, combo_idx):
        ep_idx = self.ep_combo.itemData(combo_idx)
        if ep_idx is None:
            return
        self._stop_playback()
        self.ep_data = self.loader.load_episode(ep_idx)
        self.video_mgr.open_episode(self.loader, ep_idx)

        n = self.ep_data['n_frames']
        self.slider.setMaximum(max(n - 1, 0))
        self.current_frame = 0
        self.slider.setValue(0)

        # Update trajectory
        states = self.ep_data['states']
        psm1_pos = states[:, 0:3].copy()
        psm2_pos = states[:, 8:11].copy()
        # Extract action positions if available
        actions = self.ep_data.get('actions')
        act_psm1 = actions[:, 0:3].copy() if actions is not None else None
        act_psm2 = actions[:, 8:11].copy() if actions is not None else None
        self.traj_widget.set_trajectory(psm1_pos, psm2_pos, act_psm1, act_psm2)

        # Update joint plots
        self.joint_plots.set_data(states)

        self._update_display()
        ep_info = self.loader.episodes[combo_idx]
        self.status.showMessage(
            f'Episode {ep_idx}  |  {n} frames  |  '
            f'Task: {ep_info.get("tasks", ["?"])[0]}  |  FPS: {self.loader.fps}'
        )

    def _reload_current_episode(self):
        self._on_episode_changed(self.ep_combo.currentIndex())

    def _set_frame(self, idx):
        if self.ep_data is None:
            return
        idx = max(0, min(idx, self.ep_data['n_frames'] - 1))
        self.current_frame = idx
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self._update_display()

    def _update_display(self):
        if self.ep_data is None:
            return
        idx = self.current_frame
        n = self.ep_data['n_frames']

        # Update frame label
        ts = self.ep_data['timestamps']
        t_str = f'{ts[idx]:.3f}s' if idx < len(ts) else '?'
        self.frame_lbl.setText(f'Frame: {idx} / {n - 1}  ({t_str})')

        # Update videos
        for key, lbl in self.vid_labels.items():
            frame = self.video_mgr.read_frame(key, idx)
            if frame is not None:
                w, h = lbl.width(), lbl.height()
                if w > 10 and h > 10:
                    pm = np_to_qpixmap(frame, w, h)
                    lbl.setPixmap(pm)
            else:
                lbl.setText('No Video')

        # Update trajectory cursor
        self.traj_widget.update_cursor(idx)

        # Update plot cursors
        self.joint_plots.update_cursor(idx)

    def _on_slider(self, value):
        self.current_frame = value
        self._update_display()

    # ── Playback ─────────────────────────────────────────────────────

    def _toggle_play(self):
        if self.playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        if self.ep_data is None:
            return
        # If at end, restart
        if self.current_frame >= self.ep_data['n_frames'] - 1:
            self.current_frame = 0
        self.playing = True
        self.play_btn.setText('⏸  Pause')
        speed = SPEED_OPTIONS[self.speed_idx]
        interval = max(1, int(1000.0 / (self.loader.fps * speed)))
        self.timer.start(interval)

    def _stop_playback(self):
        self.playing = False
        self.play_btn.setText('▶  Play')
        self.timer.stop()

    def _on_timer_tick(self):
        if self.ep_data is None:
            return
        nxt = self.current_frame + 1
        if nxt >= self.ep_data['n_frames']:
            self._stop_playback()
            return
        self._set_frame(nxt)

    def _step(self, delta):
        self._stop_playback()
        self._set_frame(self.current_frame + delta)

    def _on_speed_changed(self, idx):
        self.speed_idx = idx
        if self.playing:
            self._stop_playback()
            self._start_playback()

    # ── Keyboard Shortcuts ───────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            self._toggle_play()
        elif key == Qt.Key_Right:
            self._step(1)
        elif key == Qt.Key_Left:
            self._step(-1)
        elif key == Qt.Key_Home:
            self._set_frame(0)
        elif key == Qt.Key_End:
            self._set_frame(self.ep_data['n_frames'] - 1 if self.ep_data else 0)
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            new_idx = min(self.speed_idx + 1, len(SPEED_OPTIONS) - 1)
            self.speed_combo.setCurrentIndex(new_idx)
        elif key in (Qt.Key_Minus, Qt.Key_Underscore):
            new_idx = max(self.speed_idx - 1, 0)
            self.speed_combo.setCurrentIndex(new_idx)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.timer.stop()
        self.video_mgr.close()
        super().closeEvent(event)


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Interactive LeRobot v2.1 dataset visualizer for DVRK data'
    )
    parser.add_argument(
        '--dataset-path', type=str,
        default=os.path.expanduser(
            r'~\.cache\huggingface\lerobot\SURPASS_Cholecystectomy'
        ),
        help='Path to the LeRobot v2.1 dataset directory'
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_path):
        print(f'Error: Dataset not found at: {args.dataset_path}')
        sys.exit(1)

    # pyqtgraph config
    pg.setConfigOptions(antialias=True, useOpenGL=True)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    win = MainWindow(args.dataset_path)
    win.show()

    print(f'Loaded dataset: {args.dataset_path}')
    print(f'  Episodes: {win.loader.num_episodes}')
    print(f'  FPS: {win.loader.fps}')
    print(f'  Video streams: {", ".join(win.loader.video_keys)}')
    print(f'  3D backend: {"OpenGL" if HAS_GL else "Matplotlib"}')
    print()
    print('Controls: Space=Play/Pause, Left/Right=Step, Home/End=Jump, +/-=Speed')

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
