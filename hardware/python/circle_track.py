#%% imports
import datetime
import json
import logging
import os
import sys
import threading
import time
import typing
from copy import copy

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import yaml
from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QCloseEvent, QFontDatabase, QImage, QPixmap, QTextOption
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ezTrack import Video
from pyMaze import Maze, null_callback


class ConsoleStream(QObject):
    entry = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()

    def write(self, msg):
        self.entry.emit(msg.strip("\n"))

    def flush(self):
        pass


class CircleTrack(QMainWindow):
    def __init__(
        self,
        config_file: str,
        parent: typing.Optional[QWidget] = None,
        flags: typing.Union[Qt.WindowFlags, Qt.WindowType] = Qt.WindowFlags(),
    ) -> None:
        # init
        super().__init__(parent=parent, flags=flags)
        with open(config_file) as ymlf:
            self._config = yaml.safe_load(ymlf)
        self._click_dat, self._click_fs = sf.read("./assets/click.wav")
        self._bk_dat, self._bk_fs = None, None
        self._context = None
        self._msconfig = None
        self._started = False
        self._displaying = True
        self._rois = None
        self._rw_rois = None
        self._sdevice = (None, self._config.get("sound_device"))
        self._mouse = None
        self._rwAfter = 0
        # gui stuff
        cstream = ConsoleStream()
        wConsole = QTextEdit()
        wConsole.setReadOnly(True)
        wConsole.setMinimumWidth(500)
        wConsole.setWordWrapMode(QTextOption.NoWrap)
        wConsole.setCurrentFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        cstream.entry.connect(wConsole.append)
        wContext = QComboBox()
        wContext.addItems(self._config["contexts"])
        wContext.setCurrentIndex(-1)
        wContext.currentTextChanged.connect(self.setContext)
        wContext.currentTextChanged.connect(self.checkReady)
        lContext = QLabel("&Context:")
        lContext.setBuddy(wContext)
        layContext = QHBoxLayout()
        layContext.addWidget(lContext)
        layContext.addWidget(wContext)
        wMs = QComboBox()
        wMs.addItems([f for f in os.listdir(self._config["miniscope_config_dir"])])
        wMs.setCurrentIndex(-1)
        wMs.currentTextChanged.connect(self.setMsConfig)
        wMs.currentTextChanged.connect(self.checkReady)
        lMs = QLabel("&Miniscope config:")
        lMs.setBuddy(wMs)
        layMs = QHBoxLayout()
        layMs.addWidget(lMs)
        layMs.addWidget(wMs)
        self._wFlush = QCheckBox("FLUSH")
        self._wFlush.stateChanged.connect(self.setFlush)
        self._wPrime = QPushButton("Prime Rewards")
        self._wPrime.clicked.connect(self.onPrime)
        self._wPrime.setEnabled(False)
        self._wStart = QPushButton("Start")
        self._wStart.clicked.connect(self.onStart)
        self._wStart.setEnabled(False)
        self._calib = False
        self._wCalib = QPushButton("Calibrate")
        self._wCalib.clicked.connect(self.onCalib)
        self._wCalib.setEnabled(True)
        self._wReset = QPushButton("Reset Tracking")
        self._wReset.clicked.connect(self.onReset)
        self._wReset.setEnabled(False)
        self._wProg = QProgressBar()
        self._wProg.setMaximum(int(self._config["session_length"] * 60))
        self._wVid = QLabel()
        layButtons = QGridLayout()
        layButtons.addWidget(self._wCalib, 1, 1)
        layButtons.addWidget(self._wReset, 1, 2)
        layButtons.addWidget(self._wPrime, 2, 1)
        layButtons.addWidget(self._wStart, 2, 2)
        layConfig = QVBoxLayout()
        layConfig.addLayout(layContext)
        layConfig.addLayout(layMs)
        layConfig.addWidget(self._wFlush)
        layConfig.addLayout(layButtons)
        layConfig.addWidget(self._wProg)
        layMonitor = QVBoxLayout()
        layMonitor.addWidget(wConsole)
        layMonitor.addWidget(self._wVid)
        layMain = QHBoxLayout()
        layMain.addLayout(layConfig)
        layMain.addLayout(layMonitor)
        layMain.setStretchFactor(layConfig, 1)
        layMain.setStretchFactor(wConsole, 3)
        wid = QWidget(self)
        self.setCentralWidget(wid)
        wid.setLayout(layMain)
        self.setWindowTitle("Circle Track GUI")
        self.show()
        # eztrack stuff
        dim = self._config["eztrack_vid_dim"]
        roi = self._config["eztrack_mask"]
        self._center = tuple(roi["center"])
        self._mask = create_ring_mask(
            dim[0], dim[1], self._center, roi["r0"], roi["r1"]
        )
        self._r0 = roi["r0"]
        self._r1 = roi["r1"]
        self._vid = Video(src=self._config["eztrack_vid_src"])
        self._vid.start()
        self._vid.track_method = "dark"
        self._vid.track_thresh = 99.9
        self._vid.track_window_use = True
        self._vid.track_window_sz = 100
        self._vid.track_window_wt = 0.2
        self._vid.track_rmvwire = True
        self._vid.track_rmvwire_krn = 3
        self._vid.mask = {"mask": self._mask}
        self._tt = threading.Thread(target=self.trackLoop)
        self._tt.start()
        self._roi_last = None
        # maze stuff
        self._maze = Maze(self._config)
        ch = logging.StreamHandler(cstream)
        ch.setFormatter(logging.Formatter(self._config["LOG_FORMAT"]))
        ch.setLevel(self._config["PRINT_LEVEL"])
        self._maze.logger.addHandler(ch)
        self._maze.logger.info("using config file: {}".format(config_file))
        self._maze.attach_interpreter(self.interpreter)
        self._maze.attach_callback("evt_null", null_callback)
        self._maze.add_state("isLicking", False)
        self._maze.add_state("lastReward", None)
        self._maze.add_state("rewarded", [])
        self._maze.add_state("lastLick")
        self._maze.add_state("nReward", 0)
        self._maze.add_state("nLick", 0)
        self._maze.add_state("isRunning", False)
        self._maze.add_state("correctDirection", False)
        self.setFlush(False)
        self._maze.start()

    def setRewards(self):
        if self._context is not None and self._mouse is not None:
            try:
                self._rw_ports = self._config["reward_port"][self._context][self._mouse]
            except KeyError:
                self._maze.logger.warning("Falling back to default config")
                self._rw_ports = self._config["reward_port"][self._context]["ALL"]
            self._maze.logger.info("Rewarding ports: {}".format(self._rw_ports))
            rw_roi = sorted([int(r[-1]) for r in self._rw_ports])
            rois = []
            for ir, r in enumerate(rw_roi):
                if ir == 0:
                    rois.append((rw_roi[ir - 1] + self._config["nport"] + r) / 2)
                else:
                    rois.append((rw_roi[ir - 1] + r) / 2)
                rois.append(r)
            self._rois = np.sort(
                (
                    -np.array(rois)
                    * 360
                    / self._config["nport"]
                    * self._config["reward_direction"]
                    + self._config["roi_offset"]
                )
                % 360
            )
            self._rw_rois = np.sort(
                (
                    -np.array(rw_roi)
                    * 360
                    / self._config["nport"]
                    * self._config["reward_direction"]
                    + self._config["roi_offset"]
                )
                % 360
            )
            self._wPrime.setEnabled(True)

    def onStart(self):
        self._started = True
        self._wFlush.setEnabled(False)
        dpath = self._msconfig["dataDirectory"]
        for ds in self._msconfig["directoryStructure"]:
            if ds == "date":
                d = datetime.date.today().strftime("%Y_%m_%d")
            elif ds == "time":
                d = datetime.datetime.now().strftime("%H_%M_%S")
            else:
                d = self._msconfig[ds]
            dpath = os.path.join(dpath, d)
        self._maze.update_dpath(dpath)
        self._maze.wait_ready()
        self._maze.states["isRunning"] = True
        if (self._bk_dat is not None) and (self._bk_fs is not None):
            sd.play(self._bk_dat, self._bk_fs, loop=True, device=self._sdevice)
        self._maze.states["nReward"] = 0
        self._maze.states["lastReward"] = None
        self._maze.states["rewarded"] = []
        self._maze.logger.info("session started")
        timer = QTimer(self)
        timer.timeout.connect(self.setProgress)
        timer.singleShot(int(self._config["session_length"] * 6e4), self.onFinish)
        self._tstart = time.time()
        self._maze.write_data({"timestamp": self._tstart, "event": "probe length: {}".format(self._rwAfter)})
        self._maze.write_data({"timestamp": self._tstart, "event": "initializing", "data": self._rw_ports[0]})
        self._maze.write_data({"timestamp": self._tstart, "event": "initializing", "data": self._rw_ports[1]})
        self._maze.write_data({"timestamp": self._tstart, "event": "START"})
        self._maze.digitalHigh("miniscope_ttl")
        timer.start()
        self._wStart.setEnabled(False)

    def onCalib(self):
        self._vid.ref_create(secs=self._config["eztrack_calib_sec"])
        self._wCalib.setEnabled(False)
        self._wReset.setEnabled(True)
        self._vid.track = True
        self._calib = True
        self.checkReady()

    def onFinish(self):
        self._maze.digitalLow("miniscope_ttl")
        self._maze.write_data({"timestamp": time.time(), "event": "TERMINATE"})
        self._maze.logger.info("session terminated")
        self._maze.logger.info(
            "total reward count: {}".format(self._maze.states["nReward"])
        )
        self._maze.states["isRunning"] = False
        sd.stop()
        self._started = False
        self._wFlush.setEnabled(True)

    def setProgress(self):
        self._wProg.setValue(int(time.time() - self._tstart))

    def setContext(self, ctx):
        self._context = ctx
        try:
            self._bk_dat, self._bk_fs = sf.read(self._config["background_sound"][ctx])
        except KeyError:
            self._bk_dat, self._bk_fs = None, None
        self.setRewards()

    def setMsConfig(self, config):
        try:
            with open(
                os.path.join(self._config["miniscope_config_dir"], config)
            ) as mconf_f:
                mconf = json.load(mconf_f)
            self._mouse = mconf["animal"]
            self._msconfig = mconf
            # randomize probe length
            if self._config["reward_after"][self._mouse]: 
                self._rwAfter = np.random.randint(45, 90)
            else:
                self._rwAfter = 0
            self._maze.logger.info("probe length: {}".format(self._rwAfter))
        except (FileNotFoundError, KeyError):
            self._maze.logger.warning("cannot read miniscope config")
        self.setRewards()
        

    def setFlush(self, flush):
        self._flush = flush
        if self._flush:
            self._maze.attach_callback("evt_lick", self.onLick_flush)
            self._maze.attach_callback("evt_release", self.onRelease_flush)
        else:
            self._maze.attach_callback("evt_lick", self.onLick_reward)
            self._maze.attach_callback("evt_release", null_callback)
        self.checkReady()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._started:
            confirm = QMessageBox.question(
                self,
                "Terminate GUI",
                "Session in progress! Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
        else:
            confirm = QMessageBox.Yes
        if confirm == QMessageBox.Yes:
            self._maze.terminate()
            self._displaying = False
            self._tt.join()
            self._vid.release()
            event.accept()
        else:
            event.ignore()

    def checkReady(self):
        if (
            (self._context is not None)
            and (self._msconfig is not None)
            and (not self._flush)
        ):
            self._wStart.setEnabled(True)
        else:
            self._wStart.setEnabled(False)

    def interpreter(self, maze, sig):
        try:
            evt = {0: "released", 1: "licked"}[sig[1]]
            try:
                port = self._config["touch_pin"][int(sig[0])]
            except KeyError:
                maze.logger.warning("pin {} touched but not a port".format(sig[0]))
                return "evt_null"
            maze.logger.debug("port {} {}".format(port, evt))
            if evt == "licked":
                return "evt_lick"
            else:
                return "evt_release"
        except:
            maze.logger.error("don't understand {}".format(sig))
            return "evt_null"

    def onLick_reward(self, maze, sig, ts):
        port = self._config["touch_pin"][int(sig[0])]
        if port == maze.states["lastLick"]:
            maze.states["nLick"] += 1
        else:
            maze.states["nLick"] = 1
            maze.states["lastLick"] = port
        if (not maze.states["isLicking"]) and maze.states["isRunning"]:
            maze.states["isLicking"] = True
        else:
            return
        if not self._started:
            return
        if (
            (port in self._rw_ports)
            and (port not in maze.states["rewarded"])
            and (port != maze.states["lastReward"])
            and (maze.states["nLick"] >= self._config["lick_threshold"])
            and (time.time() - self._tstart > self._rwAfter) #removed *60 as this is already in minutes
            and (maze.states["correctDirection"])
        ):
            maze.states["rewarded"].append(port)
            maze.states["lastReward"] = port
            maze.states["nReward"] += 1
            maze.logger.info("rewarding port {}".format(port))
            maze.write_data({"timestamp": ts, "event": "REWARD", "data": port})
            sd.play(self._click_dat, self._click_fs, device=self._sdevice)
            maze.digitalHigh(port)
            time.sleep(self._config["reward_length"])
            maze.digitalLow(port)
            time.sleep(1)
            if (self._bk_dat is not None) and (self._bk_fs is not None):
                sd.play(self._bk_dat, self._bk_fs, loop=True, device=self._sdevice)
        else:
            maze.write_data({"timestamp": ts, "event": "LICK", "data": port})
        if len(maze.states["rewarded"]) >= len(self._rw_ports):
            maze.states["rewarded"] = []
        maze.states["isLicking"] = False

    def onLick_flush(self, maze, sig, ts):
        port = self._config["touch_pin"][int(sig[0])]
        maze.logger.info("{} on".format(port))
        maze.digitalHigh(port)

    def onRelease_flush(self, maze, sig, ts):
        port = self._config["touch_pin"][int(sig[0])]
        maze.logger.info("{} off".format(port))
        maze.digitalLow(port)

    def onReset(self):
        self._vid.track_window_use = False
        prev_yx = copy(self._vid.track_yx)
        while self._vid.track_yx == prev_yx:
            pass
        self._vid.track_window_use = True

    def onPrime(self):
        rw_ports = self._config["reward_port"][self._context][self._mouse]
        sd.play(self._click_dat, self._click_fs, device=self._sdevice)
        for port in rw_ports:
            self._maze.digitalHigh(port)
        time.sleep(self._config["reward_length"])
        for port in rw_ports:
            self._maze.digitalLow(port)

    def pixmap_fromarray(self, img):
        w, h = img.shape
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        qimg = QImage(img.data, h, w, 3 * h, QImage.Format_RGB888)
        qpixmap = QPixmap(qimg)
        return qpixmap

    def trackLoop(self):
        while self._displaying:
            if self._vid.frame is None:
                continue
            frame = self._vid.frame.copy()
            frame[self._mask] = (frame[self._mask].astype(float) * 0.5).astype(np.uint8)
            yx = self._vid.track_yx
            if yx is not None:
                y, x = np.array(yx).astype(int)
                frame = cv2.drawMarker(
                    img=frame, position=(x, y), color=255, thickness=2
                )
                frame = cv2.line(frame, self._center, (x, y), 255, 1)
                ang = np.arctan2(self._center[1] - y, x - self._center[0])
                if ang < 0:
                    ang = 2 * np.pi + ang
                ang = np.degrees(ang)
                frame = cv2.putText(
                    frame,
                    str(int(ang)),
                    org=(frame.shape[1] - 50, frame.shape[0] - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=255,
                    thickness=2,
                )
                if self._started:
                    self._maze.write_data(
                        {
                            "timestamp": time.time(),
                            "event": "LOCATION",
                            "data": "X{}Y{}A{}".format(x, y, int(ang)),
                        }
                    )
            if self._rois is not None:
                for r in self._rois:
                    rr = np.deg2rad(r)
                    start = (
                        self._center[0] + self._r0 * np.cos(rr),
                        self._center[1] - self._r0 * np.sin(rr),
                    )
                    if r in self._rw_rois:
                        end = (
                            start[0] + 15 * np.cos(rr),
                            start[1] - 15 * np.sin(rr),
                        )
                    else:
                        end = (
                            start[0] - 10 * np.cos(rr),
                            start[1] + 10 * np.sin(rr),
                        )
                    frame = cv2.line(
                        frame,
                        np.array(start).astype(int),
                        np.array(end).astype(int),
                        255,
                        3,
                    )
                if yx is not None:
                    diff = ang - self._rois
                    idx = (np.abs(diff)).argmin()
                    cur_roi = self._rois[idx]
                    frame = cv2.ellipse(
                        frame,
                        self._center,
                        (self._r0, self._r0),
                        angle=0,
                        startAngle=360 - (cur_roi - 5),
                        endAngle=360 - (cur_roi + 5),
                        color=255,
                        thickness=3,
                    )
                    if cur_roi != self._roi_last:
                        try:
                            prev_roi = self._rois[
                                idx + np.sign(self._config["reward_direction"])
                            ]
                        except IndexError:
                            prev_roi = self._rois[
                                (idx + np.sign(self._config["reward_direction"]))
                                % len(self._rois)
                            ]
                        if self._roi_last == prev_roi:
                            self._maze.states["correctDirection"] = True
                        else:
                            self._maze.states["correctDirection"] = False
                        self._roi_last = cur_roi
                    dir_text = (
                        "Correct Direction"
                        if self._maze.states["correctDirection"]
                        else "Wrong Direction"
                    )
                    frame = cv2.putText(
                        frame,
                        dir_text,
                        org=(3, frame.shape[0] - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        color=255,
                        thickness=2,
                    )
            else:
                frame = cv2.line(
                    frame,
                    (self._center[0], self._center[1]),
                    (self._center[0] + self._r0, self._center[1]),
                    255,
                    2,
                )
            self._wVid.setPixmap(self.pixmap_fromarray(frame))
            time.sleep(1 / 60)


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


def create_ring_mask(h, w, center, r0, r1):
    m0 = create_circular_mask(h, w, center, r0)
    m1 = ~create_circular_mask(h, w, center, r1)
    return np.logical_or(m0, m1)


if __name__ == "__main__":
    try:
        configf = sys.argv[1]
    except IndexError:
        configf = "./maze3.yml"
    app = QApplication([])
    circle_track = CircleTrack(configf)
    sys.exit(app.exec_())
