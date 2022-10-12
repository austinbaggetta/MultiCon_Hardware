import datetime
import json
import logging
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import warnings

import pandas as pd
import serial
import yaml

USE_MP = False
# if sys.platform == "linux":
#     USE_MP = True


class Maze:
    def __init__(self, config) -> None:
        if USE_MP:
            self._manager = mp.Manager()
            self._tasks = self._manager.Queue()
            self.states = self._manager.dict()
            self._callbacks = self._manager.dict()
        else:
            self._tasks = queue.Queue()
            self.states = dict()
            self._callbacks = dict()
        self._interpreter = lambda x: x
        self.states["READY"] = False
        self.states["TERMINATE"] = False
        if type(config) is str:
            with open(config) as yamlf:
                self._config = yaml.safe_load(yamlf)
        elif type(config) is dict:
            self._config = config
        else:
            raise TypeError("config has to be a file path or dictionary")
        self._CMD_FLAG = bytes([self._config["CMD_FLAG"]])
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.NOTSET)
        self._lformater = logging.Formatter(self._config["LOG_FORMAT"])
        ch = logging.StreamHandler()
        ch.setFormatter(self._lformater)
        ch.setLevel(self._config["PRINT_LEVEL"])
        self.logger.addHandler(ch)
        self.update_dpath(config["DPATH"])

    def start(self) -> None:
        if USE_MP:
            self._ps = mp.Process(target=self._main)
        else:
            self._ps = threading.Thread(target=self._main)
        self._ps.start()

    def _send(self, sig, ts) -> None:
        self._tasks.put((ts, bytes(sig)))

    def _main(self) -> None:
        try:
            ser = serial.Serial(
                port=self._config["PORT"],
                baudrate=self._config["BAUDRATE"],
                timeout=self._config["TIMEOUT"],
                write_timeout=self._config["TIMEOUT"],
            )
        except:
            self.logger.error("cannot open serial port {}".format(self._config["PORT"]))
            self.states["READY"] = None
            return
        # handshake
        hstart = time.time()
        self.logger.info("handshake start")
        while True:
            ser.write(self._CMD_FLAG)
            data = ser.read()
            if data == self._CMD_FLAG:
                self.logger.info("handshake success")
                break
            if time.time() - hstart > self._config["TIMEOUT_HANDSHAKE"]:
                self.logger.error(
                    "handshake timeout on port {}".format(self._config["PORT"])
                )
                raise RuntimeError("handshake failed")
        # initialize all output pins
        for pname, pin in self._config["OUTPUT_PINS"].items():
            self.modeOut(pin)
        # main loop
        self.states["READY"] = True
        self.logger.info("maze ready")
        while True:
            # read signal
            while ser.in_waiting > 0:
                cur_sig = ser.read_until(expected=self._CMD_FLAG)
                cur_evt = self._interpreter(self, cur_sig)
                try:
                    cur_cb = self._callbacks[cur_evt]
                except KeyError:
                    self.logger.warning(
                        "no callback registered for event {}. proceeding".format(
                            cur_evt
                        )
                    )
                    continue
                ts = time.time()
                if USE_MP:
                    ps = mp.Process(target=cur_cb, args=(self, cur_sig, ts))
                else:
                    ps = threading.Thread(target=cur_cb, args=(self, cur_sig, ts))
                ps.start()
                self.logger.debug(
                    "exectued callback: {}, input signal: {}".format(cur_evt, cur_sig)
                )
            # send commands
            cur_ts = time.time()
            hold_tasks = []
            while not self._tasks.empty():
                ts, cmd = self._tasks.get()
                if ts <= cur_ts:
                    cmd = cmd + self._CMD_FLAG
                    ser.write(cmd)
                    self.logger.debug("sent command: {}".format(cmd))
                else:
                    hold_tasks.append((ts, cmd))
            for tsk in hold_tasks:
                self._tasks.put(tsk)
            # check for termination
            if self.states["TERMINATE"]:
                for pname, pin in self._config["OUTPUT_PINS"].items():
                    self.digitalLow(pin)
                self.logger.info("TERMINATE signal received.")
                ser.close()
                break

    def digitalHigh(self, pin, hold: float = None) -> None:
        if type(pin) is str:
            try:
                pin = self._config["OUTPUT_PINS"][pin]
            except KeyError:
                self.logger.error("pin not understood: {}".format(pin))
                return
        cur_ts = time.time()
        self._send((pin, 1), cur_ts)
        if hold is not None:
            self._send((pin, 0), cur_ts + hold)

    def digitalLow(self, pin, hold: float = None) -> None:
        if type(pin) is str:
            try:
                pin = self._config["OUTPUT_PINS"][pin]
            except KeyError:
                self.logger.error("pin not understood: {}".format(pin))
                return
        cur_ts = time.time()
        self._send((pin, 0), cur_ts)
        if hold is not None:
            self._send((pin, 1), cur_ts + hold)

    def modeOut(self, pin) -> None:
        if type(pin) is str:
            try:
                pin = self._config["OUTPUT_PINS"][pin]
            except KeyError:
                self.logger.error("pin not understood: {}".format(pin))
                return
        cur_ts = time.time()
        self._send((pin, 2), cur_ts)

    def wait_ready(self) -> None:
        if self.states["READY"] is not None:
            while not self.states["READY"]:
                pass

    def terminate(self) -> None:
        self.states["TERMINATE"] = True
        self._ps.join()

    def attach_interpreter(self, cb) -> None:
        self._interpreter = cb

    def attach_callback(self, evt, cb) -> None:
        self._callbacks[evt] = cb

    def add_state(self, key, val=None) -> None:
        self.states[key] = val

    def write_data(self, dat) -> None:
        self._data = self._data.append(dat, ignore_index=True)
        self._data.iloc[-1:].to_csv(self._datafile, mode="a", header=False, index=False)

    def update_dpath(self, dpath) -> None:
        self._dpath = dpath
        os.makedirs(self._dpath, exist_ok=True)
        fh = logging.FileHandler(
            os.path.join(self._dpath, self._config["LOG_FILE"]),
            mode="a",
            encoding="utf-8",
        )
        fh.setFormatter(self._lformater)
        fh.setLevel(self._config["LOG_LEVEL"])
        self.logger.addHandler(fh)
        self._datafile = os.path.join(self._dpath, self._config["DATA_FILE"])
        self._data = pd.DataFrame(columns=self._config["DATA_HEADER"])
        self._data.to_csv(self._datafile, index=False)
        self.logger.info("writing data to {}".format(self._dpath))


def null_callback(maze, sig, ts):
    pass
