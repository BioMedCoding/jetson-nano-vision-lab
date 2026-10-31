# -*- coding: utf-8 -*-
"""Wrapper per jetson-stats (jtop) gia' installato sul sistema.

Schema reale verificato su questo Jetson (jetson-stats 4.3.2): jetson.stats ha
chiavi 'CPU1'..'CPU4' (percento per-core), 'GPU' (percento), 'Temp CPU'/'Temp
GPU'/'Temp AO'/'Temp PLL'/'Temp thermal' (gradi C), 'Power TOT'/'Power
POM_5V_CPU'/'Power POM_5V_GPU' (mW), 'nvp model' (power mode). RAM totale/usata
e fan rpm NON sono in jetson.stats: servono jetson.memory['RAM'] (KB) e
jetson.fan['tegra_pwmfan']['rpm'|'speed'].

get_snapshot() non solleva mai eccezioni: se jtop non e' disponibile o il
servizio non risponde, ritorna jtop_ok=False con valori a None, cosi' la demo
non dipende da jtop per funzionare.
"""
import threading
import time

_EMPTY = {
    "jtop_ok": False, "cpu_percent": None, "gpu_percent": None,
    "ram_used_mb": None, "ram_total_mb": None, "temps": {},
    "fan_rpm": None, "fan_percent": None, "fan_profile": None,
    "power_mode": None, "power_mw": None,
    "power_modes": [], "power_mode_id": None,
    "error": "non ancora avviato",
}

FAN_NAME = "tegra_pwmfan"


class JtopClient(object):
    def __init__(self, poll_interval=1.0):
        self._poll_interval = poll_interval
        self._lock = threading.Lock()
        self._snapshot = dict(_EMPTY)
        self._thread = None
        self._stop = False
        self._command_lock = threading.Lock()
        self._pending_command = None  # un solo slot: l'ultimo comando vince

    def start(self):
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self._stop = True

    def get_snapshot(self):
        with self._lock:
            snap = dict(self._snapshot)
            snap["temps"] = dict(snap["temps"])
            return snap

    def _set(self, **kwargs):
        with self._lock:
            self._snapshot.update(kwargs)

    # ---- comandi: scritti da un thread HTTP, eseguiti SOLO dal thread di
    # _run() che possiede la connessione jtop() -- stesso principio del
    # cambio modello TensorRT in demo_server.py (mai toccare una risorsa
    # legata a un thread da un thread diverso). ----

    def request_command(self, cmd):
        """Thread-safe. cmd: dict con 'type' in
        {'power_mode', 'fan', 'max_performance'}."""
        with self._command_lock:
            self._pending_command = cmd

    def _pop_command(self):
        with self._command_lock:
            cmd, self._pending_command = self._pending_command, None
            return cmd

    def _apply_command(self, jetson, cmd):
        try:
            ctype = cmd.get("type")
            if ctype == "power_mode":
                jetson.nvpmodel.set_nvpmodel_id(cmd["id"], force=False)
            elif ctype == "fan":
                if cmd.get("speed") is not None:
                    jetson.fan.set_profile(FAN_NAME, "manual")
                    jetson.fan.set_speed(FAN_NAME, cmd["speed"], idx=0)
                elif cmd.get("profile"):
                    jetson.fan.set_profile(FAN_NAME, cmd["profile"])
            elif ctype == "max_performance":
                models = list(jetson.nvpmodel.models) if jetson.nvpmodel else []
                target_id = models.index("MAXN") if "MAXN" in models else 0
                jetson.nvpmodel.set_nvpmodel_id(target_id, force=False)
                jetson.fan.set_profile(FAN_NAME, "manual")
                jetson.fan.set_speed(FAN_NAME, 100, idx=0)
        except Exception as e:  # un comando fallito non deve far cadere jtop
            self._set(error="comando jtop fallito: {}".format(e))

    def _run(self):
        try:
            from jtop import jtop
        except ImportError as e:
            self._set(jtop_ok=False, error="jtop non importabile: {}".format(e))
            return

        while not self._stop:
            try:
                with jtop() as jetson:
                    while jetson.ok() and not self._stop:
                        self._update_from(jetson)
                        cmd = self._pop_command()
                        if cmd:
                            self._apply_command(jetson, cmd)
            except Exception as e:  # servizio jtop assente/riavviato: ritenta
                self._set(jtop_ok=False, error=str(e))
                time.sleep(self._poll_interval)

    def _update_from(self, jetson):
        stats = jetson.stats or {}
        mem = (jetson.memory or {}).get("RAM", {})
        fan = (jetson.fan or {}).get("tegra_pwmfan", {})

        cpu_vals = [v for k, v in stats.items()
                   if k.startswith("CPU") and isinstance(v, (int, float))]
        cpu_percent = sum(cpu_vals) / len(cpu_vals) if cpu_vals else None
        temps = {k[len("Temp "):]: v for k, v in stats.items()
                if k.startswith("Temp ") and isinstance(v, (int, float))}
        fan_rpm = fan.get("rpm") or []
        fan_speed = fan.get("speed") or []
        nvpmodel = jetson.nvpmodel

        self._set(
            jtop_ok=True,
            cpu_percent=round(cpu_percent, 1) if cpu_percent is not None else None,
            gpu_percent=stats.get("GPU"),
            ram_used_mb=round(mem["used"] / 1024.0, 1) if "used" in mem else None,
            ram_total_mb=round(mem["tot"] / 1024.0, 1) if "tot" in mem else None,
            temps=temps,
            fan_rpm=fan_rpm[0] if fan_rpm else None,
            fan_percent=round(fan_speed[0], 1) if fan_speed else None,
            fan_profile=fan.get("profile"),
            power_mode=stats.get("nvp model"),
            power_mw=stats.get("Power TOT"),
            power_modes=list(nvpmodel.models) if nvpmodel else [],
            power_mode_id=nvpmodel.id if nvpmodel else None,
            error=None,
        )
