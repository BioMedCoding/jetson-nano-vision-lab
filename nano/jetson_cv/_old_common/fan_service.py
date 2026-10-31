# -*- coding: utf-8 -*-
"""Controllo del servizio di sistema automagic-fan.

E' un servizio esterno indipendente da jtop (script di terzi,
/usr/local/bin/automagic-fan/fanctl.py, gira come root e scrive
direttamente su /sys/devices/pwm-fan/target_pwm in base alla temperatura,
rileggendo il config SOLO all'avvio). Quando e' attivo sovrascrive qualunque
valore manuale impostato via jtop_client nel giro di 1-2s (verificato):
controllarlo da qui e' l'unico modo per rendere utile lo slider manuale
della dashboard.

Richiede un setup privilegiato una tantum fatto MANUALMENTE dall'utente
(mai da questo codice): config.json reso scrivibile dal gruppo dell'utente
che esegue la dashboard, e una regola sudoers NOPASSWD ristretta a
'systemctl start/stop/restart automagic-fan.service' (nessun altro
comando). Se quel setup manca, le funzioni qui sotto fanno fallire in modo
pulito (eccezione con messaggio leggibile) senza mai lanciare 'sudo' con
una password interattiva: la dashboard resta usabile per il resto.
"""
import json
import subprocess

CONFIG_PATH = "/etc/automagic-fan/config.json"
SERVICE_NAME = "automagic-fan.service"
SYSTEMCTL = "/bin/systemctl"

# (minimo, massimo) accettati per ogni chiave del config.json di fanctl.py
_CONFIG_LIMITS = {
    "FAN_OFF_TEMP": (10, 80),
    "FAN_MAX_TEMP": (20, 95),
    "UPDATE_INTERVAL": (1, 30),
    "MAX_PERF": (0, 1),
}


def is_active():
    r = subprocess.run(["systemctl", "is-active", SERVICE_NAME],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return r.stdout.decode().strip() == "active"


def read_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def write_config(patch):
    """Valida e fonde patch nel config esistente (chiavi non toccate
    restano quelle su disco, es. MIN_SPD_DELTA/TEMP_ZONE non esposte in UI).
    Solleva ValueError con messaggio leggibile se qualcosa non torna."""
    config = read_config()
    for key, value in patch.items():
        if key not in _CONFIG_LIMITS:
            raise ValueError("parametro sconosciuto: {}".format(key))
        lo, hi = _CONFIG_LIMITS[key]
        value = int(value)
        if not (lo <= value <= hi):
            raise ValueError("{} deve essere in [{},{}]".format(key, lo, hi))
        config[key] = value
    if config["FAN_OFF_TEMP"] >= config["FAN_MAX_TEMP"]:
        raise ValueError("la temperatura di spegnimento deve essere minore "
                         "di quella di velocita' massima")
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    return config


def control(action):
    if action not in ("start", "stop", "restart"):
        raise ValueError("azione sconosciuta: {}".format(action))
    r = subprocess.run(["sudo", "-n", SYSTEMCTL, action, SERVICE_NAME],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(
            "comando '{}' fallito (manca il setup sudoers?): {}".format(
                action, (r.stderr.decode().strip()
                        or r.stdout.decode().strip())))


def get_status():
    """Non solleva mai: se qualcosa non e' disponibile, ritorna
    available=False con un messaggio, cosi' la dashboard mostra un avviso
    invece di rompersi."""
    try:
        active = is_active()
    except Exception as e:
        return {"available": False, "active": None, "config": None,
                "error": str(e)}
    try:
        config = read_config()
    except Exception as e:
        return {"available": False, "active": active, "config": None,
                "error": str(e)}
    return {"available": True, "active": active, "config": config,
            "error": None}
