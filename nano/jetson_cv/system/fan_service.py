"""Control of the automagic-fan system service (optional).

automagic-fan is a third-party fan controller (vendored in
nano/vendor/jetson-fan-ctl, MIT licensed), independent of jtop: it runs as
root and writes /sys/devices/pwm-fan/target_pwm from the temperature,
reading its config ONLY at start. While it runs it overrides any manual
speed set through jtop within 1-2 s (verified), so stopping it from the
dashboard is the only way to make the manual slider meaningful.

Needs a one-time privileged setup, done by the user, never by this code
(nano/setup/install_fan_service.sh): the config file made writable by the
dashboard user's group, and a sudoers rule allowing exactly
'systemctl start/stop/restart automagic-fan.service' without a password.
Without that setup the functions below fail cleanly with a readable
message (sudo is always called with -n, never prompting) and the rest of
the dashboard keeps working.
"""
import json
import subprocess

CONFIG_PATH = "/etc/automagic-fan/config.json"
SERVICE_NAME = "automagic-fan.service"
SYSTEMCTL = "/bin/systemctl"

# (min, max) accepted for each key of fanctl.py's config.json
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
    """Validate `patch` and merge it into the existing config (untouched
    keys keep their on-disk value, e.g. MIN_SPD_DELTA / TEMP_ZONE, which the
    UI does not expose). Raises ValueError with a readable message."""
    config = read_config()
    for key, value in patch.items():
        if key not in _CONFIG_LIMITS:
            raise ValueError("unknown parameter: {}".format(key))
        lo, hi = _CONFIG_LIMITS[key]
        value = int(value)
        if not (lo <= value <= hi):
            raise ValueError("{} must be in [{},{}]".format(key, lo, hi))
        config[key] = value
    if config["FAN_OFF_TEMP"] >= config["FAN_MAX_TEMP"]:
        raise ValueError("the fan-off temperature must be lower than the "
                         "full-speed temperature")
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    return config


def control(action):
    if action not in ("start", "stop", "restart"):
        raise ValueError("unknown action: {}".format(action))
    r = subprocess.run(["sudo", "-n", SYSTEMCTL, action, SERVICE_NAME],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(
            "'{}' failed (is the sudoers rule installed? see "
            "nano/setup/install_fan_service.sh): {}".format(
                action, (r.stderr.decode().strip()
                        or r.stdout.decode().strip())))


def get_status():
    """Never raises: if something is unavailable it returns
    available=False with a message, so the dashboard shows a notice
    instead of breaking."""
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
