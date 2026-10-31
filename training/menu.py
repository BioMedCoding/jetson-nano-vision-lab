#!/usr/bin/env python
"""Text menu for the training side. It wraps the existing scripts
(setup/setup_host.sh, quickstart.py, nnictl, nni/promote.py,
host/deploy_model.sh) without duplicating their logic.

Usage: ./menu.sh (activates the venv), or `python menu.py` inside it.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
NNI_DIR = TRAINING_DIR / "nni"
REPO_DIR = TRAINING_DIR.parent
sys.path.insert(0, str(NNI_DIR))
sys.path.insert(0, str(TRAINING_DIR))


def run(cmd, cwd=None) -> int:
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n")
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd or TRAINING_DIR)).returncode


def pause() -> None:
    input("\n[press Enter to go back to the menu] ")


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    return input(f"{prompt}{suffix}: ").strip() or default


def ask_port(prompt: str, default: str) -> str:
    """Like ask(), but rejects non-numbers and ports below 1024 (they need
    root and make nnictl fail with an unclear EACCES)."""
    while True:
        val = ask(prompt, default)
        if val.isdigit() and 1024 <= int(val) <= 65535:
            return val
        print(f"Invalid port '{val}': use a number between 1024 and 65535.")


def confirm(prompt: str, default_no: bool = True) -> bool:
    val = input(f"{prompt} {'[y/N]' if default_no else '[Y/n]'} ").strip().lower()
    return (not default_no) if not val else val == "y"


def list_promising(k):
    from promote import list_promising as _lp
    return _lp(k)


def menu_setup() -> None:
    if (TRAINING_DIR / ".venv").is_dir() and not confirm(
            "The environment exists. Run setup/setup_host.sh again (idempotent)?"):
        return
    run(["bash", "setup/setup_host.sh"])
    pause()


def menu_quickstart() -> None:
    data = ask("Path to your dataset's data.yaml")
    if not data:
        return
    arch = ask("Architecture (yolov5n / yolov5s / yolov7-tiny / yolov8n)", "yolov5n")
    epochs = ask("Max epochs", "100")
    img = ask("Image size (= inference size on the Nano)", "416")
    name = ask("Model name (empty = <dataset>-<arch>)")
    cmd = [sys.executable, "quickstart.py", "--data", data, "--arch", arch,
           "--epochs", epochs, "--img", img]
    if name:
        cmd += ["--name", name]
    if confirm("Deploy to the Nano when done?"):
        cmd.append("--deploy")
    run(cmd)
    pause()


def menu_edit_classes() -> None:
    editor = os.environ.get("EDITOR", "nano")
    run([editor, NNI_DIR / "class_maps" / "target_classes.yaml"])


def menu_smoke_test() -> None:
    print("=== [1/2] Dataset builder on its own (few images, no GPU) ===")
    if run([sys.executable, "smoke_test/make_tiny_sample.py"], cwd=NNI_DIR) != 0:
        print("\nThe dataset builder failed: fix the error above first.")
        pause()
        return
    if not confirm("\nBuilder OK. Also run the NNI smoke experiment (2 short trials)?",
                   default_no=False):
        return
    port = ask_port("NNI port", "8081")
    run(["nnictl", "create", "--config", "nni_config_smoke.yml", "--port", port], cwd=NNI_DIR)
    print(f"\nRunning at http://localhost:{port}. Use 'Status' until both trials are "
          "SUCCEEDED, then 'Stop an NNI experiment'.")
    pause()


def menu_status() -> None:
    print("=== Known NNI experiments ===")
    run(["nnictl", "experiment", "list"])
    print("\n=== Finished trials in runs/nni/ (best mAP50-95 first) ===")
    ranked = list_promising(10)
    for trial_id, score, arch in ranked:
        print(f"  {trial_id}  map50_95={score:.4f}  arch={arch}")
    if not ranked:
        print("(none)")
    pause()


def menu_start_stage_a() -> None:
    print(f"Reminder: check {NNI_DIR / 'class_maps' / 'target_classes.yaml'} first.")
    port = ask_port("NNI port", "8080")
    print("\nStage A: broad search (architecture x dataset x hyperparameters). It "
          "downloads tens of thousands of images; the 48 h cap is a safety net, "
          "not a target (see nni_config_explore.yml).")
    if not confirm(f"Start Stage A on port {port}?"):
        return
    if run(["nnictl", "create", "--config", "nni_config_explore.yml", "--port", port],
           cwd=NNI_DIR) == 0:
        print(f"\nRunning. Web UI: http://localhost:{port}. You can close this menu: "
              "the experiment keeps running in the background.")
    pause()


def menu_stop() -> None:
    run(["nnictl", "experiment", "list"])
    exp_id = ask("\nExperiment id to stop (empty = all)")
    if exp_id:
        run(["nnictl", "stop", exp_id])
    elif confirm("Stop ALL running NNI experiments?"):
        run(["nnictl", "stop", "--all"])
    pause()


def menu_promote() -> None:
    ranked = list_promising(10)
    if not ranked:
        print("No trials in runs/nni/ yet. Run Stage A first.")
        pause()
        return
    for trial_id, score, arch in ranked:
        print(f"  {trial_id}  map50_95={score:.4f}  arch={arch}")
    trial_id = ask("\nTrial id to promote (empty = cancel)")
    if not trial_id:
        return
    cmd = [sys.executable, "promote.py", "--trial-id", trial_id,
           "--epochs", ask("Epochs", "250"), "--patience", ask("Patience", "40"),
           "--max-hours", ask("Hour cap (safety net, not a target)", "48")]
    if confirm("Deploy to the Nano when done?"):
        cmd.append("--deploy")
    run(cmd, cwd=NNI_DIR)
    pause()


def menu_deploy() -> None:
    export_dir = TRAINING_DIR / "export"
    found = sorted(p.name for p in export_dir.iterdir()) if export_dir.is_dir() else []
    print("Model folders in export/: " + (", ".join(found) or "(none)"))
    folder = ask("Model folder to deploy (name in export/ or a path)")
    if not folder:
        return
    path = export_dir / folder if (export_dir / folder).is_dir() else Path(folder)
    run([REPO_DIR / "host" / "deploy_model.sh", path])
    pause()


def menu_system_status() -> None:
    run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
         "--format=csv"])
    run(["df", "-h", str(TRAINING_DIR)])
    pause()


MENU = [
    ("Set up the environment (create/update .venv)", menu_setup),
    ("Quickstart: fine-tune one model on your dataset", menu_quickstart),
    ("NNI: edit the target class list", menu_edit_classes),
    ("NNI: smoke test (dataset builder + 2 short trials)", menu_smoke_test),
    ("NNI: status (experiments + finished trials)", menu_status),
    ("NNI: start Stage A (exploration)", menu_start_stage_a),
    ("NNI: stop an experiment", menu_stop),
    ("NNI: promote a trial (Stage B: long training + export)", menu_promote),
    ("Deploy a model folder to the Nano", menu_deploy),
    ("GPU / disk status", menu_system_status),
]


def main() -> None:
    while True:
        print("\n=== Jetson Nano vision: training menu ===")
        for i, (label, _) in enumerate(MENU, 1):
            print(f"  {i:>2}) {label}")
        print("   0) Quit")
        choice = input("\nChoice: ").strip()
        if choice == "0":
            break
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(MENU):
                MENU[idx][1]()
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid choice.")
        except KeyboardInterrupt:
            print("\nInterrupted.")
        except Exception as e:  # keep the menu alive whatever a step does
            print(f"\nError: {e}")
            pause()


if __name__ == "__main__":
    main()
