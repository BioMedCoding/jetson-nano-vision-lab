#!/usr/bin/env python
"""Menu testuale di controllo per la pipeline di training.

Avvolge gli script gia' esistenti (setup_host.sh, dataset builder, nnictl,
promote.py, deploy.sh) in un'interfaccia a menu -- non duplica la loro
logica, li chiama e basta. Uso: ./menu.sh dalla root del repo, oppure
`python menu.py` da dentro nni_pipeline/ col venv attivo.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent


def run(cmd: list[str], cwd: Path | None = None) -> int:
    print(f"\n$ {' '.join(cmd)}\n")
    proc = subprocess.run(cmd, cwd=str(cwd or PIPELINE_DIR))
    return proc.returncode


def pause() -> None:
    input("\n[premi invio per tornare al menu] ")


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val or default


def ask_port(prompt: str, default: str) -> str:
    """Come ask(), ma rifiuta porte <1024 (richiedono root, causano un
    fallimento poco chiaro EACCES lato nnictl -- vedi nni_pipeline/README.md)
    e valori non numerici, ri-chiedendo finche' non e' valida."""
    while True:
        val = ask(prompt, default)
        if val.isdigit() and 1024 <= int(val) <= 65535:
            return val
        print(f"Porta non valida: '{val}' -- serve un numero tra 1024 e 65535 "
              "(sotto 1024 richiede permessi di root e fallisce con un errore poco chiaro).")


def confirm(prompt: str, default_no: bool = True) -> bool:
    hint = "[s/N]" if default_no else "[S/n]"
    val = input(f"{prompt} {hint} ").strip().lower()
    if not val:
        return not default_no
    return val == "s"


def menu_setup() -> None:
    venv = REPO_ROOT / ".venv"
    if venv.is_dir():
        print(f"Ambiente gia' presente: {venv}")
        if not confirm("Rilanciare comunque setup_host.sh (idempotente)?"):
            return
    else:
        print("Nessun venv trovato, eseguo setup_host.sh...")
    run(["bash", "./setup_host.sh"], cwd=REPO_ROOT)
    pause()


def menu_edit_classes() -> None:
    editor = os.environ.get("EDITOR", "nano")
    target = PIPELINE_DIR / "class_maps" / "target_classes.yaml"
    print(f"Apro {target} con {editor} (imposta $EDITOR per usarne un altro)")
    run([editor, str(target)])


def menu_smoke_test() -> None:
    print("=== [1/2] Dataset builder standalone (poche immagini, no GPU) ===")
    rc = run([sys.executable, "smoke_test/make_tiny_sample.py"])
    if rc != 0:
        print("\nIl dataset builder ha fallito, controlla l'errore sopra prima di procedere.")
        pause()
        return
    if not confirm("\nBuilder OK. Lanciare anche l'esperimento NNI di smoke test (2 trial, pochi minuti)?", default_no=False):
        pause()
        return
    port = ask_port("Porta NNI", "8081")
    run(["nnictl", "create", "--config", "nni_config_smoke.yml", "--port", port])
    print(f"\nAvviato su http://localhost:{port} -- segui da li' o torna al menu e usa"
          " 'Stato' per controllare quando i 2 trial sono SUCCEEDED, poi 'Ferma esperimento NNI'.")
    pause()


def menu_status() -> None:
    print("=== Esperimenti NNI noti (in corso o passati) ===")
    run(["nnictl", "experiment", "list"])

    print("\n=== Trial completati in runs/ (migliori per mAP@0.5:0.95) ===")
    sys.path.insert(0, str(PIPELINE_DIR))
    from promote import list_promising

    ranked = list_promising(10)
    if not ranked:
        print("(nessuno)")
    else:
        for trial_id, score, arch in ranked:
            print(f"  {trial_id}  map50_95={score:.4f}  arch={arch}")
    pause()


def menu_start_stage_a() -> None:
    classes_file = PIPELINE_DIR / "class_maps" / "target_classes.yaml"
    print(f"Promemoria: ricontrolla {classes_file} prima di partire (voce 2 del menu).")
    port = ask_port("Porta NNI", "8080")
    print(f"\nStage A: ricerca ampia (architettura x dataset x iperparametri), rete di"
          f" sicurezza 48h (non un obiettivo -- vedi nni_config_explore.yml).")
    if not confirm(f"Avviare Stage A su porta {port}?", default_no=True):
        return
    rc = run(["nnictl", "create", "--config", "nni_config_explore.yml", "--port", port])
    if rc == 0:
        print(f"\nAvviato. Web UI: http://localhost:{port}")
        print("Puoi chiudere questo menu: l'esperimento continua in background.")
        print("Torna qui quando vuoi con 'Stato' o 'Ferma esperimento NNI'.")
    pause()


def menu_stop() -> None:
    run(["nnictl", "experiment", "list"])
    exp_id = ask("\nID esperimento da fermare (vuoto = ferma tutti)")
    if exp_id:
        run(["nnictl", "stop", exp_id])
    else:
        if confirm("Fermare TUTTI gli esperimenti NNI in corso?"):
            run(["nnictl", "stop", "--all"])
    pause()


def menu_promote() -> None:
    sys.path.insert(0, str(PIPELINE_DIR))
    from promote import list_promising

    ranked = list_promising(10)
    if not ranked:
        print("Nessun trial trovato in runs/. Lancia prima Stage A (voce 5).")
        pause()
        return
    print("Top trial trovati:")
    for trial_id, score, arch in ranked:
        print(f"  {trial_id}  map50_95={score:.4f}  arch={arch}")

    trial_id = ask("\ntrial_id da promuovere (vuoto = annulla)")
    if not trial_id:
        return
    epochs = ask("Epoche", "250")
    patience = ask("Patience", "40")
    max_hours = ask("Tetto ore (rete di sicurezza, non un obiettivo)", "48")
    deploy_host = ask("Deploy su host dopo l'export (vuoto = solo export, es. matteo@10.42.0.24)")

    cmd = [
        sys.executable, "promote.py",
        "--trial-id", trial_id,
        "--epochs", epochs,
        "--patience", patience,
        "--max-hours", max_hours,
    ]
    if deploy_host:
        cmd += ["--deploy", deploy_host]
    run(cmd)
    pause()


def menu_deploy_manual() -> None:
    print("Deploy manuale di un onnx gia' esportato (es. da una run promossa in precedenza).")
    onnx = ask("Path del file .onnx")
    if not onnx:
        return
    classes = ask("Path di classes.txt")
    img_size = ask("img_size", "416")
    postproc = ask("postprocess (v5/v7/v8)")
    host = ask("user@host (vuoto = stampa solo i comandi scp)")

    cmd = ["./deploy.sh", onnx, classes, img_size, postproc]
    if host:
        cmd.append(host)
    run(cmd)
    pause()


def menu_system_status() -> None:
    run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv"])
    print()
    run(["df", "-h", str(REPO_ROOT)])
    pause()


MENU = [
    ("Setup ambiente (crea/aggiorna .venv)", menu_setup),
    ("Modifica lista classi target", menu_edit_classes),
    ("Test rapido (dataset builder + smoke NNI)", menu_smoke_test),
    ("Stato: esperimenti NNI + trial completati", menu_status),
    ("Avvia Stage A (esplorazione)", menu_start_stage_a),
    ("Ferma esperimento NNI", menu_stop),
    ("Promuovi un trial (Stage B: training esteso + export)", menu_promote),
    ("Deploy manuale (onnx gia' pronto)", menu_deploy_manual),
    ("Stato GPU / disco", menu_system_status),
]


def main() -> None:
    while True:
        print("\n=== Pipeline training CV -- menu di controllo ===")
        for i, (label, _) in enumerate(MENU, 1):
            print(f"  {i}) {label}")
        print("  0) Esci")
        choice = input("\nScelta: ").strip()
        if choice == "0":
            break
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(MENU):
                MENU[idx][1]()
            else:
                print("Scelta non valida.")
        except ValueError:
            print("Scelta non valida.")
        except KeyboardInterrupt:
            print("\nInterrotto.")
        except Exception as e:
            print(f"\nErrore: {e}")
            pause()


if __name__ == "__main__":
    main()
