#!/bin/bash
# Punto di ingresso unico per la pipeline di training: attiva il venv (se
# esiste) e lancia il menu testuale. Se il venv non esiste ancora, il menu
# stesso offre di eseguire setup_host.sh come prima voce.
set -e
cd "$(dirname "$0")"

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "Nota: .venv non trovato -- uso il python di sistema solo per mostrare"
    echo "il menu; scegli 'Setup ambiente' come prima voce."
fi

python3 nni_pipeline/menu.py
