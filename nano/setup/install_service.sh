#!/bin/bash
# Installa/disinstalla il servizio systemd della dashboard CV (autostart al
# boot, restart automatico su crash). Esegui SULLA JETSON, nel tuo terminale
# (usa sudo internamente: serve un TTY reale, non lanciarlo dal tool SSH).
#
# Uso:
#   ./install_demo_service.sh              # installa e avvia
#   ./install_demo_service.sh --force      # sovrascrive se gia' presente e diverso
#   ./install_demo_service.sh --uninstall  # ferma, disabilita, rimuove
set -e

UNIT_NAME="cv-demo.service"
SRC="$(cd "$(dirname "$0")" && pwd)/$UNIT_NAME"
DEST="/etc/systemd/system/$UNIT_NAME"

if [ ! -f /etc/nv_tegra_release ]; then
    echo "ERRORE: questo script va eseguito sulla Jetson, non sull'host."; exit 1
fi

if [ "$1" = "--uninstall" ]; then
    echo "Fermo e disabilito $UNIT_NAME..."
    sudo systemctl stop "$UNIT_NAME" 2>/dev/null || true
    sudo systemctl disable "$UNIT_NAME" 2>/dev/null || true
    sudo rm -f "$DEST"
    sudo systemctl daemon-reload
    echo "Rimosso."
    exit 0
fi

if ! groups "$USER" | grep -qw video; then
    echo "ATTENZIONE: l'utente $USER non e' nel gruppo 'video' (serve per /dev/video0)."
    echo "  Esegui:  sudo usermod -aG video $USER"
    echo "  poi disconnettiti/riconnettiti (serve una nuova sessione) e rilancia questo script."
    exit 1
fi

if [ -f "$DEST" ] && ! cmp -s "$SRC" "$DEST" && [ "$1" != "--force" ]; then
    echo "Un $UNIT_NAME diverso e' gia' installato in $DEST."
    echo "Controlla la differenza con: diff '$SRC' '$DEST'"
    echo "Rilancia con --force per sovrascriverlo."
    exit 1
fi

echo "Installo $UNIT_NAME in $DEST..."
sudo cp "$SRC" "$DEST"
sudo systemctl daemon-reload
sudo systemctl enable --now "$UNIT_NAME"

echo
echo "Fatto. Verifica con:"
echo "  systemctl status $UNIT_NAME"
echo "  curl -s http://localhost:8090/api/stats | python3 -m json.tool"
