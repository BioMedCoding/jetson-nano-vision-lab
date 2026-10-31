#!/bin/bash
# Single entry point for the training side: activates the venv (if it exists)
# and starts the text menu. Without a venv, the menu's first entry runs the setup.
set -e
cd "$(dirname "$0")"
if [ -f .venv/bin/activate ]; then
    . .venv/bin/activate
else
    echo "Note: no .venv yet -- using the system python just for the menu;"
    echo "choose 'Set up the environment' first."
fi
exec python3 menu.py
