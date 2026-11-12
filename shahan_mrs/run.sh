#!/bin/sh
# Install the pinned packages, repair SciPy and the symptom model if needed, then open the app.
set -eu
cd "$(dirname "$0")"
bash setup.sh
exec .venv/bin/streamlit run app.py
