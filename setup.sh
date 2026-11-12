#!/bin/sh
# Create the app environment with Python 3.9–3.12. Never use python3 when it is 3.13 or newer.
set -eu
cd "$(dirname "$0")"

python_ok() {
  "$1" -c 'import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)'
}

selected=""
if [ -x .venv/bin/python ] && python_ok .venv/bin/python; then
  selected=".venv/bin/python"
else
  for candidate in python3.12 python3.11 python3.10 python3.9 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && python_ok "$candidate"; then
      selected="$candidate"
      break
    fi
  done
fi

if [ -z "$selected" ]; then
  echo "This project needs Python 3.9, 3.10, 3.11, or 3.12."
  echo "Python 3.13 and newer has no prebuilt packages for this app, so pip compiles NumPy for hours."
  if command -v python3 >/dev/null 2>&1; then
    echo "The python3 command here is: $(python3 --version 2>&1)"
  fi
  echo "On macOS, install a supported Python with: brew install python@3.12"
  echo "Then run this script again."
  exit 1
fi

if [ "$selected" != ".venv/bin/python" ]; then
  "$selected" -m venv .venv
fi

.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/ensure_runtime.py
