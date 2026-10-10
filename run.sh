#!/usr/bin/env bash
# Start the person tracker: full screen, servos KILLED until you press g.
# Any fcb_track.py option can be added, e.g.  ./run.sh --windowed
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
if [ -x "$HERE/.venv/bin/python" ]; then
    PY="$HERE/.venv/bin/python"
else
    PY="${FCB_PYTHON:-python3}"
fi
if ! "$PY" -c 'import ultralytics' 2>/dev/null; then
    echo "ultralytics is not installed for $PY -- set it up once with:"
    echo "  python3 -m venv $HERE/.venv && $HERE/.venv/bin/pip install ultralytics pyserial"
    exit 1
fi
exec "$PY" fcb_track.py "$@"
