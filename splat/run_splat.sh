#!/usr/bin/env bash
set -euo pipefail
pack=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
sid=${1:-}
case "$sid" in VN4b1MGnnCR|HFwcH424n9w) shift;; *) echo "Usage: $0 {VN4b1MGnnCR|HFwcH424n9w} [--smoke] [runner options]" >&2; exit 2;; esac
root=${HP_SPLAT_ROOT:-"$HOME/hp_splat"}
work="$root/$sid"
if [[ " ${*:-} " == *' --smoke '* ]]; then work="$root/smoke-$sid"; fi
mkdir -p "$work" "$root/send_back/$(basename "$work")"
if [[ ${HP_SPLAT_LOGGED:-0} != 1 ]]; then
  set +e
  HP_SPLAT_LOGGED=1 bash "$pack/run_splat.sh" "$sid" "$@" 2>&1 | tee -a "$work/run.log"
  result=${PIPESTATUS[0]}
  cp "$work/run.log" "$root/send_back/$(basename "$work")/run.log"
  inventory_python=${HP_SPLAT_PYTHON:-"$HOME/.local/envs/hp-splat/bin/python"}
  [[ -x "$inventory_python" ]] || inventory_python=python3
  "$inventory_python" - "$root/send_back/$(basename "$work")" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
files = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'files.tsv')
(root / 'files.tsv').write_text('path\tbytes\n' + ''.join(f'{p.relative_to(root)}\t{p.stat().st_size}\n' for p in files))
PY
  exit "$result"
fi
echo "$(date -u '+%FT%TZ') run $sid $*"
lock="$work/.runner-lock"
if ! mkdir "$lock" 2>/dev/null; then
  old=$(cat "$lock/pid" 2>/dev/null || true)
  if [[ $old =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "Runner already active, PID $old"; exit 1
  fi
  echo 'Removing stale lock from interrupted run.'
  rm -f "$lock/pid"; rmdir "$lock"; mkdir "$lock"
fi
echo $$ > "$lock/pid"
finish() {
  code=$?
  cp "$work/run.log" "$root/send_back/$(basename "$work")/run.log" || true
  rm -f "$lock/pid"; rmdir "$lock" || true
  exit "$code"
}
trap finish EXIT
if [[ " ${*:-} " == *' --smoke '* ]]; then
  python_bin=${HP_SPLAT_PYTHON:-python3}
else
  bash "$pack/setup_linux.sh"
  source "$HOME/hp_splat/send_back/proof/activate.sh"
  python_bin=python
fi
export PYTHONUNBUFFERED=1
"$python_bin" "$pack/tools/run_splat.py" "$sid" --pack "$pack" --root "$root" "$@"
