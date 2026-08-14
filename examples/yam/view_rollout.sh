#!/usr/bin/env bash
# Open a saved YAM rollout in a network-accessible Rerun web viewer.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: examples/yam/view_rollout.sh ROLLOUT_RRD [RERUN_OPTIONS...]

Serves the recording on 0.0.0.0 and prints a LAN URL that loads it directly.

Environment variables:
  RERUN_BIN        Rerun executable to use
  RERUN_WEB_PORT   Web viewer port (default: 9090)
  RERUN_GRPC_PORT  Recording server port (default: 9876)
  RERUN_PUBLIC_IP  IP/hostname placed in the printed URL (default: LAN IP)
EOF
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  [[ $# -gt 0 ]] && exit 0
  exit 2
fi

rrd_path=$1
shift

if [[ ! -f "$rrd_path" ]]; then
  echo "error: RRD file not found: $rrd_path" >&2
  exit 1
fi

if [[ -n "${RERUN_BIN:-}" ]]; then
  rerun_bin=$RERUN_BIN
elif [[ -x /home/npow/molmoact2-venv/bin/rerun ]]; then
  rerun_bin=/home/npow/molmoact2-venv/bin/rerun
elif command -v rerun >/dev/null 2>&1; then
  rerun_bin=$(command -v rerun)
else
  echo "error: rerun not found; set RERUN_BIN to its executable path" >&2
  exit 1
fi

if [[ ! -x "$rerun_bin" ]]; then
  echo "error: Rerun executable is not executable: $rerun_bin" >&2
  exit 1
fi

web_port=${RERUN_WEB_PORT:-9090}
grpc_port=${RERUN_GRPC_PORT:-9876}
public_ip=${RERUN_PUBLIC_IP:-}
if [[ -z "$public_ip" ]]; then
  public_ip=$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i == "src") {print $(i + 1); exit}}')
fi

echo "[rerun] serving: $rrd_path"
echo "[rerun] listening on: 0.0.0.0:$web_port (viewer), 0.0.0.0:$grpc_port (recording)"
if [[ -n "$public_ip" ]]; then
  viewer_url="http://$public_ip:$web_port?url=rerun%2Bhttp%3A%2F%2F$public_ip%3A$grpc_port%2Fproxy"
  echo "[rerun] open this URL (it loads the recording directly):"
  echo "$viewer_url"
else
  echo "error: could not determine this machine's LAN IP; set RERUN_PUBLIC_IP" >&2
  exit 1
fi

exec "$rerun_bin" \
  --bind 0.0.0.0 \
  --serve-web \
  --web-viewer-port "$web_port" \
  --port "$grpc_port" \
  "$rrd_path" \
  "$@"
