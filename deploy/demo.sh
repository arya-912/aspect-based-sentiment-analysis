#!/usr/bin/env bash
#
# Start the API locally, expose it through a Cloudflare quick tunnel, and
# print the link that drives the deployed frontend against it.
#
#   ./deploy/demo.sh
#
# Ctrl-C stops both. Nothing here needs an account: a quick tunnel is
# anonymous and its hostname is new every run, which is why the frontend
# takes the API origin from ?api= rather than from its build.

set -euo pipefail

PORT="${PORT:-8000}"
FRONTEND="${FRONTEND:-https://absa-sentiment.vercel.app}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
LOG_DIR="$(mktemp -d)"

for tool in "$PYTHON" "$(command -v cloudflared || true)"; do
  if [ ! -x "$tool" ]; then
    echo "missing: ${tool:-cloudflared}" >&2
    echo "the venv is at .venv; install cloudflared from" >&2
    echo "https://github.com/cloudflare/cloudflared/releases" >&2
    exit 1
  fi
done

pids=()
cleanup() {
  # Disarm first: without this the handler runs for INT/TERM and then a
  # second time for EXIT, printing "stopped." twice.
  trap - EXIT INT TERM
  for pid in ${pids[@]+"${pids[@]}"}; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  echo
  echo "stopped."
}
trap cleanup EXIT INT TERM

echo "starting the API on port $PORT ..."
( cd "$ROOT/backend" && "$PYTHON" -m uvicorn app.main:app \
    --host 127.0.0.1 --port "$PORT" ) > "$LOG_DIR/api.log" 2>&1 &
pids+=($!)

# The model is ~450MB and takes a few seconds to load; the tunnel is useless
# before /health answers, so wait for it rather than racing.
for _ in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:$PORT/api/v1/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "http://127.0.0.1:$PORT/api/v1/health" >/dev/null 2>&1; then
  echo "the API did not come up; its log:" >&2
  tail -20 "$LOG_DIR/api.log" >&2
  exit 1
fi
echo "API is up: $(curl -fsS "http://127.0.0.1:$PORT/api/v1/health")"

echo "opening a tunnel ..."
cloudflared tunnel --url "http://localhost:$PORT" > "$LOG_DIR/tunnel.log" 2>&1 &
pids+=($!)

tunnel=""
for _ in $(seq 1 60); do
  tunnel="$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' \
    "$LOG_DIR/tunnel.log" 2>/dev/null | head -1 || true)"
  [ -n "$tunnel" ] && break
  sleep 1
done

if [ -z "$tunnel" ]; then
  echo "no tunnel URL appeared; cloudflared's log:" >&2
  tail -20 "$LOG_DIR/tunnel.log" >&2
  exit 1
fi

# The hostname is printed before Cloudflare's edge is routing to it, so poll
# rather than test once and cry wolf.
reachable=""
for _ in $(seq 1 20); do
  if curl -fsS "$tunnel/api/v1/health" >/dev/null 2>&1; then
    reachable=yes
    break
  fi
  sleep 1
done
[ -n "$reachable" ] || echo "warning: $tunnel is not answering yet." >&2

cat <<BANNER

  ready. open this, and the deployed UI will talk to this machine:

    $FRONTEND/?api=$tunnel

  the API itself:  $tunnel/api/v1/health
  logs:            $LOG_DIR

  the page remembers the tunnel, so later visits need no ?api=.
  Ctrl-C to stop.

BANNER

wait
