#!/usr/bin/env bash
#
# verify.sh -- end-to-end verification for the fetal-health-ctg project.
#
# Runs 11 checks in order and prints PASS/FAIL for each. Exits 0 only if all
# pass, 1 otherwise.
#
# Preconditions: the backend must be reachable on $BASE_URL (default
# http://localhost:8000), either from `uvicorn main:app` in backend/ or from
# `docker compose up`. If it is down, checks 4-11 fail gracefully with a
# connection error rather than hanging.
#
# Environment overrides:
#   BASE_URL   backend base URL           (default http://localhost:8000)
#   PYTHON     interpreter for check 3    (default: .venv/bin/python if present,
#                                          else python3)
#   SKIP_BUILD set to 1 to skip checks 1-2 (the slow frontend ones)
#
# Note on ports: if a host dev server and a Docker container both publish 8000,
# `localhost` may resolve to either. Pass BASE_URL explicitly to disambiguate,
# e.g. BASE_URL='http://[::1]:8000' for the container on Docker Desktop.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_URL="${BASE_URL:-http://localhost:8000}"
CURL_TIMEOUT="${CURL_TIMEOUT:-120}"

if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="python3"
fi

TOTAL=13
PASSED=0
FAILED=0
SKIPPED=0
declare -a FAILURES=()

# ---- helpers ----------------------------------------------------------------

pass() {  # $1 = index, $2 = label
  printf '[%d/%d] %s ... PASS\n' "$1" "$TOTAL" "$2"
  PASSED=$((PASSED + 1))
}

fail() {  # $1 = index, $2 = label, $3 = detail
  printf '[%d/%d] %s ... FAIL\n' "$1" "$TOTAL" "$2"
  if [[ -n "${3:-}" ]]; then
    printf '%s\n' "$3" | sed 's/^/        /'
  fi
  FAILED=$((FAILED + 1))
  FAILURES+=("[$1/$TOTAL] $2")
}

# Run a command, capturing combined output; pass on exit 0.
check_cmd() {  # $1 = index, $2 = label, $3.. = command
  local idx="$1" label="$2"; shift 2
  local out rc
  out="$("$@" 2>&1)"; rc=$?
  if [[ $rc -eq 0 ]]; then
    pass "$idx" "$label"
  else
    fail "$idx" "$label" "exit code $rc
$(printf '%s\n' "$out" | tail -15)"
  fi
}

# HTTP check: expects 200 AND the body to contain a needle.
# Body and status are captured in one request; -w appends the status code.
check_http() {  # $1 idx, $2 label, $3 needle, $4.. curl args
  local idx="$1" label="$2" needle="$3"; shift 3
  local raw code body
  raw="$(curl -s -S --max-time "$CURL_TIMEOUT" -w $'\n%{http_code}' "$@" 2>&1)"
  code="$(printf '%s' "$raw" | tail -1)"
  body="$(printf '%s' "$raw" | sed '$d')"

  if [[ ! "$code" =~ ^[0-9]{3}$ ]]; then
    fail "$idx" "$label" "connection error: $(printf '%s' "$raw" | tr '\n' ' ' | cut -c1-200)"
    return
  fi
  if [[ "$code" == "000" ]]; then
    fail "$idx" "$label" "connection error (no HTTP response)
$(printf '%s' "$body" | cut -c1-300)"
    return
  fi
  if [[ "$code" != "200" ]]; then
    fail "$idx" "$label" "HTTP $code
body: $(printf '%s' "$body" | cut -c1-300)"
    return
  fi
  # Tolerate any JSON spacing around the colon (e.g. "n":2 vs "n": 2).
  if printf '%s' "$body" | tr -d ' \n' | grep -q "$(printf '%s' "$needle" | tr -d ' ')"; then
    pass "$idx" "$label"
  else
    fail "$idx" "$label" "HTTP 200 but response did not contain: $needle
body: $(printf '%s' "$body" | cut -c1-300)"
  fi
}

# ---- the Pathological demo case (artifacts/example_cases.pkl) ---------------

read -r -d '' ROW <<'JSON'
{
  "baseline value": 151.0,
  "accelerations": 0.0,
  "fetal_movement": 0.002,
  "uterine_contractions": 0.0,
  "light_decelerations": 0.0,
  "severe_decelerations": 0.0,
  "prolongued_decelerations": 0.0,
  "abnormal_short_term_variability": 81.0,
  "mean_value_of_short_term_variability": 0.4,
  "percentage_of_time_with_abnormal_long_term_variability": 29.0,
  "mean_value_of_long_term_variability": 6.4,
  "histogram_width": 30.0,
  "histogram_min": 130.0,
  "histogram_max": 160.0,
  "histogram_number_of_peaks": 2.0,
  "histogram_number_of_zeroes": 0.0,
  "histogram_mode": 152.0,
  "histogram_mean": 151.0,
  "histogram_median": 153.0,
  "histogram_variance": 1.0,
  "histogram_tendency": 1.0
}
JSON

BATCH="{\"samples\": [${ROW}, ${ROW}]}"
# Counterfactual: the Pathological demo case, target Normal (class 0).
CF_BODY="{\"features\": ${ROW}, \"target_class\": 0, \"k\": 3}"
JSON_HDR='Content-Type: application/json'

# ---- run --------------------------------------------------------------------

echo "verify.sh: checking fetal-health-ctg project"

if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
  printf '[1/%d] Frontend typecheck ... SKIP (SKIP_BUILD=1)\n' "$TOTAL"
  printf '[2/%d] Frontend build ... SKIP (SKIP_BUILD=1)\n' "$TOTAL"
  SKIPPED=2
else
  check_cmd 1 "Frontend typecheck" npm --prefix frontend run typecheck
  check_cmd 2 "Frontend build"     npm --prefix frontend run build
fi

check_cmd 3 "Backend import check" \
  env PYTHONPATH="$ROOT/backend:$ROOT/src" "$PY" -c \
  "import main, inference, artifacts; print('OK')"

check_http 4  "GET /api/health"         '"model_loaded":true' "$BASE_URL/api/health"         -H "$JSON_HDR"
check_http 5  "GET /api/features"       '"groups"'            "$BASE_URL/api/features"       -H "$JSON_HDR"
check_http 6  "GET /api/examples"       '"examples"'          "$BASE_URL/api/examples"       -H "$JSON_HDR"
check_http 7  "GET /api/metrics"        '"ranked_table"'      "$BASE_URL/api/metrics"        -H "$JSON_HDR"
check_http 8  "GET /api/explain/global" '"spearman_rho"'      "$BASE_URL/api/explain/global" -H "$JSON_HDR"

check_http 9  "POST /api/predict"       '"predicted_label"' \
  -X POST "$BASE_URL/api/predict"       -H "$JSON_HDR" -d "$ROW"
check_http 10 "POST /api/predict/batch" '"predictions"' \
  -X POST "$BASE_URL/api/predict/batch" -H "$JSON_HDR" -d "$BATCH"
check_http 11 "POST /api/explain"       '"per_class"' \
  -X POST "$BASE_URL/api/explain"       -H "$JSON_HDR" -d "$ROW"

# ---- FC4/FA3 additions ----
check_http 12 "GET /api/calibration"    '"chosen_variant"' \
  "$BASE_URL/api/calibration" -H "$JSON_HDR"
# The counterfactual search runs a multi-restart greedy sweep; allow it room.
CURL_TIMEOUT="${CF_TIMEOUT:-180}" \
check_http 13 "POST /api/counterfactual" '"counterfactuals"' \
  -X POST "$BASE_URL/api/counterfactual" -H "$JSON_HDR" -d "$CF_BODY"

# Check 10 must also report n == 2.
if [[ $FAILED -eq 0 || " ${FAILURES[*]:-} " != *"POST /api/predict/batch"* ]]; then
  N="$(curl -s -S --max-time "$CURL_TIMEOUT" -X POST "$BASE_URL/api/predict/batch" \
        -H "$JSON_HDR" -d "$BATCH" 2>/dev/null | tr -d ' \n' | grep -o '"n":[0-9]*' | head -1)"
  if [[ -n "$N" && "$N" != '"n":2' ]]; then
    echo "        WARNING: /api/predict/batch returned ${N}, expected \"n\":2"
  fi
fi

echo
if [[ $SKIPPED -gt 0 ]]; then
  printf 'Result: %d/%d PASS (%d skipped)\n' "$PASSED" "$((TOTAL - SKIPPED))" "$SKIPPED"
else
  printf 'Result: %d/%d PASS\n' "$PASSED" "$TOTAL"
fi

if [[ $FAILED -gt 0 ]]; then
  echo
  echo "Failed checks:"
  for f in "${FAILURES[@]}"; do echo "  - $f"; done
  exit 1
fi
exit 0
