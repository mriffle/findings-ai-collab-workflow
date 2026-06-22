#!/usr/bin/env bash
# Findings Workflow hook — raw data is read-only (spec doc 05.1).
# Blocks Write/Edit to anything under the project's data/ directory, and makes a
# best-effort attempt to block obvious Bash writes into data/.
#
# Scope: only engages inside an initialized Findings Workflow project (state/workflow.json
# present), so it never interferes with unrelated repositories. Fails OPEN (allows the
# action) if jq is missing or input is unparseable — the gate is defense-in-depth alongside
# orchestrator behavior, and should never wedge a session over its own tooling.
#
# Reads the PreToolUse JSON event on stdin; exit 2 + stderr message blocks the tool call.

input=$(cat)
command -v jq >/dev/null 2>&1 || { echo "findings-workflow: jq not found; data/ read-only guard skipped" >&2; exit 0; }

cwd=$(printf '%s' "$input" | jq -r '.cwd // empty' 2>/dev/null)
[ -n "$cwd" ] || cwd="$PWD"
# Only engage inside an initialized Findings Workflow project.
[ -f "$cwd/state/workflow.json" ] || exit 0

tool=$(printf '%s' "$input" | jq -r '.tool_name // empty' 2>/dev/null)

is_under_data() {
  # Return 0 if $1 resolves under <cwd>/data/.
  local p="$1"
  case "$p" in
    "$cwd"/data/*|"$cwd"/data) return 0 ;;
    /*) return 1 ;;                                   # other absolute path
    data/*|data|./data/*|./data) return 0 ;;
  esac
  return 1
}

case "$tool" in
  Write|Edit)
    fp=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null)
    [ -n "$fp" ] || exit 0
    if is_under_data "$fp"; then
      echo "Blocked: data/ holds immutable raw inputs and is read-only in the Findings Workflow (doc 05.1). Outputs belong in results/ or figures/, which are regenerated from data + a script." >&2
      exit 2
    fi
    ;;
  Bash)
    cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // empty' 2>/dev/null)
    [ -n "$cmd" ] || exit 0
    # Best-effort: redirects into data/, or rm/mv/cp/tee/truncate/sed -i targeting data/.
    if printf '%s' "$cmd" | grep -Eq '(>>?|tee[[:space:]]+(-a[[:space:]]+)?)[[:space:]]*\.?/?data/|(\b(rm|mv|cp|truncate|dd)\b|sed[[:space:]]+-i)[^|]*[[:space:]]\.?/?data/'; then
      echo "Blocked (best-effort): this command appears to write or modify the read-only data/ directory (doc 05.1). If this is a false positive, perform the write outside data/." >&2
      exit 2
    fi
    ;;
esac
exit 0
