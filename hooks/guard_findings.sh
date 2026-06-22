#!/usr/bin/env bash
# Findings Workflow hook — integrity-gate + promoted-script guard on finding documents.
# Spec: docs 02.3, 03, 05. Enforces two invariants when a finding file (findings/NNNN-*.md)
# is written or edited:
#   1. A finding may not claim `integrity_signoff: true` or `status: validated` before the
#      integrity gate has passed (state/workflow.json .integrity_gate.passed == true).
#   2. A `validated` finding may link only to a promoted script (scripts/promoted/), never
#      scripts/scratch/.
#
# This is a deterministic backstop; the findings-manager is the authoritative enforcer and
# applies the full `validated` bar. Scope + fail-open behavior match the other guards.
#
# Reads the PreToolUse JSON event on stdin; exit 2 + stderr blocks the tool call.

input=$(cat)
command -v jq >/dev/null 2>&1 || exit 0

cwd=$(printf '%s' "$input" | jq -r '.cwd // empty' 2>/dev/null)
[ -n "$cwd" ] || cwd="$PWD"
[ -f "$cwd/state/workflow.json" ] || exit 0

fp=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null)
[ -n "$fp" ] || exit 0

# Only finding documents: a path containing findings/ and a basename like NNNN-*.md.
case "$fp" in
  *findings/*) : ;;
  *) exit 0 ;;
esac
base=$(basename "$fp")
printf '%s' "$base" | grep -Eq '^[0-9]{4}.*\.md$' || exit 0

# Content being written (Write: .content; Edit: .new_string).
content=$(printf '%s' "$input" | jq -r '.tool_input.content // .tool_input.new_string // empty' 2>/dev/null)
[ -n "$content" ] || exit 0

passed=$(jq -r '.integrity_gate.passed // false' "$cwd/state/workflow.json" 2>/dev/null)

claims_signoff=no
# Allow an optional single/double quote before the value, so a YAML-legal quoted
# scalar (integrity_signoff: "true") can't slip past the gate. Matches the quote
# handling on the promoted-link check below.
printf '%s' "$content" | grep -Eq '^[[:space:]]*integrity_signoff:[[:space:]]*"?'"'"'?true\b' && claims_signoff=yes
claims_validated=no
printf '%s' "$content" | grep -Eq '^[[:space:]]*status:[[:space:]]*"?'"'"'?validated\b' && claims_validated=yes

if [ "$passed" != "true" ] && { [ "$claims_signoff" = "yes" ] || [ "$claims_validated" = "yes" ]; }; then
  echo "Blocked: a finding cannot claim integrity_signoff: true or status: validated before the integrity gate passes (state/workflow.json .integrity_gate.passed is not true). Complete stage3-loaders and obtain sign-off first (docs 02.3, 05)." >&2
  exit 2
fi

if [ "$claims_validated" = "yes" ] && printf '%s' "$content" | grep -Eq 'path:[[:space:]]*"?'"'"'?scripts/scratch/'; then
  echo "Blocked: a validated finding may link only to a promoted script (scripts/promoted/), not scripts/scratch/ (docs 03, 05). Promote the script and re-point provenance.script.path before validating." >&2
  exit 2
fi

exit 0
