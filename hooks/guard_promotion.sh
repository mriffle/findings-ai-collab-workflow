#!/usr/bin/env bash
# Findings Workflow hook — promotion gate for analysis scripts (spec doc 05).
# A script under scripts/promoted/ must pass static checks before it is accepted, because a
# finding may link only to a promoted script. This hook enforces the deterministic, fast,
# side-effect-free part of the bar:
#   - ruff  (lint, strict rule set) — blocks on failure when available
#   - mypy  (strict type checking)  — blocks on failure when available AND configured
# TESTS (unit/property/planted-truth/edge cases) are also required by convention but are
# verified by the code-reviewer agent, not run here (running arbitrary tests synchronously
# in a hook is slow and unsafe). See conventions/enforcement-map.md.
#
# STRICTNESS is driven by the PROJECT config (pyproject [tool.mypy] strict=true and
# [tool.ruff.lint].select), which setup-env seeds. mypy is run from the project dir so it
# loads that config; ruff is pointed at it explicitly (--config) so the strict rule set
# applies even when checking the PreToolUse temp copy (ruff resolves config from the file's
# own directory, not cwd — without --config a temp copy would fall back to ruff's defaults).
#
# PreToolUse(Write): validate the incoming content on a temp file; exit 2 blocks the write.
# PostToolUse(Write|Edit): re-validate the on-disk file; warn only (the write already happened).
#
# Scope + fail-open behavior match the other guards.

input=$(cat)
command -v jq >/dev/null 2>&1 || exit 0

cwd=$(printf '%s' "$input" | jq -r '.cwd // empty' 2>/dev/null)
[ -n "$cwd" ] || cwd="$PWD"
[ -f "$cwd/state/workflow.json" ] || exit 0

event=$(printf '%s' "$input" | jq -r '.hook_event_name // empty' 2>/dev/null)
fp=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null)
[ -n "$fp" ] || exit 0
case "$fp" in
  *scripts/promoted/*.py) : ;;
  *) exit 0 ;;
esac

# Resolve a relative file_path against the JSON cwd (the rest of the hook trusts cwd, not the
# hook process's cwd), so the PostToolUse on-disk re-check can't silently no-op on a relative path.
case "$fp" in
  /*) fpath="$fp" ;;
  *)  fpath="$cwd/$fp" ;;
esac

# Resolve a tool from the project-local venv first, then PATH. The zero-global-footprint
# setup (setup-env) installs ruff/mypy into ./.venv rather than on PATH, so prefer that;
# fall back to PATH. Echoes a runnable path, or nothing if unavailable (caller skips — fail open).
resolve_tool() {
  local name="$1"
  if [ -x "$cwd/.venv/bin/$name" ]; then printf '%s' "$cwd/.venv/bin/$name"; return 0; fi
  if [ -x "$cwd/.venv/Scripts/$name.exe" ]; then printf '%s' "$cwd/.venv/Scripts/$name.exe"; return 0; fi
  if [ -x "$cwd/.venv/Scripts/$name" ]; then printf '%s' "$cwd/.venv/Scripts/$name"; return 0; fi
  command -v "$name" 2>/dev/null
}

CHECK_MSGS=""
run_checks() { # $1 = path to a python file; returns 0 if all available checks pass
  local f="$1" failed=0 out ruff_bin mypy_bin
  CHECK_MSGS=""
  ruff_bin=$(resolve_tool ruff)
  if [ -n "$ruff_bin" ]; then
    # Point ruff at the project config explicitly so the strict rule set applies even to the
    # PreToolUse temp copy. Precedence follows ruff's own: .ruff.toml > ruff.toml > pyproject.toml.
    local ruff_cfg="" rc
    [ -f "$cwd/pyproject.toml" ] && ruff_cfg="$cwd/pyproject.toml"
    [ -f "$cwd/ruff.toml" ] && ruff_cfg="$cwd/ruff.toml"
    [ -f "$cwd/.ruff.toml" ] && ruff_cfg="$cwd/.ruff.toml"
    if [ -n "$ruff_cfg" ]; then
      out=$("$ruff_bin" check --config "$ruff_cfg" "$f" 2>&1); rc=$?
    else
      out=$("$ruff_bin" check "$f" 2>&1); rc=$?
    fi
    if [ "$rc" -ne 0 ]; then failed=1; CHECK_MSGS="$CHECK_MSGS"$'\n'"[ruff]"$'\n'"$out"; fi
  else
    CHECK_MSGS="$CHECK_MSGS"$'\n'"[ruff] not installed — lint check skipped (install ruff, e.g. via setup-env, to enforce)"
  fi
  mypy_bin=$(resolve_tool mypy)
  if [ -n "$mypy_bin" ] && { [ -f "$cwd/pyproject.toml" ] || [ -f "$cwd/mypy.ini" ] || [ -f "$cwd/setup.cfg" ]; }; then
    if ! out=$(cd "$cwd" && "$mypy_bin" "$f" 2>&1); then failed=1; CHECK_MSGS="$CHECK_MSGS"$'\n'"[mypy]"$'\n'"$out"; fi
  else
    CHECK_MSGS="$CHECK_MSGS"$'\n'"[mypy] not installed or no type-check config — type check skipped"
  fi
  return $failed
}

if [ "$event" = "PreToolUse" ]; then
  # Only Write carries full content; Edit is handled at PostToolUse.
  tmp=$(mktemp --suffix=.py 2>/dev/null || mktemp 2>/dev/null)
  [ -n "$tmp" ] || exit 0
  # Materialize the incoming content with `jq -j` (raw, no added/stripped newline) so the temp
  # copy is byte-faithful to what will land on disk. A $(...) capture strips trailing newlines,
  # which under the strict ruleset trips a spurious W292 ("no newline at end of file") on every
  # well-formed script. Absent/empty content (e.g. an Edit event) yields an empty file -> skip.
  printf '%s' "$input" | jq -j '.tool_input.content // empty' > "$tmp" 2>/dev/null
  [ -s "$tmp" ] || { rm -f "$tmp"; exit 0; }
  if run_checks "$tmp"; then
    rm -f "$tmp"; exit 0
  else
    rm -f "$tmp"
    printf 'Blocked: %s does not pass the promotion checks; a finding may not link to it until it does (doc 05). Fix the issues below or keep the script in scripts/scratch/.\nTests are still required and are verified by the code-reviewer.%s\n' "$(basename "$fp")" "$CHECK_MSGS" >&2
    exit 2
  fi
else
  # PostToolUse — file is on disk; cannot block, so warn.
  if [ -f "$fpath" ] && ! run_checks "$fpath"; then
    printf 'Warning: promoted script %s currently fails the promotion checks; it is not validly promoted and findings must not link to it until fixed (doc 05).%s\n' "$(basename "$fp")" "$CHECK_MSGS" >&2
  fi
  exit 0
fi
