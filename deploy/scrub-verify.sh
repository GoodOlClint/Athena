#!/usr/bin/env bash
# scrub-verify.sh — publication scrub verifier (ADR 040 / docs/publication-plan.md P0.3).
#
# Fails (exit 1) if any scrub term appears in a git repo's working tree, commit
# messages, or full history. Term-free by construction: the term list lives ONLY
# in untracked scratch and is passed in, so this script is safe to track publicly.
#
# Usage:
#   scrub-verify.sh --terms <term-file> [--repo <path>]
#   SCRUB_TERMS=<term-file> scrub-verify.sh [--repo <path>]
#
# Term file: one rg regex per line; blank lines and #-comments ignored. Matched
# case-insensitively. Use \b for short/ambiguous codenames.
#
# History scan uses `git log --all -p` (complete on a linear history — this repo
# has zero merge commits; the script asserts that and warns if merges appear).
# No `pipefail`: the display pipelines truncate with `head`, which would SIGPIPE
# upstream stages; result capture already guards rg's no-match exit with `|| true`.
set -eu

terms="${SCRUB_TERMS:-}"
repo="."
while [[ $# -gt 0 ]]; do
  case "$1" in
    --terms) terms="$2"; shift 2 ;;
    --repo)  repo="$2";  shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$terms" && -f "$terms" ]] || { echo "error: --terms <file> required (or SCRUB_TERMS env)" >&2; exit 2; }
command -v rg >/dev/null || { echo "error: ripgrep (rg) not found" >&2; exit 2; }
cd "$repo"
git rev-parse --git-dir >/dev/null 2>&1 || { echo "error: $repo is not a git repo" >&2; exit 2; }

# Strip comments/blanks to a clean rg pattern file.
pat="$(mktemp)"; trap 'rm -f "$pat"' EXIT
grep -vE '^\s*(#|$)' "$terms" > "$pat"
[[ -s "$pat" ]] || { echo "error: no patterns after stripping comments" >&2; exit 2; }

if [[ "$(git log --all --merges --oneline | wc -l | tr -d ' ')" != "0" ]]; then
  echo "WARNING: repo has merge commits — 'git log -p' history scan may be incomplete." >&2
fi

fail=0
hr() { printf '%s\n' "----------------------------------------------------------------"; }

echo "scrub-verify: terms=$terms repo=$(pwd)"
hr

# (a) Working tree — tracked files only.
echo "[1/3] working tree (tracked files)"
wt="$(git ls-files -z | xargs -0 rg -i -n -f "$pat" 2>/dev/null || true)"
if [[ -n "$wt" ]]; then
  fail=1
  echo "  HIT — $(printf '%s\n' "$wt" | wc -l | tr -d ' ') line(s) in $(printf '%s\n' "$wt" | cut -d: -f1 | sort -u | wc -l | tr -d ' ') file(s):"
  printf '%s\n' "$wt" | cut -d: -f1 | sort | uniq -c | sort -rn | head -40 | sed 's/^/    /'
else
  echo "  clean"
fi
hr

# (b) Commit messages.
echo "[2/3] commit messages (all refs)"
msg="$(git log --all --format='%H %s%n%b' | rg -i -f "$pat" 2>/dev/null || true)"
if [[ -n "$msg" ]]; then
  fail=1
  echo "  HIT — $(printf '%s\n' "$msg" | wc -l | tr -d ' ') matching line(s). Sample:"
  printf '%s\n' "$msg" | head -8 | sed 's/^/    /'
else
  echo "  clean"
fi
hr

# (c) Full history content (added/removed lines across all revisions & tags).
echo "[3/3] history content (git log --all -p)"
hist="$(git log --all -p -U0 --format='%n== commit %H ==' | rg -i -f "$pat" 2>/dev/null || true)"
if [[ -n "$hist" ]]; then
  fail=1
  echo "  HIT — $(printf '%s\n' "$hist" | wc -l | tr -d ' ') matching diff line(s). Sample:"
  printf '%s\n' "$hist" | head -8 | sed 's/^/    /'
else
  echo "  clean"
fi
hr

if [[ "$fail" -ne 0 ]]; then
  echo "RESULT: CONTAMINATED — scrub incomplete."
  exit 1
fi
echo "RESULT: CLEAN — no scrub term found in tree, messages, or history."
