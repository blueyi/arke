#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

branch="$(git rev-parse --abbrev-ref HEAD)"
remote="${1:-origin}"

if ! git remote get-url "$remote" >/dev/null 2>&1; then
  echo "error: git remote '$remote' not found" >&2
  exit 1
fi

current_upstream="$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null || true)"
if [[ -z "$current_upstream" ]]; then
  echo "==> Setting upstream: $remote/$branch"
  git branch --set-upstream-to="$remote/$branch" "$branch" 2>/dev/null || true
fi

git config push.default current
git config push.autoSetupRemote true
git config alias.pub '!git push -u ${1:-origin} HEAD'
git config alias.sync '!git pull --rebase --autostash && git push'

echo "Git defaults configured for repository:"
echo "  push.default = $(git config --get push.default)"
echo "  push.autoSetupRemote = $(git config --get push.autoSetupRemote)"
echo "  alias.pub = $(git config --get alias.pub)"
echo "  alias.sync = $(git config --get alias.sync)"
if git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
  echo "  upstream = $(git rev-parse --abbrev-ref --symbolic-full-name @{u})"
else
  echo "  upstream = <not set>"
fi
