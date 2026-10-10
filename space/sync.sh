#!/bin/bash
# Copies what the Hugging Face Space runs from this repo into a clone of the
# Space (a duplicate of 24yearsold/see-through-demo), leaving its README, git
# history and common/assets (kept in LFS there) alone. Review with
# `git -C <space> diff --stat`, then commit and push from there.
#
#   space/sync.sh ~/hf/see-through-demo
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
space="${1:?usage: space/sync.sh <space clone>}"
[ -d "$space/.git" ] || { echo "$space is not a git clone" >&2; exit 1; }

rsync -a --delete --exclude '__pycache__' --exclude '/assets/' "$repo/common/" "$space/common/"
mkdir -p "$space/annotators"
cp "$repo/annotators/__init__.py" "$space/annotators/__init__.py"
rsync -a --delete --exclude '__pycache__' "$repo/annotators/lama_inpainter/" "$space/annotators/lama_inpainter/"
cp "$repo/inference/scripts/refine_hidden.py" "$space/refine_hidden.py"
cp "$repo/inference/scripts/turn_keyforms.py" "$space/turn_keyforms.py"
cp "$repo/inference/scripts/turn_judge.py" "$space/turn_judge.py"
cp "$repo/inference/scripts/hair_locks.py" "$space/hair_locks.py"
cp "$repo/inference/scripts/figure_head.py" "$space/figure_head.py"
cp "$repo/inference/scripts/figure_tiles.py" "$space/figure_tiles.py"
cp "$repo/inference/scripts/upscale.py" "$space/upscale.py"
cp "$repo/inference/scripts/body_turn.py" "$space/body_turn.py"
cp "$repo/space/app.py" "$space/app.py"
cp "$repo/space/requirements.txt" "$space/requirements.txt"
git -C "$space" status --short
