#!/usr/bin/env bash
# Symlink pre-compiled artifacts (.so kernels, generated Python files) from an
# existing vLLM wheel install into this source tree, so we can import from
# source without recompiling C++/CUDA.
#
# Usage: ./link_precompiled.sh [/path/to/site-packages/vllm]
set -euo pipefail

SITE="${1:-}"
if [ -z "$SITE" ]; then
  SITE=$(python -c 'import importlib.util, sys; \
spec = importlib.util.find_spec("vllm"); \
p = spec.submodule_search_locations[0] if spec else None; \
print(p) if p and "site-packages" in p else sys.exit(1)' 2>/dev/null) || {
    echo "error: couldn't locate an installed vllm in site-packages."
    echo "Pass the path explicitly: $0 /path/to/site-packages/vllm"
    exit 1
  }
fi

FORK="$(cd "$(dirname "$0")" && pwd)/vllm"

echo "Linking precompiled artifacts:"
echo "  from: $SITE"
echo "  into: $FORK"

linked=0
while IFS= read -r f; do
  mkdir -p "$FORK/$(dirname "$f")"
  ln -sf "$SITE/$f" "$FORK/$f"
  linked=$((linked+1))
done < <(comm -23 \
  <(cd "$SITE" && find . -type f | grep -v __pycache__ | sort) \
  <(cd "$FORK" && find . -type f | grep -v __pycache__ | sort))

echo "Linked $linked files."
