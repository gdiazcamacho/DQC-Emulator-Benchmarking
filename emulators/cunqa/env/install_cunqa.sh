#!/bin/bash
# Build and install the pinned CUNQA release for this benchmark suite.
#
#   bash emulators/cunqa/env/install_cunqa.sh
#
# Same procedure on every cluster: clone the tag, configure/build/install with
# CMake into $CUNQA_PREFIX, record the exact commit. Environment and overrides
# are documented in cunqa_env.sh. Building takes a while and compiles a patched
# qiskit-aer: run it on a compute node (e.g. `compute -c 16` at CESGA), not a
# shared login node.
#
#   CUNQA_SRC         where to clone      (default $HOME/src/cunqa-$CUNQA_VERSION)
#   CUNQA_BUILD_JOBS  parallel build jobs (default 8)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/cunqa_env.sh"
set -eo pipefail

CUNQA_SRC="${CUNQA_SRC:-$HOME/src/cunqa-$CUNQA_VERSION}"
JOBS="${CUNQA_BUILD_JOBS:-8}"

echo "CUNQA $CUNQA_VERSION: source $CUNQA_SRC -> prefix $CUNQA_PREFIX (STORE=$STORE)"

if [ ! -d "$CUNQA_SRC/.git" ]; then
    mkdir -p "$(dirname "$CUNQA_SRC")"
    git clone --depth 1 --branch "$CUNQA_VERSION" \
        https://github.com/CESGA-Quantum-Spain/cunqa.git "$CUNQA_SRC"
fi
cd "$CUNQA_SRC"
COMMIT="$(git rev-parse HEAD)"

cmake -B build/ -DCMAKE_INSTALL_PREFIX="$CUNQA_PREFIX"
cmake --build build/ --parallel "$JOBS"
cmake --install build/

# CUNQA's VERSION.txt is not always bumped with the tag (3.1.0 still says
# 3.0.0), so record what was actually built.
printf 'tag=%s\ncommit=%s\nbuilt=%s\nhost=%s\n' "$CUNQA_VERSION" "$COMMIT" \
    "$(date -Iseconds)" "$(hostname)" > "$CUNQA_PREFIX/BUILD_INFO"

python3 -c "import cunqa, cunqa.qc_protocols, cunqa.qpu; print('cunqa importable from', cunqa.__file__)"
command -v qraise >/dev/null && echo "qraise: $(command -v qraise)"
echo "Done. Recorded in $CUNQA_PREFIX/BUILD_INFO"
