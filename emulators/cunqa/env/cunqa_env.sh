# Runtime (and build) environment for CUNQA in this benchmark suite.
#
#   source emulators/cunqa/env/cunqa_env.sh
#
# Used by install_cunqa.sh, by the SLURM template and interactively, so that
# building and running always see the same toolchain. Override any of these
# before sourcing:
#
#   CUNQA_VERSION   git tag to build/use                    (default 3.1.0)
#   CUNQA_PREFIX    install prefix: python package in
#                   $CUNQA_PREFIX/cunqa, tools in .../bin    (default $HOME/opt/cunqa-$CUNQA_VERSION)
#   STORE           CUNQA keeps $STORE/.cunqa (qpus.json, logs); must be on a
#                   filesystem shared by login and compute nodes
#   CUNQA_SITE_MODULES  file sourced instead of the built-in module lines on
#                   clusters other than CESGA's QMIO / FT3
#
# The module lines mirror configure.sh of CUNQA at the pinned tag; the Python
# that CUNQA's pybind11 extension was built against must also be the one that
# runs it, which is why runtime loads the same modules as the build.

: "${CUNQA_VERSION:=3.1.0}"
: "${CUNQA_PREFIX:=$HOME/opt/cunqa-$CUNQA_VERSION}"
export CUNQA_VERSION CUNQA_PREFIX

if [ -z "${STORE:-}" ]; then
    export STORE="$HOME"
    echo "cunqa_env: STORE was unset; using \$HOME ($STORE). It must be visible from compute nodes." >&2
fi

if [ -n "${CUNQA_SITE_MODULES:-}" ]; then
    # shellcheck disable=SC1090
    source "$CUNQA_SITE_MODULES"
else
    case "${LMOD_SYSTEM_NAME:-}" in
        QMIO)
            module load qmio/hpc gcc/12.3.0 hpcx-ompi flexiblas/3.3.0 boost cmake/3.27.6 \
                gcccore/12.3.0 eigen/5.0.0 ninja/1.9.0 nlohmann_json/3.12.0 \
                pybind11/2.13.6-python-3.11.9 qiskit/1.2.4-python-3.11.9
            ;;
        FT3)
            module load cesga/2022 gcc/system gcccore/system cmake boost openmpi/5.0.5 \
                cython/3.0.11 pybind11/2.12.0 qiskit/1.2.4
            ;;
        *)
            echo "cunqa_env: unknown cluster (LMOD_SYSTEM_NAME='${LMOD_SYSTEM_NAME:-}')." >&2
            echo "  Provide gcc>=12, CMake>=3.24, Python 3.11 + pybind11, MPI, OpenMP, Boost," >&2
            echo "  Eigen, BLAS/LAPACK, Cython and qiskit 1.2.4 (see CUNQA's README), e.g. in a" >&2
            echo "  file pointed to by CUNQA_SITE_MODULES." >&2
            ;;
    esac
fi

# A conda Python earlier on PATH would shadow the module Python CUNQA is built for.
if [ -n "${CONDA_PREFIX:-}" ] && command -v conda >/dev/null 2>&1; then
    conda deactivate 2>/dev/null || true
fi

case ":$PATH:" in *":$CUNQA_PREFIX/bin:"*) ;; *) export PATH="$CUNQA_PREFIX/bin:$PATH" ;; esac
case ":${PYTHONPATH:-}:" in *":$CUNQA_PREFIX:"*) ;; *) export PYTHONPATH="$CUNQA_PREFIX${PYTHONPATH:+:$PYTHONPATH}" ;; esac

# CUNQA 3.1.0 bug (fixed upstream after the tag, PR #641): the Python qraise
# creates a missing qpus.json as the JSON *string* "{}" instead of an object,
# and the C++ qraise then fails with json type_error.305. Make sure the file
# exists and holds an object; never touch a valid registry.
_cunqa_qpus="$STORE/.cunqa/qpus.json"
if ! python3 -c 'import json,sys; sys.exit(0 if isinstance(json.load(open(sys.argv[1])), dict) else 1)' "$_cunqa_qpus" 2>/dev/null; then
    mkdir -p "$STORE/.cunqa"
    echo '{}' > "$_cunqa_qpus"
    echo "cunqa_env: (re)initialised $_cunqa_qpus as {} (CUNQA 3.1.0 qpus.json bug)" >&2
fi
unset _cunqa_qpus
