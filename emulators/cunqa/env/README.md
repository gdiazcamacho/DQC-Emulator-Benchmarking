# CUNQA environment

CUNQA is a C++/Python platform built from source, not a conda/pip package, so
this directory holds a build script instead of an `environment.yml`. The suite
pins a release tag and builds it the same way on every cluster, CESGA included
(the site Lmod modules lag behind the releases: QMIO had 2.4.0 when 3.1.0 was
current, and 2.4.0 has a different, incompatible telegate API).

Pinned: **CUNQA 3.1.0** (tag `3.1.0`, commit `29130f6`). Its `VERSION.txt`
still reads 3.0.0; cite the tag and the commit recorded in
`$CUNQA_PREFIX/BUILD_INFO`.

## Install (once)

On a compute node, from the repository root:

```bash
bash emulators/cunqa/env/install_cunqa.sh
```

This sources `cunqa_env.sh`, clones the tag to `$HOME/src/cunqa-3.1.0`, builds,
installs to `$HOME/opt/cunqa-3.1.0` and writes `BUILD_INFO`. All paths are
overridable (see the header of `cunqa_env.sh`).

* **CESGA (QMIO, FT3):** module lines are built in, selected by
  `LMOD_SYSTEM_NAME`. `$STORE` is already defined.
* **Other clusters:** write a small file that loads the dependencies listed in
  CUNQA's README (gcc 12, CMake >= 3.24, Python 3.11 + pybind11, MPI, OpenMP,
  Boost, Eigen, BLAS/LAPACK, Cython, qiskit 1.2.4), point `CUNQA_SITE_MODULES`
  at it, and set `STORE` to a directory shared by login and compute nodes.

## Use

```bash
source emulators/cunqa/env/cunqa_env.sh
python3 -m pytest -q tests/                     # CUNQA tests must not be skipped
python3 emulators/cunqa/scripts/qft_multinode.py --num_qubits 4 --num_nodes 2 \
        --theta 0.1875 --comm_mode packed --shadow --shots 50
```

The SLURM template sources the same file, so jobs see exactly the build-time
toolchain. `pytest` may need `python3 -m pip install --user pytest`.
