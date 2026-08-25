# CUNQA environment

CUNQA is not a conda/pip package -- it is a C++/Python platform built from
source (github.com/CESGA-Quantum-Spain/cunqa) and installed under `$HOME` on
CESGA systems. There is no `environment.yml` here for that reason.

Follow CESGA's own installation guide:
https://cesga-quantum-spain.github.io/cunqa/installation/getting_started.html

`scripts/qft_multinode.py` appends `$HOME` to `sys.path` to find the
installed `cunqa` package, matching CUNQA's own examples. If your
installation lives elsewhere, adjust that line.
