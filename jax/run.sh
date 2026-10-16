#!/bin/bash
# fpm with the Python headers and libpython of the venv whose JAX is embedded.
# Usage: ./run.sh test | ./run.sh run --profile release [-- nelem ntrial]
# FE_PYTHON selects another interpreter (default ../.venv/bin/python).
set -e
cd "$(dirname "$0")"
PY=${FE_PYTHON:-../.venv/bin/python}
INC=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['include'])")
LIB=$("$PY" -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
VER=$("$PY" -c "import sysconfig; print(sysconfig.get_config_var('LDVERSION'))")
cmd=$1
shift
# fpm puts link flags before the objects; --no-as-needed keeps libpython anyway
FE_PYTHON=$PY fpm "$cmd" --c-flag "-I$INC" \
  --link-flag "-L$LIB -Wl,-rpath,$LIB -Wl,--no-as-needed -lpython$VER" "$@"
