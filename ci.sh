#!/bin/bash
# Local multi-compiler check: test in both profiles with every compiler
# found on PATH, then run the app; then the jax/ and socket/ paradigm tests
# if .venv (with JAX) exists. Compilers not found are listed, not hidden.
# GitHub Actions (.github/workflows/ci.yml) covers more platforms.

cd "$(dirname "$0")"
# setvars.sh refuses to run twice in one environment, even if ifx is not on PATH
command -v ifx > /dev/null || source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
FAILED=()
SKIPPED=()

for FC in gfortran ifx nvfortran flang; do
    command -v "$FC" > /dev/null || { SKIPPED+=("$FC"); continue; }
    for PROFILE in debug release; do
        echo "--- $FC $PROFILE"
        if fpm test --compiler "$FC" --profile "$PROFILE" \
            && fpm run --compiler "$FC" --profile "$PROFILE" -- 50 1 > /dev/null; then
            echo "✅ $FC $PROFILE"
        else
            FAILED+=("$FC $PROFILE")
        fi
    done
done

if [ -x .venv/bin/python ]; then
    echo "--- jax/ and socket/ (gfortran release)"
    jax/run.sh test --profile release && echo "✅ jax" || FAILED+=("jax")
    (cd socket && fpm test --profile release) && echo "✅ socket" || FAILED+=("socket")
else
    SKIPPED+=("jax/ and socket/ (no .venv)")
fi

[ ${#SKIPPED[@]} -gt 0 ] && printf '⏭️  skipped: %s\n' "${SKIPPED[@]}"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "✅ All checks passed"
else
    printf '❌ %s\n' "${FAILED[@]}"
    exit 1
fi
