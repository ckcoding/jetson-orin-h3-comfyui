#!/usr/bin/env bash

# Source this file before running ComfyUI:
#   source /opt/m/comfyui-env.sh
# It may also be run with bash for diagnostics, but activation only persists
# in the current shell when the file is sourced.

if ! mount -o remount,exec /opt/m; then
    echo "[env] ERROR: could not remount /opt/m with exec enabled" >&2
    return 1 2>/dev/null || exit 1
fi

mount_info="$(findmnt -no SOURCE,FSTYPE,OPTIONS /opt/m)"
echo "[env] /opt/m: ${mount_info}"
if [[ ",${mount_info##* }," == *,noexec,* ]]; then
    echo "[env] ERROR: /opt/m is still mounted noexec" >&2
    return 1 2>/dev/null || exit 1
fi

year="$(date -u +%Y 2>/dev/null || echo 0)"
if ! [[ "${year}" =~ ^[0-9]+$ ]] || (( year < 2024 || year > 2100 )); then
    echo "[env] WARNING: system time looks invalid: $(date -Ins 2>&1)" >&2
    echo "[env] Trying the board's configured NTP server once (no fixed date)." >&2
    if command -v ntpdate >/dev/null 2>&1; then
        if command -v timeout >/dev/null 2>&1; then
            timeout 20 ntpdate 172.31.254.10 || true
        else
            ntpdate 172.31.254.10 || true
        fi
    fi
fi
echo "[env] time: $(date -Ins)"

cd /opt/m/ComfyUI || {
    echo "[env] ERROR: /opt/m/ComfyUI is missing" >&2
    return 1 2>/dev/null || exit 1
}
source /opt/m/ComfyUI/venv/bin/activate

export PATH="/opt/m/tools/git-root/usr/bin:${PATH}"
export GIT_EXEC_PATH=/opt/m/tools/git-root/usr/lib/git-core
export GIT_TEMPLATE_DIR=/opt/m/tools/git-root/usr/share/git-core/templates
export PERL5LIB="/opt/m/tools/git-root/usr/share/perl5${PERL5LIB:+:${PERL5LIB}}"

# Runtime libraries already present on the device. Do not install or replace
# system CUDA, MPI, OpenBLAS, or PyTorch packages.
export LD_LIBRARY_PATH="/opt/m/ComfyUI/runtime/usr/lib/aarch64-linux-gnu/openblas-pthread:/opt/m/ComfyUI/runtime/usr/lib/aarch64-linux-gnu:/opt/m/ComfyUI/runtime/usr/local/cuda-11.4/targets/sbsa-linux/lib:/usr/local/cuda-11.4/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu:/opt/m/tools/git-root/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export GIT_SSL_CAINFO=/opt/m/tools/certs/cacert.pem
export SSL_CERT_FILE=/opt/m/tools/certs/cacert.pem
export PIP_CERT=/opt/m/tools/certs/cacert.pem

echo "[env] git: $(git --version)"
echo "[env] python: $(command -v python)"
echo "[env] pip: $(command -v pip)"
python --version
python -m pip --version
