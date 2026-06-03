# Reproducible build/test environment for hrx-pytorch-smoke-test.
#
# The image carries ONLY the system build toolchain (clang-23/lld/cmake/ninja/
# ccache/python3.12/libzstd). The HRX source, the nightly ROCm+PyTorch venv, and
# the A/B run all happen at `docker run` time so they track whatever HRX ref you
# point at — nothing GPU-specific or /home-specific is baked in.
#
# Build:   docker build -t hrx-smoke-toolchain .
# Run:     see scripts/docker_smoke.sh (handles GPU passthrough + clone + run),
#          or the README "Run it in a clean container" section.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

# Toolchain layer — cached and reused across runs.
COPY scripts/install_system_deps.sh /tmp/install_system_deps.sh
RUN bash /tmp/install_system_deps.sh \
 && rm -f /tmp/install_system_deps.sh \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /work
CMD ["bash"]
