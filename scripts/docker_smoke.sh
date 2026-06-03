#!/usr/bin/env bash
# docker_smoke.sh — the acceptance check: build the toolchain image, then in a
# FRESH container with the GPU passed through, clone the test repo and run
# build_and_test.sh end to end. Proves the repo is self-contained.
#
# By default it clones THIS working tree's committed state (so commit first) into
# the container. Override:
#   TESTREPO_SRC=<git url or path>   what to clone as the test repo (default: this repo)
#   IMAGE=<name>                     toolchain image tag (default: hrx-smoke-toolchain)
#   HRX_REF=<branch>                 HRX branch/tag/sha to build (forwarded to the script)
# Any extra args are forwarded to build_and_test.sh, e.g.:
#   scripts/docker_smoke.sh --stress
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-hrx-smoke-toolchain}"
SRC="${TESTREPO_SRC:-$REPO_ROOT}"
EXTRA="$*"

command -v docker >/dev/null || { echo "ERROR: docker not found." >&2; exit 1; }

# GPU passthrough: /dev/kfd + each render node, plus render/video groups.
gpu_args=(--device=/dev/kfd --security-opt seccomp=unconfined)
for d in /dev/dri/renderD* /dev/dri/card*; do
  [ -e "$d" ] && gpu_args+=(--device="$d")
done
for g in render video; do
  gid="$(getent group "$g" | cut -d: -f3 || true)"
  [ -n "$gid" ] && gpu_args+=(--group-add "$gid")
done

# Mount the clone source read-only if it's a local path; a URL is cloned directly.
mount_args=(); clone_src="$SRC"
if [ -d "$SRC" ]; then mount_args=(-v "$SRC:/srcrepo:ro"); clone_src="/srcrepo"; fi

echo ">> building toolchain image: $IMAGE"
docker build -t "$IMAGE" -f "$REPO_ROOT/Dockerfile" "$REPO_ROOT"

echo ">> running build_and_test.sh in a fresh container (GPU passthrough)"
docker run --rm "${gpu_args[@]}" "${mount_args[@]}" \
  -e HRX_REF="${HRX_REF:-}" -e EXTRA_FLAGS="$EXTRA" -e CLONE_SRC="$clone_src" \
  "$IMAGE" bash -lc '
    set -euo pipefail
    git config --global --add safe.directory "*" || true   # throwaway container; mount is owned by another uid
    git clone "$CLONE_SRC" /work/test-repo
    cd /work/test-repo
    [ -n "${HRX_REF:-}" ] && export HRX_REF
    # shellcheck disable=SC2086
    ./build_and_test.sh $EXTRA_FLAGS
  '
