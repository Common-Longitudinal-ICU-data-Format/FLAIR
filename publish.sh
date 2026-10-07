#!/usr/bin/env bash
# Build, verify and publish flair-benchmark. See PUBLISHING.md.
#   ./publish.sh --check            build + verify only, no upload
#   ./publish.sh --test [TOKEN]     upload to test.pypi.org
#   ./publish.sh [TOKEN]            upload to pypi.org, then tag v<version>
# Omit TOKEN to be prompted (keeps it out of shell history).
set -euo pipefail
cd "$(dirname "$0")"

MODE=pypi
case "${1:-}" in --test|--check) MODE=${1#--}; shift ;; esac
TOKEN=${1:-${UV_PUBLISH_TOKEN:-}}
die() { echo "error: $*" >&2; exit 1; }

[ -z "$(git status --porcelain)" ] || die "git tree not clean; commit first"
VER=$(uv run python -c "import flair_benchmark; print(flair_benchmark.__version__)")
echo "==> flair-benchmark $VER ($MODE)"

if [ "$MODE" = pypi ]; then
  [ "$(curl -s -o /dev/null -w '%{http_code}' "https://pypi.org/pypi/flair-benchmark/$VER/json")" != 200 ] \
    || die "$VER already on PyPI; bump __version__ in flair_benchmark/constants.py"
  ! git rev-parse -q --verify "refs/tags/v$VER" >/dev/null || die "tag v$VER already exists"
fi
if [ "$MODE" != check ] && [ -z "$TOKEN" ]; then
  read -rsp "$MODE token: " TOKEN; echo
  [ -n "$TOKEN" ] || die "no token given"
fi

echo "==> tests"
uv run --extra dev pytest -q

echo "==> build"
rm -rf wheel && uv build --out-dir wheel/ && rm -f wheel/.gitignore
! tar -tzf wheel/*.tar.gz | grep -qE '/(docs|assets)/|uv\.lock' || die "sdist leaked docs/assets/uv.lock"
uvx twine check wheel/*
uv run --no-project --with wheel/*.whl flair --help >/dev/null
echo "==> artifacts OK"

[ "$MODE" = check ] && exit 0

if [ "$MODE" = test ]; then
  UV_PUBLISH_TOKEN=$TOKEN uv publish --publish-url https://test.pypi.org/legacy/ wheel/*
  echo "==> https://test.pypi.org/project/flair-benchmark/$VER/"
else
  UV_PUBLISH_TOKEN=$TOKEN uv publish wheel/*
  git tag "v$VER" && git push origin "v$VER"
  echo "==> https://pypi.org/project/flair-benchmark/$VER/"
fi
git status --porcelain wheel | grep -q . && echo "note: wheel/ rebuilt; commit it"
exit 0
