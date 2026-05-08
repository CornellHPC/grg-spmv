"""Build and cache the official BOLT-LMM binary used by the comparison."""

from __future__ import annotations

import argparse
import inspect
import logging
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import subprocess
import tarfile
import textwrap
import tempfile
import urllib.request


LOGGER = logging.getLogger(__name__)
BOLT_VERSION = "BOLT-LMM_v2.5"
BOLT_DOWNLOAD_URL = f"https://storage.googleapis.com/broad-alkesgroup-public/BOLT-LMM/downloads/{BOLT_VERSION}.tar.gz"
BOLT_DOWNLOAD_TIMEOUT_SECONDS = 120


def _cache_root(cache_dir: Path) -> Path:
    return Path(cache_dir).expanduser().resolve() / BOLT_VERSION


def cached_bolt_binary(cache_dir: Path) -> Path:
    return _cache_root(cache_dir) / "src" / "bolt"


def bolt_make_command(*, jobs: int) -> list[str]:
    return [
        "make",
        "-j",
        str(int(jobs)),
        "CC=g++",
        "BOOST_INSTALL_DIR=",
        "NLOPT_INSTALL_DIR=",
        "ZSTD_DIR=/usr/include",
        "linking=dynamic",
        "LLIBS=-lboost_program_options -lboost_iostreams -lzstd -lz",
        "LLAPACK=-llapack -lblas -lgfortran",
    ]


def _nlopt_stub_source() -> str:
    return textwrap.dedent(
        r'''
        #include "NonlinearOptMulti.hpp"
        #include <stdexcept>

        namespace NonlinearOptMulti {
          namespace ublas = boost::numeric::ublas;

          std::vector < ublas::matrix <double> > constrainedNR
          (double &dLLpred, ublas::vector <double> &p,
           const std::vector < ublas::matrix <double> > &Vegs,
           const ublas::vector <double> &grad,
           const ublas::matrix <double> &AI,
           double maxStepNorm) {
            throw std::runtime_error("NLopt-dependent REML-AI path is disabled in the lmmInfOnly reference build");
          }
        }
        '''
    ).strip() + "\n"


def preflight_bolt(binary: Path) -> None:
    result = subprocess.run([str(binary), "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"BOLT binary failed preflight: {binary}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    output = result.stdout + result.stderr
    if "BOLT-LMM" not in output:
        raise RuntimeError(f"BOLT binary preflight output did not look like BOLT-LMM: {binary}")


def _download_tarball(cache_dir: Path, url: str = BOLT_DOWNLOAD_URL) -> Path:
    tarball = Path(cache_dir).expanduser().resolve() / f"{BOLT_VERSION}.tar.gz"
    if tarball.exists() and tarball.stat().st_size > 0:
        return tarball
    tarball.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading BOLT-LMM from %s", url)
    with urllib.request.urlopen(url, timeout=BOLT_DOWNLOAD_TIMEOUT_SECONDS) as response:
        status = getattr(response, "status", None)
        if status is not None and int(status) >= 400:
            raise RuntimeError(f"BOLT-LMM download failed with HTTP status {status}: {url}")
        with tempfile.NamedTemporaryFile(dir=tarball.parent, delete=False) as tmp:
            shutil.copyfileobj(response, tmp)
            tmp_path = Path(tmp.name)
    if tmp_path.stat().st_size <= 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"BOLT-LMM download produced an empty tarball: {url}")
    tmp_path.replace(tarball)
    return tarball


def _validated_tar_members(archive: tarfile.TarFile, destination: Path) -> tuple[tarfile.TarInfo, ...]:
    dest = Path(destination).expanduser().resolve()
    members: list[tarfile.TarInfo] = []
    for member in archive.getmembers():
        name = str(member.name)
        member_path = PurePosixPath(name)
        parts = member_path.parts
        if not name or not parts:
            raise ValueError(f"unsafe BOLT tar member name: {name!r}")
        if member_path.is_absolute() or any(part == ".." for part in parts):
            raise ValueError(f"unsafe BOLT tar member path: {name!r}")
        if parts[0] != BOLT_VERSION:
            raise ValueError(f"unexpected BOLT tar member outside {BOLT_VERSION}: {name!r}")
        if len(parts) == 1 and not member.isdir():
            raise ValueError(f"unexpected non-directory BOLT tar root member: {name!r}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"unsupported BOLT tar member type for {name!r}")
        target = (dest / name).resolve(strict=False)
        if target != dest and not target.is_relative_to(dest):
            raise ValueError(f"unsafe BOLT tar member destination: {name!r}")
        members.append(member)
    return tuple(members)


def _extract_tarball(cache_dir: Path, tarball: Path) -> None:
    cache_root = _cache_root(cache_dir)
    if (cache_root / "src" / "Makefile").exists():
        return
    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Extracting %s under %s", tarball, cache_root.parent)
    with tarfile.open(tarball, "r:gz") as archive:
        members = _validated_tar_members(archive, cache_root.parent)
        if "filter" in inspect.signature(archive.extractall).parameters:
            archive.extractall(cache_root.parent, members=members, filter="data")
        else:
            archive.extractall(cache_root.parent, members=members)
    if not (cache_root / "src" / "Makefile").exists():
        raise FileNotFoundError(f"BOLT source Makefile not found after extraction: {cache_root / 'src' / 'Makefile'}")


def _patch_cached_source(cache_dir: Path) -> None:
    source = _cache_root(cache_dir) / "src" / "NonlinearOptMulti.cpp"
    if not source.exists():
        raise FileNotFoundError(source)
    source.write_text(_nlopt_stub_source(), encoding="utf-8")


def build_cached_bolt(*, cache_dir: Path, jobs: int) -> Path:
    tarball = _download_tarball(cache_dir)
    _extract_tarball(cache_dir, tarball)
    _patch_cached_source(cache_dir)
    src = _cache_root(cache_dir) / "src"
    cmd = bolt_make_command(jobs=int(jobs))
    LOGGER.info("Building cached BOLT-LMM in %s", src)
    result = subprocess.run(cmd, cwd=src, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"BOLT source build failed with exit {result.returncode}\n{result.stdout}")
    binary = cached_bolt_binary(cache_dir)
    preflight_bolt(binary)
    return binary


def ensure_cached_bolt(*, cache_dir: Path, jobs: int, force: bool = False) -> Path:
    binary = cached_bolt_binary(cache_dir)
    if binary.exists() and not force:
        preflight_bolt(binary)
        LOGGER.info("Using cached BOLT-LMM binary at %s", binary)
        return binary
    return build_cached_bolt(cache_dir=cache_dir, jobs=int(jobs))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the cached BOLT-LMM v2.5 binary for scripts.bolt_lmm_inf")
    parser.add_argument("--cacheDir", type=Path, required=True, help="Writable cache directory for the BOLT tarball and build tree.")
    parser.add_argument("--jobs", type=int, required=True, help="Parallel make jobs for building BOLT.")
    parser.add_argument("--force", action="store_true", help="Rebuild BOLT even if a cached binary already exists.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), required=True, help="Python logging level.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level)), format="%(levelname)s:%(name)s:%(message)s")
    if int(args.jobs) < 1:
        parser.error("--jobs must be >= 1")
    binary = ensure_cached_bolt(cache_dir=Path(args.cacheDir), jobs=int(args.jobs), force=bool(args.force))
    print(binary)


if __name__ == "__main__":
    main()
