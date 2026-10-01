"""Per-subject sandboxing primitive.

Standardizes how every harness executes untrusted target code. Strategy:
  * Prefer Docker when available (network=none, read-only root, tmpfs /tmp,
    pid/cpu/memory limits).
  * Fall back to subprocess with rlimit-based CPU + memory caps and an
    in-temp-dir working tree. POSIX only.

The public entrypoint is :func:`run_sandboxed`, used by C, Rust, Java, JS, TS,
and Kotlin adapters uniformly. KLEE has its own Docker image (see
``docker/klee/``); this module is intended for the per-subject execution path
during validation runs.
"""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    killed_by_resource: bool = False


@dataclass(frozen=True)
class SandboxLimits:
    cpu_seconds: int = 10
    wall_seconds: int = 30
    memory_mb: int = 512
    file_size_mb: int = 32

    def to_rlimits(self) -> list[tuple[int, int, int]]:
        # (rlimit, soft, hard)
        return [
            (resource.RLIMIT_CPU, self.cpu_seconds, self.cpu_seconds),
            (resource.RLIMIT_AS, self.memory_mb * 1024 * 1024, self.memory_mb * 1024 * 1024),
            (resource.RLIMIT_FSIZE, self.file_size_mb * 1024 * 1024, self.file_size_mb * 1024 * 1024),
        ]


def _has_docker() -> bool:
    return shutil.which("docker") is not None


def _set_rlimits(limits: SandboxLimits) -> None:
    for rlimit, soft, hard in limits.to_rlimits():
        try:
            resource.setrlimit(rlimit, (soft, hard))
        except (ValueError, OSError):
            # Best-effort; some macOS versions reject specific limits.
            pass


def run_sandboxed(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    stdin_text: str | None = None,
    limits: SandboxLimits | None = None,
    use_docker: bool | None = None,
    docker_image: str = "ubuntu:22.04",
    docker_mounts: Sequence[tuple[Path, str]] = (),
) -> SandboxResult:
    """Execute ``argv`` under a sandbox.

    Args:
      argv: command line to execute.
      cwd: working directory (None -> temp dir).
      env: extra environment variables.
      stdin_text: optional stdin payload.
      limits: resource caps.
      use_docker: force docker on (True) / off (False); None -> auto.
      docker_image: image to run in when docker is used.
      docker_mounts: (host_path, container_path) bind mounts (read-only).
    """
    limits = limits or SandboxLimits()
    docker_available = _has_docker()
    use = docker_available if use_docker is None else (use_docker and docker_available)

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(cwd) if cwd is not None else Path(tmp)
        if cwd is None:
            work_dir.mkdir(parents=True, exist_ok=True)

        proc_env = dict(env or {})
        # Always start from an empty environment for determinism.
        full_env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
                    "LANG": "C.UTF-8"}
        full_env.update(proc_env)

        if use:
            mount_args: list[str] = []
            for host, container in docker_mounts:
                mount_args.extend(["-v", f"{host}:{container}:ro"])
            mount_args.extend(["-v", f"{work_dir}:/work"])
            cmd = [
                "docker", "run", "--rm",
                "--network=none",
                "--read-only",
                "--tmpfs", "/tmp:rw,size=64m",
                "--cpus=1",
                f"--memory={limits.memory_mb}m",
                *mount_args,
                "-w", "/work",
            ]
            for k, v in full_env.items():
                cmd.extend(["-e", f"{k}={v}"])
            cmd.append(docker_image)
            cmd.extend(argv)
        else:
            cmd = list(argv)

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(work_dir) if not use else None,
                env=None if use else full_env,
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=limits.wall_seconds,
                preexec_fn=None if use else (lambda: _set_rlimits(limits)),
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(
                returncode=-1,
                stdout=(exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr=(exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
                timed_out=True,
            )
        killed = proc.returncode in (-9, -15) or (proc.returncode != 0 and "MemoryError" in (proc.stderr or ""))
        return SandboxResult(
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            timed_out=False,
            killed_by_resource=killed,
        )
