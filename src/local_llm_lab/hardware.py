from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .fixtures import get_fixture
from .optional import optional_import


@dataclass
class HardwareProfile:
    name: str
    os: str
    arch: str
    cpu: str
    gpu: str
    unified_memory: bool
    memory_total_gib: float
    memory_available_gib: float
    disk_available_gib: float
    memory_bandwidth_gbps: float | None = None
    vram_total_gib: float | None = None
    metal: bool = False
    cuda: bool = False
    avx: bool = False
    neon: bool = False
    backends: dict[str, str | None] = field(default_factory=dict)
    probes: dict[str, float | str] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _run(args: list[str], timeout: float = 5.0) -> str:
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True, timeout=timeout).strip()
    except Exception:
        return ""


def _sysctl(name: str) -> str:
    return _run(["sysctl", "-n", name])


def _sysconf_total_gib() -> float:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return 0.0
    return pages * page_size / (1024**3)


def _vm_stat_available_gib() -> float | None:
    output = _run(["vm_stat"])
    if not output:
        return None
    page_size = 4096
    available_pages = 0
    for line in output.splitlines():
        if "page size of" in line:
            parts = [part for part in line.split() if part.isdigit()]
            if parts:
                page_size = int(parts[0])
        lowered = line.lower()
        if any(label in lowered for label in ["pages free", "pages inactive", "pages speculative"]):
            digits = "".join(ch for ch in line.split(":")[-1] if ch.isdigit())
            if digits:
                available_pages += int(digits)
    if available_pages <= 0:
        return None
    return available_pages * page_size / (1024**3)


def _cpu_flags() -> tuple[bool, bool]:
    machine = platform.machine().lower()
    cpu_text = " ".join([_sysctl("machdep.cpu.features"), _sysctl("machdep.cpu.leaf7_features")]).lower()
    avx = "avx" in cpu_text
    neon = machine in {"arm64", "aarch64"} or "neon" in cpu_text
    return avx, neon


def _gpu_info() -> tuple[str, bool, float | None]:
    if platform.system() == "Darwin":
        output = _run(["system_profiler", "SPDisplaysDataType"], timeout=10)
        gpu = "Unknown GPU"
        metal = "Metal" in output
        vram = None
        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith("Chipset Model:"):
                gpu = stripped.split(":", 1)[1].strip()
            if stripped.startswith("VRAM") and "GB" in stripped:
                digits = "".join(ch if ch.isdigit() or ch == "." else " " for ch in stripped)
                values = [float(item) for item in digits.split() if item]
                if values:
                    vram = max(values)
        return gpu, metal, vram
    nvidia = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if nvidia:
        first = nvidia.splitlines()[0]
        name = first.split(",", 1)[0].strip()
        total_mib = 0.0
        for line in nvidia.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 2:
                try:
                    total_mib += float(parts[1])
                except ValueError:
                    pass
        return name, False, total_mib / 1024 if total_mib else None
    return "Unknown GPU", False, None


def _memory_probe() -> float:
    size = 16 * 1024 * 1024
    blob = bytearray(os.urandom(size))
    start = time.perf_counter()
    checksum = 0
    for _ in range(12):
        checksum ^= sum(blob[::4096])
    elapsed = max(time.perf_counter() - start, 1e-9)
    _ = checksum
    touched_gib = (size / 4096 * 12) / (1024**2)
    return touched_gib / elapsed


def _disk_probe() -> tuple[float, float]:
    payload = os.urandom(8 * 1024 * 1024)
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        path = Path(handle.name)
        start = time.perf_counter()
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        write_elapsed = max(time.perf_counter() - start, 1e-9)
    try:
        start = time.perf_counter()
        _ = path.read_bytes()
        read_elapsed = max(time.perf_counter() - start, 1e-9)
    finally:
        try:
            path.unlink()
        except OSError:
            pass
    gib = len(payload) / (1024**3)
    return gib / read_elapsed, gib / write_elapsed


def detect_hardware(*, skip_probes: bool = False, fixture: str | None = None) -> HardwareProfile:
    if fixture:
        return HardwareProfile(**get_fixture(fixture))

    system = platform.system() or "Unknown"
    arch = platform.machine() or "Unknown"
    total_gib = 0.0
    cpu = platform.processor() or platform.machine() or "Unknown CPU"
    if system == "Darwin":
        mem = _sysctl("hw.memsize")
        total_gib = int(mem) / (1024**3) if mem.isdigit() else 0.0
        if total_gib <= 0:
            total_gib = _sysconf_total_gib()
        cpu = _sysctl("machdep.cpu.brand_string") or cpu
    else:
        total_gib = _sysconf_total_gib()

    psutil = optional_import("psutil")
    available = None
    if psutil:
        try:
            virtual = psutil.virtual_memory()
            total_gib = virtual.total / (1024**3)
            available = virtual.available / (1024**3)
        except Exception:
            available = None
    if total_gib <= 0:
        raise ValueError(
            "Could not determine total system memory. Pass --hardware fixture:<name> or --hardware profile:<name> to plan against known hardware."
        )
    available_estimated = False
    if available is None and system == "Darwin":
        available = _vm_stat_available_gib()
    if available is None:
        available_estimated = True
        try:
            stat = os.statvfs("/")
            # This is disk fallback only when memory APIs are unavailable.
            _ = stat
            available = max(total_gib * 0.65, 0.0)
        except Exception:
            available = max(total_gib * 0.65, 0.0)

    available = min(max(available, 0.0), total_gib)

    disk = shutil.disk_usage("/")
    disk_available_gib = disk.free / (1024**3)
    gpu, metal, vram = _gpu_info()
    avx, neon = _cpu_flags()
    cuda = shutil.which("nvidia-smi") is not None
    backends = {
        "llama.cpp": shutil.which("llama-cli") or shutil.which("llama-server"),
        "ollama": shutil.which("ollama"),
        "mlx": shutil.which("mlx_lm.generate"),
        "vllm": shutil.which("vllm"),
        "blender": shutil.which("blender"),
    }

    probes: dict[str, float | str] = {}
    if available_estimated:
        probes["memory_available_estimated"] = "estimated from total memory; real available memory could not be read"
    bandwidth = None
    if not skip_probes:
        try:
            measured = _memory_probe()
            probes["memory_probe_gib_s"] = round(measured, 2)
            # Python-level probe is not true DRAM bandwidth. Keep it as a low-confidence signal.
            bandwidth = max(measured * 12, 30.0)
        except Exception as exc:
            probes["memory_probe_error"] = str(exc)
        try:
            read, write = _disk_probe()
            probes["disk_read_gib_s"] = round(read, 2)
            probes["disk_write_gib_s"] = round(write, 2)
        except Exception as exc:
            probes["disk_probe_error"] = str(exc)

    unified = system == "Darwin" and arch.lower() in {"arm64", "aarch64"}
    name = f"{system} {arch}"
    return HardwareProfile(
        name=name,
        os=system,
        arch=arch,
        cpu=cpu,
        gpu=gpu,
        unified_memory=unified,
        memory_total_gib=round(total_gib, 2),
        memory_available_gib=round(available, 2),
        disk_available_gib=round(disk_available_gib, 2),
        memory_bandwidth_gbps=round(bandwidth, 2) if bandwidth else None,
        vram_total_gib=vram,
        metal=metal,
        cuda=cuda,
        avx=avx,
        neon=neon,
        backends=backends,
        probes=probes,
        note="Local detection. Microbenchmarks are lightweight signals, not lab-grade measurements.",
    )


def hardware_to_json(profile: HardwareProfile) -> str:
    return json.dumps(profile.to_dict(), indent=2, sort_keys=True)
