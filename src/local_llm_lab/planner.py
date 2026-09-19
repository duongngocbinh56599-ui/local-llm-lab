from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

from .hardware import HardwareProfile, detect_hardware
from .models import ModelProfile, apply_overrides, generic_model_from_params, get_model
from .quantization import QUANTIZATIONS, Quantization, get_quantization
from .units import format_gib


BACKEND_ORDER = ("mlx", "llama.cpp", "ollama", "vllm")


@dataclass
class PlanInputs:
    model: ModelProfile
    hardware: HardwareProfile
    quant: Quantization
    context_tokens: int
    concurrency: int
    model_format: str
    backend: str
    kv_dtype_bytes: float = 2.0


@dataclass
class MemoryEstimate:
    weights_gib: float
    kv_cache_gib: float
    backend_overhead_gib: float
    activation_gib: float
    runtime_required_gib: float
    os_reserve_gib: float
    total_required_gib: float
    available_runtime_gib: float
    margin_gib: float


@dataclass
class PlanResult:
    inputs: PlanInputs
    memory: MemoryEstimate
    verdict: str
    risk_level: str
    recommended_backend: str
    recommended_quantization: str
    expected_decode_tokens_s: dict[str, float]
    confidence: str
    warnings: list[str]
    downgrade_options: list[str]
    formulas: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["inputs"]["quant"] = asdict(self.inputs.quant)
        data["inputs"]["model"] = self.inputs.model.to_dict()
        data["inputs"]["hardware"] = self.inputs.hardware.to_dict()
        return data


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _require_positive_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite value greater than 0.")


def _validate_model_geometry(model: ModelProfile) -> None:
    _require_positive_number("params_b", model.params_b)
    _require_positive_integer("layers", model.layers)
    _require_positive_integer("heads", model.heads)
    _require_positive_integer("kv_heads", model.kv_heads)
    _require_positive_integer("head_dim", model.head_dim)
    if model.kv_heads > model.heads:
        raise ValueError("kv_heads cannot exceed heads.")


def choose_backend(hardware: HardwareProfile, model_format: str, requested: str | None = None) -> str:
    if requested and requested != "auto":
        return requested
    fmt = model_format.lower()
    if hardware.os == "macOS" and hardware.unified_memory:
        if fmt in {"mlx", "safetensors"} and hardware.metal:
            return "mlx"
        return "llama.cpp"
    if hardware.cuda:
        return "vllm" if fmt in {"safetensors", "hf"} else "llama.cpp"
    return "llama.cpp"


def _backend_overhead(backend: str, weights_gib: float) -> float:
    if backend == "mlx":
        return 4.0 + weights_gib * 0.05
    if backend == "ollama":
        return 3.0 + weights_gib * 0.06
    if backend == "vllm":
        return 6.0 + weights_gib * 0.08
    return 2.0 + weights_gib * 0.035


def _format_overhead(model_format: str) -> float:
    fmt = model_format.lower()
    if fmt == "gguf":
        return 1.04
    if fmt in {"mlx", "safetensors", "hf"}:
        return 1.07
    return 1.08


def _os_reserve(hardware: HardwareProfile) -> float:
    total = hardware.memory_total_gib
    if hardware.os == "macOS":
        return max(8.0, total * 0.14)
    if hardware.cuda and hardware.vram_total_gib:
        return max(6.0, total * 0.08)
    return max(6.0, total * 0.10)


def estimate_memory(inputs: PlanInputs) -> MemoryEstimate:
    model = inputs.model
    quant = inputs.quant
    weights_gib = model.params_b * 1_000_000_000 * quant.bytes_per_param / (1024**3)
    weights_gib *= _format_overhead(inputs.model_format)

    kv_bytes = (
        inputs.context_tokens
        * inputs.concurrency
        * model.layers
        * model.kv_heads
        * model.head_dim
        * 2
        * inputs.kv_dtype_bytes
    )
    kv_cache_gib = kv_bytes / (1024**3)
    activation_gib = max(0.75, min(8.0, kv_cache_gib * 0.18 + inputs.concurrency * 0.25))
    backend_overhead_gib = _backend_overhead(inputs.backend, weights_gib)
    runtime_required_gib = weights_gib + kv_cache_gib + activation_gib + backend_overhead_gib
    os_reserve_gib = _os_reserve(inputs.hardware)
    total_required_gib = runtime_required_gib + os_reserve_gib

    if inputs.hardware.unified_memory:
        available_runtime_gib = max(min(inputs.hardware.memory_available_gib, inputs.hardware.memory_total_gib - os_reserve_gib), 0)
    elif inputs.hardware.vram_total_gib:
        available_runtime_gib = max(inputs.hardware.vram_total_gib * 0.92, 0)
    else:
        available_runtime_gib = max(inputs.hardware.memory_available_gib - os_reserve_gib, 0)

    margin_gib = available_runtime_gib - runtime_required_gib
    return MemoryEstimate(
        weights_gib=round(weights_gib, 2),
        kv_cache_gib=round(kv_cache_gib, 2),
        backend_overhead_gib=round(backend_overhead_gib, 2),
        activation_gib=round(activation_gib, 2),
        runtime_required_gib=round(runtime_required_gib, 2),
        os_reserve_gib=round(os_reserve_gib, 2),
        total_required_gib=round(total_required_gib, 2),
        available_runtime_gib=round(available_runtime_gib, 2),
        margin_gib=round(margin_gib, 2),
    )


def _verdict(memory: MemoryEstimate, hardware: HardwareProfile) -> tuple[str, str]:
    total = hardware.memory_total_gib
    if memory.runtime_required_gib <= memory.available_runtime_gib * 0.72 and memory.total_required_gib <= total * 0.78:
        return "smooth", "low"
    if memory.runtime_required_gib <= memory.available_runtime_gib * 0.94 and memory.total_required_gib <= total * 0.92:
        return "tight", "medium"
    if memory.runtime_required_gib <= memory.available_runtime_gib * 1.08 and memory.total_required_gib <= total * 1.02:
        return "not-recommended", "high"
    return "does-not-fit", "extreme"


def _fits_memory_budget(memory: MemoryEstimate, hardware: HardwareProfile, target_ratio: float) -> bool:
    """A plan must respect both the static planning envelope and the runtime memory available now."""
    if memory.total_required_gib > hardware.memory_total_gib * target_ratio:
        return False
    return memory.margin_gib >= 0


def _estimate_tokens_s(inputs: PlanInputs, memory: MemoryEstimate) -> dict[str, float]:
    bandwidth = inputs.hardware.memory_bandwidth_gbps
    if not bandwidth:
        bandwidth = 70.0 if inputs.hardware.unified_memory else 45.0
    active_weight_gib = inputs.model.speed_params_b * 1_000_000_000 * inputs.quant.bytes_per_param / (1024**3)
    active_weight_gib = max(active_weight_gib, 0.5)

    if inputs.hardware.cuda:
        efficiency = 0.72 if inputs.backend in {"vllm", "llama.cpp"} else 0.55
    elif inputs.hardware.unified_memory and inputs.hardware.metal:
        efficiency = 0.64 if inputs.backend in {"mlx", "llama.cpp"} else 0.50
    else:
        efficiency = 0.24

    pressure_penalty = 1.0
    if memory.margin_gib < 0:
        pressure_penalty = 0.25
    elif memory.margin_gib < 8:
        pressure_penalty = 0.55
    elif memory.margin_gib < 20:
        pressure_penalty = 0.78

    mid = max((bandwidth / active_weight_gib) * efficiency * pressure_penalty, 0.05)
    return {"low": round(mid * 0.65, 2), "mid": round(mid, 2), "high": round(mid * 1.35, 2)}


def _recommend_quant(inputs: PlanInputs, target_ratio: float = 0.86) -> str:
    ordered = sorted(QUANTIZATIONS.values(), key=lambda q: q.bytes_per_param, reverse=True)
    for quant in ordered:
        trial_inputs = PlanInputs(
            model=inputs.model,
            hardware=inputs.hardware,
            quant=quant,
            context_tokens=inputs.context_tokens,
            concurrency=inputs.concurrency,
            model_format=inputs.model_format,
            backend=inputs.backend,
            kv_dtype_bytes=inputs.kv_dtype_bytes,
        )
        trial_memory = estimate_memory(trial_inputs)
        if _fits_memory_budget(trial_memory, inputs.hardware, target_ratio):
            return quant.name
    return "no-safe-local-quant"


def _downgrades(inputs: PlanInputs, memory: MemoryEstimate, verdict: str) -> list[str]:
    options: list[str] = []
    if verdict in {"not-recommended", "does-not-fit"}:
        recommended = _recommend_quant(inputs, target_ratio=0.86)
        if recommended == "no-safe-local-quant":
            options.append("No supported quantization fits safely at this context on this hardware.")
        elif recommended != inputs.quant.name:
            options.append(f"Try {recommended} instead of {inputs.quant.name}.")
        for ctx in (32768, 16384, 8192, 4096, 2048):
            if ctx >= inputs.context_tokens:
                continue
            trial = PlanInputs(
                model=inputs.model,
                hardware=inputs.hardware,
                quant=inputs.quant,
                context_tokens=ctx,
                concurrency=max(1, inputs.concurrency),
                model_format=inputs.model_format,
                backend=inputs.backend,
                kv_dtype_bytes=inputs.kv_dtype_bytes,
            )
            trial_memory = estimate_memory(trial)
            if _fits_memory_budget(trial_memory, inputs.hardware, 0.9):
                options.append(f"Reduce context to {ctx} tokens.")
                break
        if inputs.concurrency > 1:
            options.append("Reduce concurrency to 1 while validating fit.")
        if inputs.model.params_b >= 200:
            options.append("Consider remote inference, hybrid/offloaded inference, or a smaller distilled model.")
        if inputs.model.params_b >= 500:
            options.append("For 600B+ models, avoid local-only deployment unless you have measured headroom.")
    elif verdict == "tight":
        options.append("Keep other apps closed and monitor memory pressure during first real run.")
        options.append("Use a shorter context or lower concurrency for long sessions.")
    else:
        options.append("Run a short real backend benchmark before relying on production latency.")
    return options[:5]


def _warnings(inputs: PlanInputs, memory: MemoryEstimate) -> list[str]:
    warnings: list[str] = []
    if inputs.model.confidence == "low":
        warnings.append("Model metadata confidence is low; override layers/heads/KV heads/head dim if known.")
    if inputs.model.architecture == "moe":
        warnings.append("MoE memory stores total parameters; speed estimate uses active parameters and is lower confidence.")
    if inputs.hardware.note:
        warnings.append(inputs.hardware.note)
    if memory.margin_gib < 0:
        warnings.append(f"Estimated runtime exceeds available runtime memory by {format_gib(abs(memory.margin_gib))}.")
    if inputs.context_tokens >= 32768:
        warnings.append("Long context can dominate KV cache and prefill time.")
    if inputs.hardware.unified_memory and inputs.hardware.memory_total_gib < 64:
        warnings.append("This is not a high unified-memory device; very large models are unlikely to fit.")
    return warnings


def make_plan(
    *,
    model_name: str | None = None,
    params: str | None = None,
    quant_name: str = "Q4_K_M",
    context_tokens: int = 8192,
    concurrency: int = 1,
    backend: str | None = None,
    model_format: str | None = None,
    hardware_fixture: str | None = None,
    hardware: HardwareProfile | None = None,
    layers: int | None = None,
    heads: int | None = None,
    kv_heads: int | None = None,
    head_dim: int | None = None,
    kv_dtype_bytes: float = 2.0,
) -> PlanResult:
    _require_positive_integer("context_tokens", context_tokens)
    _require_positive_integer("concurrency", concurrency)
    _require_positive_number("kv_dtype_bytes", kv_dtype_bytes)
    if model_name:
        model = get_model(model_name)
    elif params:
        model = generic_model_from_params(params)
    else:
        raise ValueError("Either --model or --params is required.")

    fmt = model_format or model.default_format
    model = apply_overrides(
        model,
        params=params if model_name and params else None,
        layers=layers,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        model_format=fmt,
    )
    _validate_model_geometry(model)
    hw = hardware or detect_hardware(skip_probes=True, fixture=hardware_fixture)
    chosen_backend = choose_backend(hw, fmt, backend)
    quant = get_quantization(quant_name)
    inputs = PlanInputs(
        model=model,
        hardware=hw,
        quant=quant,
        context_tokens=context_tokens,
        concurrency=concurrency,
        model_format=fmt,
        backend=chosen_backend,
        kv_dtype_bytes=kv_dtype_bytes,
    )
    memory = estimate_memory(inputs)
    verdict, risk = _verdict(memory, hw)
    tokens_s = _estimate_tokens_s(inputs, memory)
    recommended_quant = _recommend_quant(inputs)
    warnings = _warnings(inputs, memory)
    downgrades = _downgrades(inputs, memory, verdict)
    confidence = model.confidence
    if hardware_fixture or hw.note.startswith("Illustrative"):
        confidence = "demo"
    if memory.margin_gib < 0 or model.confidence == "low":
        confidence = "low" if confidence != "demo" else "demo"
    formulas = {
        "weights": "params * quant_effective_bytes_per_param * format_overhead",
        "kv_cache": "ctx * concurrency * layers * kv_heads * head_dim * 2(K,V) * kv_dtype_bytes",
        "runtime_required": "weights + kv_cache + activation + backend_overhead",
        "total_required": "runtime_required + os_reserve",
    }
    return PlanResult(
        inputs=inputs,
        memory=memory,
        verdict=verdict,
        risk_level=risk,
        recommended_backend=chosen_backend,
        recommended_quantization=recommended_quant,
        expected_decode_tokens_s=tokens_s,
        confidence=confidence,
        warnings=warnings,
        downgrade_options=downgrades,
        formulas=formulas,
    )
