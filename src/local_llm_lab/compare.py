from __future__ import annotations

import html
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hardware import HardwareProfile
from .planner import PlanResult, make_plan
from .report import _svg_bar_chart


DEFAULT_COMPARE_QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "IQ2_XS"]
DEFAULT_COMPARE_CONTEXTS = [4096, 8192, 16384, 32768]
DEFAULT_COMPARE_BACKENDS = ["auto"]

VERDICT_RANK = {"smooth": 0, "tight": 1, "not-recommended": 2, "does-not-fit": 3}
RISK_RANK = {"low": 0, "medium": 1, "high": 2, "extreme": 3}


@dataclass(frozen=True)
class CompareRequest:
    model_name: str | None
    params: str | None
    quantizations: list[str]
    contexts: list[int]
    backends: list[str]
    concurrency: int
    model_format: str | None
    hardware_fixture: str | None
    hardware_profile: HardwareProfile | None = None
    hardware_label: str | None = None
    layers: int | None = None
    heads: int | None = None
    kv_heads: int | None = None
    head_dim: int | None = None
    kv_dtype_bytes: float = 2.0


def parse_csv(value: str | None, defaults: list[str]) -> list[str]:
    if not value:
        return list(defaults)
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated value.")
    return items


def parse_int_csv(value: str | None, defaults: list[int]) -> list[int]:
    if not value:
        return list(defaults)
    try:
        items = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid integer list: {value!r}") from exc
    if not items:
        raise ValueError("Expected at least one comma-separated integer.")
    if any(item <= 0 for item in items):
        raise ValueError("Integer list values must all be positive.")
    return items


def _sort_key(plan: PlanResult) -> tuple[float, ...]:
    return (
        VERDICT_RANK.get(plan.verdict, 9),
        RISK_RANK.get(plan.risk_level, 9),
        -plan.inputs.quant.bytes_per_param,
        -float(plan.inputs.context_tokens),
        -plan.expected_decode_tokens_s["mid"],
        -plan.memory.margin_gib,
    )


def _deploy_command(request: CompareRequest, plan: PlanResult) -> str:
    command = ["python3", "-m", "local_llm_lab", "deploy"]
    if request.model_name:
        command.extend(["--model", request.model_name])
    elif request.params:
        command.extend(["--params", request.params])
    command.extend(["--quant", plan.inputs.quant.name])
    command.extend(["--ctx", str(plan.inputs.context_tokens)])
    command.extend(["--backend", plan.recommended_backend])
    if request.model_format:
        command.extend(["--format", request.model_format])
    if request.hardware_label and request.hardware_label != "local":
        command.extend(["--hardware", request.hardware_label])
    command.extend(["--out", f".local-llm-lab/deploy/{plan.inputs.model.id}-{plan.inputs.quant.name.lower()}-{plan.inputs.context_tokens}"])
    return " ".join(shlex.quote(part) for part in command)


def _recommend_command(request: CompareRequest, *, target: str = "tight") -> str:
    command = ["python3", "-m", "local_llm_lab", "recommend"]
    if request.model_name:
        command.extend(["--model", request.model_name])
    elif request.params:
        command.extend(["--params", request.params])
    if request.hardware_label:
        command.extend(["--hardware", request.hardware_label])
    command.extend(["--quants", ",".join(request.quantizations)])
    command.extend(["--contexts", ",".join(str(item) for item in request.contexts)])
    command.extend(["--backends", ",".join(request.backends)])
    command.extend(["--target", target])
    return " ".join(shlex.quote(part) for part in command)


def compare_plans(request: CompareRequest) -> dict[str, Any]:
    plans: list[PlanResult] = []
    for backend in request.backends:
        for quant in request.quantizations:
            for context in request.contexts:
                plans.append(
                    make_plan(
                        model_name=request.model_name,
                        params=request.params,
                        quant_name=quant,
                        context_tokens=context,
                        concurrency=request.concurrency,
                        backend=backend,
                        model_format=request.model_format,
                        hardware_fixture=request.hardware_fixture,
                        hardware=request.hardware_profile,
                        layers=request.layers,
                        heads=request.heads,
                        kv_heads=request.kv_heads,
                        head_dim=request.head_dim,
                        kv_dtype_bytes=request.kv_dtype_bytes,
                    )
                )

    sorted_plans = sorted(plans, key=_sort_key)
    runnable = [plan for plan in sorted_plans if plan.verdict in {"smooth", "tight"}]
    best = runnable[0] if runnable else sorted_plans[0]
    hardware_label = request.hardware_label or (f"fixture:{request.hardware_fixture}" if request.hardware_fixture else "local")
    verdict_counts: dict[str, int] = {}
    for plan in plans:
        verdict_counts[plan.verdict] = verdict_counts.get(plan.verdict, 0) + 1

    rows = []
    for plan in sorted_plans:
        rows.append(
            {
                "model": plan.inputs.model.id,
                "backend": plan.recommended_backend,
                "requested_backend": plan.inputs.backend,
                "quantization": plan.inputs.quant.name,
                "context_tokens": plan.inputs.context_tokens,
                "concurrency": plan.inputs.concurrency,
                "verdict": plan.verdict,
                "risk_level": plan.risk_level,
                "runtime_required_gib": plan.memory.runtime_required_gib,
                "available_runtime_gib": plan.memory.available_runtime_gib,
                "margin_gib": plan.memory.margin_gib,
                "weights_gib": plan.memory.weights_gib,
                "kv_cache_gib": plan.memory.kv_cache_gib,
                "decode_tokens_s_mid": plan.expected_decode_tokens_s["mid"],
                "confidence": plan.confidence,
                "deploy_command": _deploy_command(request, plan),
            }
        )

    return {
        "compare": {
            "request": {
                "model": request.model_name,
                "params": request.params,
                "quantizations": request.quantizations,
                "contexts": request.contexts,
                "backends": request.backends,
                "concurrency": request.concurrency,
                "format": request.model_format,
                "hardware": hardware_label,
            },
            "summary": {
                "total_plans": len(plans),
                "runnable_plans": len(runnable),
                "verdict_counts": verdict_counts,
                "best": {
                    "backend": best.recommended_backend,
                    "quantization": best.inputs.quant.name,
                    "context_tokens": best.inputs.context_tokens,
                    "verdict": best.verdict,
                    "risk_level": best.risk_level,
                    "margin_gib": best.memory.margin_gib,
                    "decode_tokens_s_mid": best.expected_decode_tokens_s["mid"],
                    "deploy_command": _deploy_command(request, best),
                },
                "recommend_command": _recommend_command(request),
            },
            "rows": rows,
        }
    }


def compare_markdown(data: dict[str, Any]) -> str:
    compare = data["compare"]
    summary = compare["summary"]
    best = summary["best"]
    lines = [
        "# local-llm-lab Compare Report",
        "",
        "## Best Candidate",
        "",
        f"- Backend: `{best['backend']}`",
        f"- Quantization: `{best['quantization']}`",
        f"- Context: `{best['context_tokens']}` tokens",
        f"- Verdict: **{best['verdict']}**",
        f"- Risk: **{best['risk_level']}**",
        f"- Margin: `{best['margin_gib']} GiB`",
        f"- Estimated decode: `{best['decode_tokens_s_mid']} tok/s`",
        "",
        "## Matrix",
        "",
        "| Backend | Quant | Context | Verdict | Risk | Runtime GiB | Margin GiB | Decode tok/s |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in compare["rows"]:
        lines.append(
            "| {backend} | {quantization} | {context_tokens} | {verdict} | {risk_level} | "
            "{runtime_required_gib} | {margin_gib} | {decode_tokens_s_mid} |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


def _verdict_class(verdict: str) -> str:
    return {
        "smooth": "ok",
        "tight": "warn",
        "not-recommended": "bad",
        "does-not-fit": "fail",
    }.get(verdict, "neutral")


def compare_html(data: dict[str, Any], chart_files: list[str]) -> str:
    compare = data["compare"]
    summary = compare["summary"]
    best = summary["best"]
    verdict_counts = summary.get("verdict_counts", {})
    best_command = html.escape(str(best.get("deploy_command", "")), quote=True)
    recommend_command = html.escape(str(summary.get("recommend_command", "")), quote=True)
    row_chunks = []
    for row in compare["rows"]:
        is_best = (
            row["backend"] == best["backend"]
            and row["quantization"] == best["quantization"]
            and row["context_tokens"] == best["context_tokens"]
            and row["verdict"] == best["verdict"]
        )
        row_chunks.append(
            "<tr"
            f" class=\"{'best-row' if is_best else ''}\""
            f" data-backend=\"{html.escape(str(row['backend']), quote=True)}\""
            f" data-quant=\"{html.escape(str(row['quantization']), quote=True)}\""
            f" data-verdict=\"{html.escape(str(row['verdict']), quote=True)}\""
            f" data-context=\"{row['context_tokens']}\""
            f" data-margin=\"{row['margin_gib']}\""
            f" data-runtime=\"{row['runtime_required_gib']}\""
            f" data-decode=\"{row['decode_tokens_s_mid']}\""
            ">"
            f"<td>{html.escape(str(row['backend']))}</td>"
            f"<td>{html.escape(str(row['quantization']))}</td>"
            f"<td>{row['context_tokens']}</td>"
            f"<td><span class=\"badge {_verdict_class(str(row['verdict']))}\">{html.escape(str(row['verdict']))}</span></td>"
            f"<td>{html.escape(str(row['risk_level']))}</td>"
            f"<td>{row['runtime_required_gib']}</td>"
            f"<td>{row['margin_gib']}</td>"
            f"<td>{row['decode_tokens_s_mid']}</td>"
            f"<td><button class=\"copy\" data-copy=\"{html.escape(str(row['deploy_command']), quote=True)}\">Copy</button></td>"
            "</tr>"
        )
    rows = "\n".join(row_chunks)
    charts = "\n".join(f'<img src="{html.escape(name)}" alt="{html.escape(name)}">' for name in chart_files)
    quant_options = "\n".join(
        f'<option value="{html.escape(str(item), quote=True)}">{html.escape(str(item))}</option>'
        for item in sorted({str(row["quantization"]) for row in compare["rows"]})
    )
    backend_options = "\n".join(
        f'<option value="{html.escape(str(item), quote=True)}">{html.escape(str(item))}</option>'
        for item in sorted({str(row["backend"]) for row in compare["rows"]})
    )
    risk_note = "No runnable candidate found. Treat every command as a dry-run starting point, not a recommendation."
    if summary["runnable_plans"]:
        risk_note = "Best candidate is the first smooth/tight plan after risk, precision, context, speed, and memory margin sorting."
    reason = (
        f"{summary['runnable_plans']} of {summary['total_plans']} candidates are runnable. "
        f"The highlighted candidate balances verdict, risk, quantization quality, context length, speed, and memory margin."
        if summary["runnable_plans"]
        else "No runnable candidate was found in this matrix. Lower quantization, shorten context, or use different hardware."
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>local-llm-lab compare</title>
  <style>
    :root {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; background: #f6f7f9; }}
    body {{ margin: 0; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 20px 56px; }}
    h1 {{ margin: 0 0 6px; font-size: 34px; letter-spacing: 0; }}
    .subtle {{ color: #5b6472; margin: 0 0 22px; }}
    .hero {{ border: 1px solid #d9dee8; background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #e1e6ef; border-radius: 8px; padding: 13px 14px; background: #fff; }}
    .metric strong {{ display: block; font-size: 22px; margin-top: 5px; }}
    .label {{ color: #667085; font-size: 12px; text-transform: uppercase; letter-spacing: 0; }}
    .badge {{ display: inline-block; border-radius: 999px; padding: 3px 10px; font-size: 12px; font-weight: 700; }}
    .ok {{ color: #065f46; background: #d1fae5; }}
    .warn {{ color: #92400e; background: #fef3c7; }}
    .bad {{ color: #991b1b; background: #fee2e2; }}
    .fail {{ color: #7f1d1d; background: #fecaca; }}
    .neutral {{ color: #334155; background: #e2e8f0; }}
    .section {{ margin-top: 20px; }}
    .notice {{ border-left: 4px solid #2563eb; background: #eff6ff; padding: 12px 14px; margin-top: 14px; color: #1e3a8a; }}
    .recommend {{ border: 1px solid #d9dee8; background: #fff; border-radius: 8px; padding: 16px; margin: 18px 0; }}
    .recommend h2 {{ margin: 0 0 8px; font-size: 18px; }}
    .recommend p {{ margin: 0 0 12px; color: #374151; line-height: 1.45; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: end; margin: 16px 0 12px; }}
    .control {{ display: grid; gap: 5px; }}
    .control label {{ color: #667085; font-size: 12px; text-transform: uppercase; }}
    select, input, button {{ border: 1px solid #cfd6e3; border-radius: 6px; background: #fff; color: #111827; font: inherit; padding: 8px 10px; }}
    button {{ cursor: pointer; }}
    button.primary {{ color: #fff; background: #1f2937; border-color: #1f2937; }}
    button.copy {{ padding: 6px 9px; font-size: 12px; }}
    code {{ display: block; white-space: pre-wrap; overflow-wrap: anywhere; background: #111827; color: #f9fafb; border-radius: 8px; padding: 12px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d9dee8; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #eef1f6; text-align: left; font-size: 14px; }}
    th {{ background: #f1f4f9; color: #344054; font-size: 12px; text-transform: uppercase; letter-spacing: 0; cursor: pointer; }}
    tr.best-row td {{ background: #f8fafc; box-shadow: inset 3px 0 0 #10b981; }}
    tr[hidden] {{ display: none; }}
    img {{ max-width: 100%; background: #fff; border: 1px solid #d9dee8; border-radius: 8px; margin: 10px 0; }}
    .count {{ color: #5b6472; margin: 8px 0; }}
  </style>
</head>
<body>
<main>
  <h1>Compare Report</h1>
  <p class="subtle">Quantization, context length, and backend tradeoffs for a local LLM run.</p>
  <section class="hero">
    <div class="grid">
      <div class="metric"><span class="label">Best backend</span><strong>{html.escape(str(best['backend']))}</strong></div>
      <div class="metric"><span class="label">Best quant</span><strong>{html.escape(str(best['quantization']))}</strong></div>
      <div class="metric"><span class="label">Context</span><strong>{best['context_tokens']}</strong></div>
      <div class="metric"><span class="label">Verdict</span><strong><span class="badge {_verdict_class(str(best['verdict']))}">{html.escape(str(best['verdict']))}</span></strong></div>
      <div class="metric"><span class="label">Margin</span><strong>{best['margin_gib']} GiB</strong></div>
      <div class="metric"><span class="label">Decode</span><strong>{best['decode_tokens_s_mid']} tok/s</strong></div>
    </div>
    <p class="notice">{html.escape(risk_note)} Verdict counts: {html.escape(json.dumps(verdict_counts, sort_keys=True))}</p>
    <code id="bestCommand">{best_command}</code>
    <button class="primary" data-copy="{best_command}">Copy best deploy dry-run</button>
  </section>
  <section class="recommend">
    <h2>Recommendation reason</h2>
    <p>{html.escape(reason)}</p>
    <code>{recommend_command}</code>
    <button class="primary" data-copy="{recommend_command}">Copy recommend command</button>
  </section>
  <section class="section">{charts}</section>
  <section class="section">
    <div class="toolbar">
      <div class="control"><label for="verdictFilter">Verdict</label><select id="verdictFilter"><option value="">All</option><option value="smooth">smooth</option><option value="tight">tight</option><option value="not-recommended">not-recommended</option><option value="does-not-fit">does-not-fit</option></select></div>
      <div class="control"><label for="quantFilter">Quant</label><select id="quantFilter"><option value="">All</option>{quant_options}</select></div>
      <div class="control"><label for="backendFilter">Backend</label><select id="backendFilter"><option value="">All</option>{backend_options}</select></div>
      <div class="control"><label for="minMargin">Min margin GiB</label><input id="minMargin" type="number" step="1" placeholder="any"></div>
      <button id="clearFilters">Clear</button>
    </div>
    <p class="count"><span id="visibleCount">{len(compare['rows'])}</span> of {len(compare['rows'])} plans visible. Click a column header to sort.</p>
    <table id="matrix">
      <thead><tr><th data-sort="backend">Backend</th><th data-sort="quant">Quant</th><th data-sort="context">Context</th><th data-sort="verdict">Verdict</th><th>Risk</th><th data-sort="runtime">Runtime GiB</th><th data-sort="margin">Margin GiB</th><th data-sort="decode">Decode tok/s</th><th>Deploy</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </section>
</main>
<script>
const rows = Array.from(document.querySelectorAll("#matrix tbody tr"));
const filters = {{
  verdict: document.querySelector("#verdictFilter"),
  quant: document.querySelector("#quantFilter"),
  backend: document.querySelector("#backendFilter"),
  minMargin: document.querySelector("#minMargin")
}};
function applyFilters() {{
  let visible = 0;
  rows.forEach(row => {{
    const ok =
      (!filters.verdict.value || row.dataset.verdict === filters.verdict.value) &&
      (!filters.quant.value || row.dataset.quant === filters.quant.value) &&
      (!filters.backend.value || row.dataset.backend === filters.backend.value) &&
      (!filters.minMargin.value || Number(row.dataset.margin) >= Number(filters.minMargin.value));
    row.hidden = !ok;
    if (ok) visible += 1;
  }});
  document.querySelector("#visibleCount").textContent = String(visible);
}}
Object.values(filters).forEach(input => input.addEventListener("input", applyFilters));
document.querySelector("#clearFilters").addEventListener("click", () => {{
  Object.values(filters).forEach(input => input.value = "");
  applyFilters();
}});
document.querySelectorAll("[data-copy]").forEach(button => {{
  button.addEventListener("click", async () => {{
    const text = button.dataset.copy || "";
    try {{
      await navigator.clipboard.writeText(text);
      const original = button.textContent;
      button.textContent = "Copied";
      setTimeout(() => button.textContent = original, 900);
    }} catch {{
      window.prompt("Copy command", text);
    }}
  }});
}});
document.querySelectorAll("th[data-sort]").forEach(header => {{
  header.addEventListener("click", () => {{
    const key = header.dataset.sort;
    const numeric = new Set(["context", "runtime", "margin", "decode"]);
    const body = document.querySelector("#matrix tbody");
    const sorted = rows.slice().sort((a, b) => {{
      const av = numeric.has(key) ? Number(a.dataset[key]) : String(a.dataset[key]);
      const bv = numeric.has(key) ? Number(b.dataset[key]) : String(b.dataset[key]);
      return av > bv ? -1 : av < bv ? 1 : 0;
    }});
    sorted.forEach(row => body.appendChild(row));
  }});
}});
</script>
</body>
</html>
"""


def write_compare_outputs(data: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "compare.json"
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    md_path = target / "compare.md"
    md_path.write_text(compare_markdown(data), encoding="utf-8")

    rows = data["compare"]["rows"]
    chart_files: list[str] = []
    decode_items = [
        (f"{row['quantization']} {row['context_tokens']}", float(row["decode_tokens_s_mid"]))
        for row in rows[:12]
    ]
    if decode_items:
        decode_svg = target / "compare_decode.svg"
        decode_svg.write_text(_svg_bar_chart(decode_items, "Estimated decode throughput", "tok/s"), encoding="utf-8")
        chart_files.append(decode_svg.name)

    margin_items = [
        (f"{row['quantization']} {row['context_tokens']}", max(float(row["margin_gib"]), 0.0))
        for row in rows[:12]
    ]
    if margin_items:
        margin_svg = target / "compare_margin.svg"
        margin_svg.write_text(_svg_bar_chart(margin_items, "Positive runtime memory margin", "GiB"), encoding="utf-8")
        chart_files.append(margin_svg.name)

    html_path = target / "index.html"
    html_path.write_text(compare_html(data, chart_files), encoding="utf-8")
    return {"out_dir": str(target), "files": [str(path) for path in sorted(target.iterdir()) if path.is_file()]}
