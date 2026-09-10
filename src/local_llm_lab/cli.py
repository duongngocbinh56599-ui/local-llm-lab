from __future__ import annotations

import argparse
import json
import sys
from math import isfinite
from pathlib import Path

from .bench import mock_benchmark, save_bench, tiny_local_benchmark
from .compare import (
    DEFAULT_COMPARE_BACKENDS,
    DEFAULT_COMPARE_CONTEXTS,
    DEFAULT_COMPARE_QUANTS,
    CompareRequest,
    compare_plans,
    parse_csv,
    parse_int_csv,
    write_compare_outputs,
)
from .deploy import generate_deploy_files
from .fixtures import list_fixture_names
from .hardware import detect_hardware
from .models import list_presets
from .planner import make_plan
from .profiles import list_profiles, load_profile, load_profile_document, save_profile
from .recommend import RecommendRequest, recommend_plans, write_recommend_outputs
from .render import print_json, print_plan
from .report import generate_report
from .server import serve_directory
from .stress import mock_stress, save_stress
from .units import parse_params


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a finite value greater than 0") from None
    if not isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite value greater than 0")
    return parsed


def _parameter_count(value: str) -> str:
    try:
        parse_params(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None
    return value


def _add_plan_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--model", help="Curated model preset, e.g. llama-3.3-70b")
    group.add_argument("--params", type=_parameter_count, help="Free-form parameter count, e.g. 70B, 120B, 600B")
    parser.add_argument("--quant", default="Q4_K_M", help="Quantization, e.g. Q4_K_M, Q5_K_M, IQ2_XS")
    parser.add_argument("--ctx", type=_positive_int, default=8192, help="Context length in tokens")
    parser.add_argument("--concurrency", type=_positive_int, default=1, help="Concurrent sequences/requests")
    parser.add_argument("--backend", default="auto", help="auto, llama.cpp, mlx, ollama, vllm")
    parser.add_argument("--format", default=None, help="gguf, mlx, safetensors, hf")
    parser.add_argument("--hardware", default=None, help="Use fixture:name or profile:name")
    parser.add_argument("--layers", type=_positive_int, default=None)
    parser.add_argument("--heads", type=_positive_int, default=None)
    parser.add_argument("--kv-heads", type=_positive_int, default=None)
    parser.add_argument("--head-dim", type=_positive_int, default=None)
    parser.add_argument("--kv-dtype-bytes", type=_positive_float, default=2.0)


def _resolve_hardware_ref(value: str | None):
    hardware_fixture = None
    hardware_profile = None
    hardware_label = "local"
    if value:
        if value.startswith("profile:"):
            name = value.split(":", 1)[1]
            hardware_profile = load_profile(name)
            hardware_label = f"profile:{name}"
        else:
            hardware_fixture = value.removeprefix("fixture:")
            hardware_label = f"fixture:{hardware_fixture}"
    return hardware_fixture, hardware_profile, hardware_label


def _plan_from_args(args: argparse.Namespace):
    if not args.model and not args.params:
        raise SystemExit("Either --model or --params is required.")
    hardware_fixture, hardware_profile, _hardware_label = _resolve_hardware_ref(args.hardware)
    return make_plan(
        model_name=args.model,
        params=args.params,
        quant_name=args.quant,
        context_tokens=args.ctx,
        concurrency=args.concurrency,
        backend=args.backend,
        model_format=args.format,
        hardware_fixture=hardware_fixture,
        hardware=hardware_profile,
        layers=args.layers,
        heads=args.heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        kv_dtype_bytes=args.kv_dtype_bytes,
    )


def cmd_detect(args: argparse.Namespace) -> int:
    fixture, hardware_profile, hardware_label = _resolve_hardware_ref(args.hardware)
    profile = hardware_profile or detect_hardware(skip_probes=args.skip_probes, fixture=fixture)
    saved_path = None
    if args.save_profile:
        saved_path = save_profile(args.save_profile, profile, source=hardware_label)
    if args.json:
        print_json(profile.to_dict())
        if saved_path:
            print(f"Saved profile: {saved_path}", file=sys.stderr)
    else:
        print(f"Hardware: {profile.name}")
        print(f"CPU: {profile.cpu}")
        print(f"GPU: {profile.gpu}")
        print(f"Memory: {profile.memory_total_gib} GiB total, {profile.memory_available_gib} GiB available")
        print(f"Disk available: {profile.disk_available_gib} GiB")
        print(f"Capabilities: Metal={profile.metal} CUDA={profile.cuda} AVX={profile.avx} NEON={profile.neon}")
        if profile.backends:
            found = {key: value for key, value in profile.backends.items() if value}
            print(f"Backends found: {found if found else 'none'}")
        if profile.probes:
            print(f"Probes: {profile.probes}")
        if profile.note:
            print(f"Note: {profile.note}")
        if saved_path:
            print(f"Saved profile: {saved_path}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    plan = _plan_from_args(args)
    if args.output:
        Path(args.output).write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print_json(plan.to_dict())
    else:
        print_plan(plan)
    return 0


def cmd_deploy(args: argparse.Namespace) -> int:
    plan = _plan_from_args(args)
    out = args.out or Path(".local-llm-lab") / "deploy" / f"{plan.inputs.model.id}-{plan.inputs.quant.name.lower()}"
    result = generate_deploy_files(plan, out, model_path=args.model_path, dry_run=not args.run)
    if args.json:
        print_json(result.to_dict())
    else:
        print(f"Generated deploy dry-run in {result.out_dir}")
        for item in result.files:
            print(f"- {item}")
        for warning in result.warnings:
            print(f"Warning: {warning}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    plan = None
    if args.model or args.params:
        plan = _plan_from_args(args)
    result = mock_benchmark(plan, seed=args.seed) if args.mock else tiny_local_benchmark()
    if args.output:
        save_bench(result, args.output)
    if args.json:
        print_json(result.to_dict())
    else:
        print(f"Benchmark: {result.kind} backend={result.backend} model={result.model}")
        for row in result.scenarios:
            print(f"- {row['name']}: decode={row['decode_tokens_s']} tok/s p95={row['p95_latency_s']}s")
        for note in result.notes:
            print(f"Note: {note}")
    return 0


def cmd_stress(args: argparse.Namespace) -> int:
    plan = None
    if args.model or args.params:
        plan = _plan_from_args(args)
    result = mock_stress(plan, seed=args.seed)
    if args.output:
        save_stress(result, args.output)
    if args.json:
        print_json(result.to_dict())
    else:
        print(f"Stress: {result.kind}")
        for row in result.scenarios:
            print(
                f"- {row['name']}: drop={row['throughput_drop_pct']}% "
                f"pressure={row['memory_pressure_ratio']} stability={row['stability']}"
            )
        for note in result.notes:
            print(f"Note: {note}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    result = generate_report(args.input, args.out)
    if args.json:
        print_json(result)
    else:
        print(f"Report generated in {result['out_dir']}")
        for item in result["files"]:
            print(f"- {item}")
    if args.serve:
        serve_directory(result["out_dir"], port=args.port)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    if not args.model and not args.params:
        raise SystemExit("Either --model or --params is required.")
    hardware_fixture, hardware_profile, hardware_label = _resolve_hardware_ref(args.hardware)
    request = CompareRequest(
        model_name=args.model,
        params=args.params,
        quantizations=parse_csv(args.quants, DEFAULT_COMPARE_QUANTS),
        contexts=parse_int_csv(args.contexts, DEFAULT_COMPARE_CONTEXTS),
        backends=parse_csv(args.backends, DEFAULT_COMPARE_BACKENDS),
        concurrency=args.concurrency,
        model_format=args.format,
        hardware_fixture=hardware_fixture,
        hardware_profile=hardware_profile,
        hardware_label=hardware_label,
        layers=args.layers,
        heads=args.heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        kv_dtype_bytes=args.kv_dtype_bytes,
    )
    result = compare_plans(request)
    if args.out:
        written = write_compare_outputs(result, args.out)
    else:
        written = None

    if args.json:
        print_json(result if not written else {"compare": result["compare"], "outputs": written})
    else:
        summary = result["compare"]["summary"]
        best = summary["best"]
        print("Best candidate:")
        print(f"- backend: {best['backend']}")
        print(f"- quantization: {best['quantization']}")
        print(f"- context: {best['context_tokens']} tokens")
        print(f"- verdict: {best['verdict']} risk={best['risk_level']}")
        print(f"- margin: {best['margin_gib']} GiB")
        print(f"- estimated decode: {best['decode_tokens_s_mid']} tok/s")
        print(f"Compared {summary['total_plans']} plans; runnable={summary['runnable_plans']}.")
        if written:
            print(f"Compare report generated in {written['out_dir']}")
    return 0


def _compare_request_from_args(args: argparse.Namespace) -> CompareRequest:
    if not args.model and not args.params:
        raise SystemExit("Either --model or --params is required.")
    hardware_fixture, hardware_profile, hardware_label = _resolve_hardware_ref(args.hardware)
    return CompareRequest(
        model_name=args.model,
        params=args.params,
        quantizations=parse_csv(args.quants, DEFAULT_COMPARE_QUANTS),
        contexts=parse_int_csv(args.contexts, DEFAULT_COMPARE_CONTEXTS),
        backends=parse_csv(args.backends, DEFAULT_COMPARE_BACKENDS),
        concurrency=args.concurrency,
        model_format=args.format,
        hardware_fixture=hardware_fixture,
        hardware_profile=hardware_profile,
        hardware_label=hardware_label,
        layers=args.layers,
        heads=args.heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        kv_dtype_bytes=args.kv_dtype_bytes,
    )


def cmd_recommend(args: argparse.Namespace) -> int:
    request = RecommendRequest(compare_request=_compare_request_from_args(args), target=args.target)
    result = recommend_plans(request)
    written = write_recommend_outputs(result, args.out) if args.out else None
    if args.json:
        print_json(result if not written else {"recommendation": result["recommendation"], "outputs": written})
    else:
        rec = result["recommendation"]
        best = rec["best"]
        print(f"Recommendation: {rec['status']} target={rec['target']}")
        print(f"- backend: {best['backend']}")
        print(f"- quantization: {best['quantization']}")
        print(f"- context: {best['context_tokens']} tokens")
        print(f"- verdict: {best['verdict']} risk={best['risk_level']}")
        print(f"- margin: {best['margin_gib']} GiB")
        print(f"- estimated decode: {best['decode_tokens_s_mid']} tok/s")
        print("Why:")
        for item in rec["why"]:
            print(f"- {item}")
        print("Next steps:")
        for item in rec["next_steps"]:
            print(f"- {item}")
        print("Deploy dry-run:")
        print(rec["deploy_command"])
        if written:
            print(f"Recommendation written in {written['out_dir']}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = make_plan(
        params="600B",
        quant_name="Q4_K_M",
        context_tokens=32768,
        hardware_fixture="apple-m4-max-128gb",
    )
    bench = mock_benchmark(plan)
    stress = mock_stress(plan, bench)
    data = {"plan": plan.to_dict(), "bench": bench.to_dict(), "stress": stress.to_dict()}
    demo_json = out_dir / "demo-run.json"
    demo_json.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    report_dir = out_dir / "report"
    generate_report(demo_json, report_dir)
    if args.json:
        print_json({"demo_json": str(demo_json), "report_dir": str(report_dir)})
    else:
        print(f"Demo data: {demo_json}")
        print(f"Demo report: {report_dir / 'index.html'}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    if args.what == "models":
        for model in list_presets():
            print(f"{model.id}\t{model.family}\t{model.params_b}B\t{model.confidence}")
    elif args.what == "hardware":
        for name in list_fixture_names():
            print(name)
    else:
        profiles = list_profiles()
        if not profiles:
            print("(no profiles saved)")
        for profile in profiles:
            memory = profile["memory_total_gib"]
            memory_text = f"{memory} GiB" if memory is not None else "unknown memory"
            print(f"{profile['name']}\t{profile['hardware_name']}\t{memory_text}")
    return 0


def cmd_profile_show(args: argparse.Namespace) -> int:
    document = load_profile_document(args.name)
    if args.json:
        print_json(document)
    else:
        hardware = document["hardware"]
        print(f"Profile: {document['profile_name']}")
        print(f"Created: {document['created_at']}")
        print(f"Source: {document['source']}")
        print(f"Hardware: {hardware['name']}")
        print(f"Memory: {hardware['memory_total_gib']} GiB total, {hardware['memory_available_gib']} GiB available")
        print(f"GPU: {hardware['gpu']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="local-llm-lab", description="Honest local LLM planning and benchmarking.")
    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect", help="Detect local hardware and backend capabilities.")
    detect.add_argument("--json", action="store_true")
    detect.add_argument("--skip-probes", action="store_true")
    detect.add_argument("--hardware", default=None, help="Use fixture:name or profile:name instead of local detection.")
    detect.add_argument("--save-profile", help="Save a sanitized hardware profile for later use.")
    detect.set_defaults(func=cmd_detect)

    plan = sub.add_parser("plan", help="Estimate whether a model can run locally.")
    _add_plan_args(plan)
    plan.add_argument("--json", action="store_true")
    plan.add_argument("--output")
    plan.set_defaults(func=cmd_plan)

    deploy = sub.add_parser("deploy", help="Generate dry-run deployment scripts/configs.")
    _add_plan_args(deploy)
    deploy.add_argument("--out")
    deploy.add_argument("--model-path", default="./model.gguf")
    deploy.add_argument("--run", action="store_true", help="Mark as non-dry-run; still does not download models.")
    deploy.add_argument("--json", action="store_true")
    deploy.set_defaults(func=cmd_deploy)

    bench = sub.add_parser("bench", help="Run mock or tiny local benchmark.")
    _add_plan_args(bench)
    bench.add_argument("--mock", action="store_true", default=False)
    bench.add_argument("--seed", default="local-llm-lab")
    bench.add_argument("--output")
    bench.add_argument("--json", action="store_true")
    bench.set_defaults(func=cmd_bench)

    stress = sub.add_parser("stress", help="Simulate LLM plus GPU/Blender-style resource competition.")
    _add_plan_args(stress)
    stress.add_argument("--mock", action="store_true", default=True)
    stress.add_argument("--seed", default="stress")
    stress.add_argument("--output")
    stress.add_argument("--json", action="store_true")
    stress.set_defaults(func=cmd_stress)

    report = sub.add_parser("report", help="Generate Markdown, JSON, SVG, and HTML report.")
    report.add_argument("--input", required=True)
    report.add_argument("--out", required=True)
    report.add_argument("--json", action="store_true")
    report.add_argument("--serve", action="store_true")
    report.add_argument("--port", type=int, default=8787)
    report.set_defaults(func=cmd_report)

    compare = sub.add_parser("compare", help="Compare quantization/context/backend planning tradeoffs.")
    group = compare.add_mutually_exclusive_group(required=False)
    group.add_argument("--model", help="Curated model preset, e.g. llama-3.3-70b")
    group.add_argument("--params", type=_parameter_count, help="Free-form parameter count, e.g. 70B, 120B, 600B")
    compare.add_argument("--quants", default=None, help="Comma-separated quantizations. Default: Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,IQ2_XS")
    compare.add_argument("--contexts", default=None, help="Comma-separated context lengths. Default: 4096,8192,16384,32768")
    compare.add_argument("--backends", default=None, help="Comma-separated backends. Default: auto")
    compare.add_argument("--concurrency", type=_positive_int, default=1)
    compare.add_argument("--format", default=None)
    compare.add_argument("--hardware", default=None, help="Use fixture:name or profile:name")
    compare.add_argument("--layers", type=_positive_int, default=None)
    compare.add_argument("--heads", type=_positive_int, default=None)
    compare.add_argument("--kv-heads", type=_positive_int, default=None)
    compare.add_argument("--head-dim", type=_positive_int, default=None)
    compare.add_argument("--kv-dtype-bytes", type=_positive_float, default=2.0)
    compare.add_argument("--out", help="Write compare.json, compare.md, charts, and index.html")
    compare.add_argument("--json", action="store_true")
    compare.set_defaults(func=cmd_compare)

    recommend = sub.add_parser("recommend", help="Pick the best local run plan from a scanned matrix.")
    group = recommend.add_mutually_exclusive_group(required=False)
    group.add_argument("--model", help="Curated model preset, e.g. llama-3.3-70b")
    group.add_argument("--params", type=_parameter_count, help="Free-form parameter count, e.g. 70B, 120B, 600B")
    recommend.add_argument("--quants", default=None, help="Comma-separated quantizations. Default: Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,IQ2_XS")
    recommend.add_argument("--contexts", default=None, help="Comma-separated context lengths. Default: 4096,8192,16384,32768")
    recommend.add_argument("--backends", default=None, help="Comma-separated backends. Default: auto")
    recommend.add_argument("--target", choices=["smooth", "tight"], default="tight", help="Required comfort target. Default: tight")
    recommend.add_argument("--concurrency", type=_positive_int, default=1)
    recommend.add_argument("--format", default=None)
    recommend.add_argument("--hardware", default=None, help="Use fixture:name or profile:name")
    recommend.add_argument("--layers", type=_positive_int, default=None)
    recommend.add_argument("--heads", type=_positive_int, default=None)
    recommend.add_argument("--kv-heads", type=_positive_int, default=None)
    recommend.add_argument("--head-dim", type=_positive_int, default=None)
    recommend.add_argument("--kv-dtype-bytes", type=_positive_float, default=2.0)
    recommend.add_argument("--out", help="Write recommend.json and recommend.md into this directory")
    recommend.add_argument("--json", action="store_true")
    recommend.set_defaults(func=cmd_recommend)

    serve = sub.add_parser("serve", help="Serve a generated report directory on localhost.")
    serve.add_argument("--dir", required=True)
    serve.add_argument("--port", type=int, default=8787)
    serve.set_defaults(func=lambda args: serve_directory(args.dir, port=args.port) or 0)

    demo = sub.add_parser("demo", help="Create reproducible demo data and report.")
    demo.add_argument("--out", default="examples")
    demo.add_argument("--json", action="store_true")
    demo.set_defaults(func=cmd_demo)

    profile = sub.add_parser("profile", help="Inspect saved measured hardware profiles.")
    profile_sub = profile.add_subparsers(dest="profile_command", required=True)
    profile_show = profile_sub.add_parser("show", help="Show one saved hardware profile.")
    profile_show.add_argument("name")
    profile_show.add_argument("--json", action="store_true")
    profile_show.set_defaults(func=cmd_profile_show)

    listing = sub.add_parser("list", help="List model presets, hardware fixtures, or saved profiles.")
    listing.add_argument("what", choices=["models", "hardware", "profiles"])
    listing.set_defaults(func=cmd_list)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
