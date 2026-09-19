from __future__ import annotations

from contextlib import redirect_stderr
from dataclasses import replace
from io import StringIO
import re
import unittest

from local_llm_lab.cli import main
from local_llm_lab.hardware import detect_hardware
from local_llm_lab.planner import make_plan
from local_llm_lab.quantization import get_quantization


def _try_quant_names(options: list[str]) -> list[str]:
    names: list[str] = []
    for option in options:
        match = re.match(r"Try (\S+) instead of", option)
        if match:
            names.append(match.group(1))
    return names


def _context_downgrades(options: list[str]) -> list[int]:
    contexts: list[int] = []
    for option in options:
        match = re.match(r"Reduce context to (\d+) tokens\.", option)
        if match:
            contexts.append(int(match.group(1)))
    return contexts


class PlannerTest(unittest.TestCase):
    def test_600b_on_128gb_does_not_fit(self) -> None:
        plan = make_plan(
            params="600B",
            quant_name="Q4_K_M",
            context_tokens=32768,
            hardware_fixture="apple-m4-max-128gb",
        )
        self.assertEqual(plan.verdict, "does-not-fit")
        self.assertEqual(plan.risk_level, "extreme")
        self.assertEqual(plan.recommended_quantization, "no-safe-local-quant")
        self.assertLess(plan.memory.margin_gib, 0)

    def test_70b_q4_on_128gb_has_positive_margin(self) -> None:
        plan = make_plan(
            model_name="llama-3.3-70b",
            quant_name="Q4_K_M",
            context_tokens=8192,
            hardware_fixture="apple-m4-max-128gb",
        )
        self.assertIn(plan.verdict, {"smooth", "tight"})
        self.assertGreater(plan.memory.margin_gib, 20)
        self.assertEqual(plan.recommended_backend, "llama.cpp")

    def test_context_increases_kv_cache(self) -> None:
        small = make_plan(
            model_name="llama-3.3-70b",
            quant_name="Q4_K_M",
            context_tokens=4096,
            hardware_fixture="apple-m4-max-128gb",
        )
        large = make_plan(
            model_name="llama-3.3-70b",
            quant_name="Q4_K_M",
            context_tokens=32768,
            hardware_fixture="apple-m4-max-128gb",
        )
        self.assertGreater(large.memory.kv_cache_gib, small.memory.kv_cache_gib)

    def test_backend_choice_mlx_for_mlx_format(self) -> None:
        plan = make_plan(
            model_name="llama-3.3-70b",
            quant_name="Q4_K_M",
            context_tokens=8192,
            model_format="mlx",
            hardware_fixture="apple-m4-max-128gb",
        )
        self.assertEqual(plan.recommended_backend, "mlx")

    def test_plan_rejects_invalid_run_and_architecture_values(self) -> None:
        base = {
            "model_name": "llama-3.3-70b",
            "quant_name": "Q4_K_M",
            "context_tokens": 8192,
            "hardware_fixture": "apple-m4-max-128gb",
        }
        invalid_values = (
            {"context_tokens": 0},
            {"concurrency": 0},
            {"kv_dtype_bytes": 0},
            {"layers": 0},
            {"heads": 0},
            {"kv_heads": 0},
            {"head_dim": 0},
            {"heads": 8, "kv_heads": 9},
        )
        for overrides in invalid_values:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    make_plan(**{**base, **overrides})

    def test_cli_rejects_nonpositive_values_without_traceback(self) -> None:
        cases = (
            (["plan", "--params", "70B", "--ctx", "0"], "argument --ctx"),
            (["compare", "--params", "70B", "--contexts", "4096,0"], "positive"),
        )
        for command, message in cases:
            with self.subTest(command=command):
                stderr = StringIO()
                with redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        main(command)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_cuda_no_fit_does_not_advise_larger_or_unfit_options(self) -> None:
        plan = make_plan(
            model_name="llama-3.3-70b",
            quant_name="Q4_K_M",
            context_tokens=8192,
            hardware_fixture="linux-rtx-4090-24gb",
        )
        self.assertEqual(plan.verdict, "does-not-fit")
        self.assertEqual(plan.recommended_quantization, "no-safe-local-quant")
        requested_bytes = get_quantization("Q4_K_M").bytes_per_param
        self.assertNotIn("Try Q8_0", plan.downgrade_options)
        for option in plan.downgrade_options:
            with self.subTest(option=option):
                for name in _try_quant_names([option]):
                    self.assertLessEqual(get_quantization(name).bytes_per_param, requested_bytes)
        context_options = _context_downgrades(plan.downgrade_options)
        for ctx in context_options:
            with self.subTest(context_tokens=ctx):
                replan = make_plan(
                    model_name="llama-3.3-70b",
                    quant_name=plan.inputs.quant.name,
                    context_tokens=ctx,
                    hardware=plan.inputs.hardware,
                )
                self.assertGreaterEqual(replan.memory.margin_gib, 0)
        self.assertEqual(context_options, [])
        self.assertNotIn("Reduce context to 4096 tokens.", plan.downgrade_options)

    def test_cuda_recommended_quantization_is_replan_verified(self) -> None:
        plan = make_plan(
            model_name="qwen2.5-32b",
            quant_name="Q4_K_M",
            context_tokens=8192,
            hardware_fixture="linux-rtx-4090-24gb",
        )
        recommended = plan.recommended_quantization
        self.assertNotEqual(recommended, "no-safe-local-quant")
        replan = make_plan(
            model_name="qwen2.5-32b",
            quant_name=recommended,
            context_tokens=8192,
            hardware_fixture="linux-rtx-4090-24gb",
        )
        self.assertGreaterEqual(replan.memory.margin_gib, 0)
        self.assertNotEqual(replan.verdict, "does-not-fit")

    def test_low_available_unified_memory_does_not_advise_larger_quant(self) -> None:
        hardware = replace(
            detect_hardware(skip_probes=True, fixture="apple-m4-max-128gb"),
            memory_available_gib=68.3,
        )
        plan = make_plan(
            params="200B",
            quant_name="IQ2_XS",
            context_tokens=8192,
            hardware=hardware,
        )
        self.assertIn(plan.verdict, {"not-recommended", "does-not-fit"})
        self.assertEqual(plan.recommended_quantization, "no-safe-local-quant")
        requested_bytes = get_quantization("IQ2_XS").bytes_per_param
        for name in _try_quant_names(plan.downgrade_options):
            with self.subTest(quant=name):
                self.assertLessEqual(get_quantization(name).bytes_per_param, requested_bytes)
        context_options = _context_downgrades(plan.downgrade_options)
        self.assertIn(4096, context_options)
        for ctx in context_options:
            with self.subTest(context_tokens=ctx):
                replan = make_plan(
                    params="200B",
                    quant_name=plan.inputs.quant.name,
                    context_tokens=ctx,
                    hardware=hardware,
                )
                self.assertGreaterEqual(replan.memory.margin_gib, 0)

    def test_high_availability_mac_recommendation_is_unchanged(self) -> None:
        plan = make_plan(
            model_name="llama-3.3-70b",
            quant_name="Q4_K_M",
            context_tokens=8192,
            hardware_fixture="apple-m4-max-128gb",
        )
        self.assertEqual(plan.verdict, "smooth")
        self.assertEqual(plan.recommended_quantization, "Q8_0")

    def test_recommended_quantization_always_has_nonnegative_margin(self) -> None:
        fixtures = ("apple-m4-max-128gb", "linux-rtx-4090-24gb", "linux-dual-h100-160gb")
        models = ("llama-3.1-8b", "llama-3.3-70b")
        for fixture in fixtures:
            for model in models:
                with self.subTest(fixture=fixture, model=model):
                    plan = make_plan(
                        model_name=model,
                        quant_name="Q4_K_M",
                        context_tokens=8192,
                        hardware_fixture=fixture,
                    )
                    if plan.recommended_quantization == "no-safe-local-quant":
                        continue
                    replan = make_plan(
                        model_name=model,
                        quant_name=plan.recommended_quantization,
                        context_tokens=8192,
                        hardware_fixture=fixture,
                    )
                    self.assertGreaterEqual(replan.memory.margin_gib, 0)


if __name__ == "__main__":
    unittest.main()
