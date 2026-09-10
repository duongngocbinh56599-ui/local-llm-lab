from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
import unittest

from local_llm_lab.cli import main
from local_llm_lab.planner import make_plan


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


if __name__ == "__main__":
    unittest.main()
