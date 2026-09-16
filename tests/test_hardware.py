from __future__ import annotations

import io
import unittest
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from unittest.mock import patch

from local_llm_lab.cli import main
from local_llm_lab.hardware import detect_hardware


GIB = 1024**3


class HardwareDetectionTest(unittest.TestCase):
    @contextmanager
    def _detection_environment(
        self,
        *,
        sysctl: str = "",
        vm_stat: float | None = None,
        sysconf_error: Exception | None = None,
    ):
        with ExitStack() as stack:
            stack.enter_context(patch("local_llm_lab.hardware.platform.system", return_value="Darwin"))
            stack.enter_context(patch("local_llm_lab.hardware._sysctl", return_value=sysctl))
            stack.enter_context(patch("local_llm_lab.hardware.optional_import", return_value=None))
            stack.enter_context(patch("local_llm_lab.hardware._vm_stat_available_gib", return_value=vm_stat))
            stack.enter_context(patch("local_llm_lab.hardware._gpu_info", return_value=("Unknown GPU", False, None)))
            stack.enter_context(patch("local_llm_lab.hardware._cpu_flags", return_value=(False, False)))
            if sysconf_error is not None:
                stack.enter_context(patch("local_llm_lab.hardware.os.sysconf", side_effect=sysconf_error))
            yield

    def test_darwin_missing_hw_memsize_falls_back_to_sysconf(self) -> None:
        with self._detection_environment(sysctl=""):
            profile = detect_hardware(skip_probes=True)
        self.assertGreater(profile.memory_total_gib, 0)
        self.assertLessEqual(profile.memory_available_gib, profile.memory_total_gib)
        self.assertIn("memory_available_estimated", profile.probes)

    def test_unknown_total_memory_raises_actionable_error(self) -> None:
        with self._detection_environment(sysctl="", sysconf_error=OSError("sysconf unavailable")):
            with self.assertRaisesRegex(ValueError, "total system memory"):
                detect_hardware(skip_probes=True)

    def test_available_memory_is_clamped_to_total(self) -> None:
        with self._detection_environment(sysctl=str(GIB), vm_stat=10.0):
            profile = detect_hardware(skip_probes=True)
        self.assertEqual(profile.memory_total_gib, 1.0)
        self.assertEqual(profile.memory_available_gib, 1.0)
        self.assertLessEqual(profile.memory_available_gib, profile.memory_total_gib)

    def test_cli_detects_unknown_total_memory_cleanly(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with self._detection_environment(sysctl="", sysconf_error=OSError("sysconf unavailable")):
            with self.assertRaises(SystemExit) as caught:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    main(["detect", "--json", "--skip-probes"])
        self.assertEqual(caught.exception.code, 2)
        self.assertNotIn("Traceback", stdout.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
