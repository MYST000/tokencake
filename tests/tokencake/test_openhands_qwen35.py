# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only launcher contracts; no vLLM/GPU imports are needed."""

import contextlib
import io
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from tools.tokencake_experiments import openhands_qwen35 as launcher


def options(command):
    result = {}
    index = 5
    while index < len(command):
        name = command[index]
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            result[name] = command[index + 1]
            index += 2
        else:
            result[name] = True
            index += 1
    return result


class TestOpenHandsQwen35Launcher(unittest.TestCase):
    def test_modes_change_only_tokencake_controls(self):
        baseline = None
        for mode in launcher.MODES:
            with self.subTest(mode=mode):
                plan = launcher.launch_plan(launcher.parse_args(["--mode", mode]))
                flags = options(plan["command"])
                settings = flags.pop("--additional-config", None)
                cpu_size = flags.pop("--kv-offloading-size", None)
                backend = flags.pop("--kv-offloading-backend", None)
                if mode == "base":
                    self.assertIsNone(settings)
                else:
                    config = json.loads(settings)["tokencake"]
                    self.assertEqual(
                        config["scheduling"]["enabled"],
                        mode in ("agent", "agent_offload"),
                    )
                    self.assertEqual(
                        config["offload"]["enabled"], mode in launcher.OFFLOAD_MODES
                    )
                if mode in launcher.OFFLOAD_MODES:
                    self.assertEqual(float(cpu_size), 32.0)
                    self.assertEqual(backend, "native")
                else:
                    self.assertIsNone(cpu_size)
                    self.assertIsNone(backend)
                if baseline is None:
                    baseline = flags
                self.assertEqual(flags, baseline)
                self.assertEqual(plan["env"], {"VLLM_USE_SIMPLE_KV_OFFLOAD": "0"})
                self.assertEqual(plan["command"][4], "Qwen/Qwen3.5-27B")
        assert baseline is not None
        self.assertEqual(baseline["--mamba-cache-mode"], "align")
        self.assertTrue(baseline["--no-disable-hybrid-kv-cache-manager"])
        self.assertTrue(baseline["--enable-prefix-caching"])
        self.assertTrue(baseline["--enable-chunked-prefill"])
        self.assertEqual(baseline["--generation-config"], "vllm")
        self.assertEqual(
            json.loads(baseline["--default-chat-template-kwargs"]),
            {"enable_thinking": False},
        )

    def test_custom_paths_are_not_interpreted_as_shell(self):
        model = "/models/Qwen 3.5 $(untouched)"
        args = launcher.parse_args(
            [
                "--model",
                model,
                "--python",
                "/custom env/.venv/bin/python",
                "--mode",
                "offload",
                "--kv-offloading-size",
                "12.5",
                "--tensor-parallel-size",
                "2",
                "--tool-call-parser",
                "qwen3_xml",
                "--enable-thinking",
            ]
        )
        plan = launcher.launch_plan(args)
        self.assertEqual(plan["command"][0], "/custom env/.venv/bin/python")
        self.assertEqual(plan["command"][4], model)
        flags = options(plan["command"])
        self.assertEqual(flags["--kv-offloading-size"], "12.5")
        self.assertEqual(flags["--tensor-parallel-size"], "2")
        self.assertEqual(flags["--tool-call-parser"], "qwen3_xml")
        self.assertTrue(
            json.loads(flags["--default-chat-template-kwargs"])["enable_thinking"]
        )

    def test_invalid_capacity_and_disabled_offload_are_rejected(self):
        cases = [
            ["--mode", "base", "--kv-offloading-size", "1"],
            ["--mode", "agent", "--kv-offloading-size", "1"],
            ["--kv-offloading-size", "0"],
            ["--kv-offloading-size", "nan"],
            ["--gpu-memory-utilization", "inf"],
            ["--gpu-memory-utilization", "1.1"],
            ["--tensor-parallel-size", "0"],
            ["--port", "65536"],
        ]
        for argv in cases:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    launcher.parse_args(argv)
                self.assertEqual(error.exception.code, 2)

    def test_dry_run_works_without_site_packages(self):
        result = subprocess.run(
            [sys.executable, "-S", launcher.__file__, "--dry-run"],
            check=True,
            capture_output=True,
            text=True,
            cwd="/tmp",
        )
        plan = json.loads(result.stdout)
        self.assertEqual(plan["mode"], "agent_offload")
        self.assertEqual(plan["cwd"], str(launcher.ROOT))

    def test_execution_preserves_environment_and_sets_source_directory(self):
        with (
            patch.object(launcher.os, "chdir") as chdir,
            patch.object(launcher.os, "execvpe") as execute,
            patch.dict(launcher.os.environ, {"VLLM_API_KEY": "test-key"}),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            launcher.main(["--mode", "base"])
        chdir.assert_called_once_with(str(launcher.ROOT))
        _, command, environment = execute.call_args.args
        self.assertEqual(command[1:4], ["-m", "vllm.entrypoints.cli.main", "serve"])
        self.assertEqual(environment["VLLM_API_KEY"], "test-key")
        self.assertEqual(environment["VLLM_USE_SIMPLE_KV_OFFLOAD"], "0")
        self.assertNotIn("test-key", output.getvalue())


if __name__ == "__main__":
    unittest.main()
