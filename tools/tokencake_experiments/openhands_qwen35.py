# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch comparable OpenHands/Qwen3.5 servers without importing vLLM.

Qwen3.5 needs aligned Mamba prefix caching, not the unsupported ``all``
mode. Native offloading supports mixed attention/Mamba groups in this checkout;
real-model offload correctness and performance still require GPU validation.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODES = ("base", "agent", "offload", "agent_offload")
OFFLOAD_MODES = ("offload", "agent_offload")
SAMPLING_DEFAULTS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "repetition_penalty": 1.0,
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="agent_offload")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--served-model-name", default="Qwen3.5-27B")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=positive_int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=positive_int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=positive_int, default=1)
    parser.add_argument("--max-model-len", type=positive_int, default=32768)
    parser.add_argument("--max-num-seqs", type=positive_int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=positive_int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=positive_float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tool-call-parser", default="qwen3_coder")
    parser.add_argument("--reasoning-parser", default="qwen3")
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="default chat-template thinking mode (default: disabled)",
    )
    parser.add_argument(
        "--kv-offloading-size",
        type=positive_float,
        help="CPU KV capacity in GiB; offload modes only (default: 32)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print a JSON launch plan without loading vLLM",
    )
    args = parser.parse_args(argv)
    if args.port > 65535:
        parser.error("--port must not exceed 65535")
    if args.gpu_memory_utilization > 1:
        parser.error("--gpu-memory-utilization must not exceed 1")
    if args.mode not in OFFLOAD_MODES and args.kv_offloading_size is not None:
        parser.error("--kv-offloading-size is only valid in offload modes")
    return args


def launch_plan(args: argparse.Namespace) -> dict:
    command = [
        args.python,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
    ]
    for name in (
        "served_model_name",
        "host",
        "port",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "gpu_memory_utilization",
        "seed",
        "tool_call_parser",
        "reasoning_parser",
    ):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    command.extend(
        [
            "--data-parallel-size",
            "1",
            "--enable-auto-tool-choice",
            "--language-model-only",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--mamba-cache-mode",
            "align",
            "--no-disable-hybrid-kv-cache-manager",
            "--generation-config",
            "vllm",
            "--override-generation-config",
            json.dumps(SAMPLING_DEFAULTS, sort_keys=True),
            "--default-chat-template-kwargs",
            json.dumps({"enable_thinking": args.enable_thinking}),
        ]
    )
    if args.mode != "base":
        settings = {
            "tokencake": {
                "scheduling": {"enabled": args.mode in ("agent", "agent_offload")},
                "offload": {"enabled": args.mode in OFFLOAD_MODES},
            }
        }
        command.extend(["--additional-config", json.dumps(settings, sort_keys=True)])
    if args.mode in OFFLOAD_MODES:
        command.extend(
            [
                "--kv-offloading-size",
                str(args.kv_offloading_size or 32.0),
                "--kv-offloading-backend",
                "native",
            ]
        )
    return {
        "mode": args.mode,
        "cwd": str(ROOT),
        "env": {"VLLM_USE_SIMPLE_KV_OFFLOAD": "0"},
        "command": command,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plan = launch_plan(args)
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    os.chdir(plan["cwd"])
    os.execvpe(plan["command"][0], plan["command"], os.environ | plan["env"])


if __name__ == "__main__":
    main()
