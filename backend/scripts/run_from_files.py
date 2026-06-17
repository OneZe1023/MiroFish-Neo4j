"""
Run the MiroFish pipeline from a seed document and a prompt file.

This script is intentionally standalone: it reuses the existing backend API
through Flask's in-process test client, so it does not change or require the
frontend/backend architecture.

Default input files:
  <project root>\\reality_seed\\Ghana vs. Panama.txt
  <project root>\\reality_seed\\prompt_v2.txt

Usage:
  uv run python scripts/run_from_files.py
  uv run python scripts/run_from_files.py --max-rounds 10  # optional truncation
  uv run python scripts/run_from_files.py --seed path\\seed.txt --prompt path\\prompt.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))


DEFAULT_SEED_PATH = PROJECT_ROOT / "reality_seed" / "Ghana vs. Panama.txt"
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "reality_seed" / "prompt_v2.txt"


TERMINAL_TASK_STATUSES = {"completed", "failed"}
TERMINAL_RUN_STATUSES = {"completed", "failed", "stopped"}


class PipelineError(RuntimeError):
    """Raised when a backend API call or pipeline stage fails."""


def load_backend_app():
    from app import create_app
    from app.config import Config

    errors = Config.validate()
    if errors:
        details = "\n".join(f"  - {item}" for item in errors)
        raise PipelineError(f"配置错误:\n{details}\n请检查项目根目录 .env")

    app = create_app()
    app.config["TESTING"] = True
    return app


def read_text(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise PipelineError(f"文件不存在: {file_path}")
    if not file_path.is_file():
        raise PipelineError(f"路径不是文件: {file_path}")
    return file_path.read_text(encoding="utf-8").strip()


def require_success(response, label: str) -> dict[str, Any]:
    try:
        payload = response.get_json()
    except Exception as exc:
        raise PipelineError(f"{label} 返回不是 JSON: HTTP {response.status_code}") from exc

    if response.status_code >= 400 or not payload or not payload.get("success"):
        error = payload.get("error") if isinstance(payload, dict) else response.get_data(as_text=True)
        traceback_text = payload.get("traceback") if isinstance(payload, dict) else None
        message = f"{label} 失败: HTTP {response.status_code}: {error}"
        if traceback_text:
            message += f"\n{traceback_text}"
        raise PipelineError(message)

    return payload


def print_stage(message: str) -> None:
    print(f"\n== {message} ==")


def poll_task(
    client,
    *,
    task_id: str,
    status_endpoint: str,
    status_method: str,
    label: str,
    timeout_seconds: int,
    poll_interval: float,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.time()
    last_line = ""

    while True:
        if status_method == "GET":
            response = client.get(status_endpoint)
        else:
            payload = {"task_id": task_id}
            if extra_payload:
                payload.update(extra_payload)
            response = client.post(status_endpoint, json=payload)

        data = require_success(response, f"查询{label}状态").get("data", {})
        status = data.get("status")
        progress = data.get("progress", 0)
        message = data.get("message", "")
        line = f"{label}: status={status}, progress={progress}%, {message}"
        if line != last_line:
            print(line)
            last_line = line

        if status in TERMINAL_TASK_STATUSES or status == "ready":
            if status == "failed":
                raise PipelineError(f"{label}失败: {data.get('error') or message}")
            return data

        if time.time() - started > timeout_seconds:
            raise PipelineError(f"{label}超时，task_id={task_id}")

        time.sleep(poll_interval)


def poll_run(
    client,
    *,
    simulation_id: str,
    timeout_seconds: int,
    poll_interval: float,
) -> dict[str, Any]:
    started = time.time()
    last_line = ""

    while True:
        response = client.get(f"/api/simulation/{simulation_id}/run-status")
        data = require_success(response, "查询模拟运行状态").get("data", {})

        status = data.get("runner_status")
        progress = data.get("progress_percent", 0)
        current_round = data.get("current_round", 0)
        total_rounds = data.get("total_rounds", 0)
        actions = data.get("total_actions_count", 0)
        line = (
            "模拟运行: "
            f"status={status}, progress={progress}%, "
            f"round={current_round}/{total_rounds}, actions={actions}"
        )
        if line != last_line:
            print(line)
            last_line = line

        if status in TERMINAL_RUN_STATUSES:
            if status == "failed":
                raise PipelineError(f"模拟运行失败: {data.get('error') or data}")
            return data

        if time.time() - started > timeout_seconds:
            raise PipelineError(f"模拟运行超时，simulation_id={simulation_id}")

        time.sleep(poll_interval)


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    seed_path = Path(args.seed).resolve()
    prompt_path = Path(args.prompt).resolve()
    prompt_text = read_text(str(prompt_path))

    app = load_backend_app()
    client = app.test_client()

    project_name = args.project_name or f"CLI {seed_path.stem}"
    summary: dict[str, Any] = {
        "seed_path": str(seed_path),
        "prompt_path": str(prompt_path),
        "project_name": project_name,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print_stage("1. 生成本体并创建项目")
    with seed_path.open("rb") as seed_file:
        response = client.post(
            "/api/graph/ontology/generate",
            data={
                "files": (seed_file, seed_path.name),
                "simulation_requirement": prompt_text,
                "project_name": project_name,
                "additional_context": args.additional_context or "",
            },
            content_type="multipart/form-data",
        )
    ontology_payload = require_success(response, "生成本体")
    ontology_data = ontology_payload["data"]
    project_id = ontology_data["project_id"]
    summary["project_id"] = project_id
    summary["ontology"] = ontology_data.get("ontology")
    print(f"project_id: {project_id}")
    print(f"total_text_length: {ontology_data.get('total_text_length')}")

    print_stage("2. 构建图谱")
    response = client.post(
        "/api/graph/build",
        json={
            "project_id": project_id,
            "graph_name": args.graph_name or project_name,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
        },
    )
    build_data = require_success(response, "启动图谱构建")["data"]
    build_task_id = build_data["task_id"]
    summary["graph_build_task_id"] = build_task_id
    build_status = poll_task(
        client,
        task_id=build_task_id,
        status_endpoint=f"/api/graph/task/{build_task_id}",
        status_method="GET",
        label="图谱构建",
        timeout_seconds=args.graph_timeout,
        poll_interval=args.poll_interval,
    )
    graph_result = build_status.get("result") or {}
    graph_id = graph_result.get("graph_id")
    if not graph_id:
        project_response = client.get(f"/api/graph/project/{project_id}")
        graph_id = require_success(project_response, "读取项目信息")["data"].get("graph_id")
    if not graph_id:
        raise PipelineError("图谱构建完成，但未找到 graph_id")
    summary["graph_id"] = graph_id
    summary["graph_result"] = graph_result
    print(f"graph_id: {graph_id}")

    print_stage("3. 创建模拟")
    response = client.post(
        "/api/simulation/create",
        json={
            "project_id": project_id,
            "graph_id": graph_id,
            "enable_twitter": args.enable_twitter,
            "enable_reddit": args.enable_reddit,
        },
    )
    simulation_data = require_success(response, "创建模拟")["data"]
    simulation_id = simulation_data["simulation_id"]
    summary["simulation_id"] = simulation_id
    print(f"simulation_id: {simulation_id}")

    print_stage("4. 准备模拟环境")
    response = client.post(
        "/api/simulation/prepare",
        json={
            "simulation_id": simulation_id,
            "use_llm_for_profiles": not args.no_llm_profiles,
            "parallel_profile_count": args.parallel_profile_count,
            "force_regenerate": args.force_prepare,
        },
    )
    prepare_data = require_success(response, "启动模拟准备")["data"]
    prepare_task_id = prepare_data.get("task_id")
    summary["prepare_task_id"] = prepare_task_id
    if prepare_data.get("already_prepared"):
        print("模拟环境已准备完成，跳过重复生成")
    else:
        if not prepare_task_id:
            raise PipelineError(f"模拟准备未返回 task_id: {prepare_data}")
        poll_task(
            client,
            task_id=prepare_task_id,
            status_endpoint="/api/simulation/prepare/status",
            status_method="POST",
            label="模拟准备",
            timeout_seconds=args.prepare_timeout,
            poll_interval=args.poll_interval,
            extra_payload={"simulation_id": simulation_id},
        )

    print_stage("5. 运行模拟")
    response = client.post(
        "/api/simulation/start",
        json={
            "simulation_id": simulation_id,
            "platform": args.platform,
            "max_rounds": args.max_rounds,
            "enable_graph_memory_update": args.enable_graph_memory_update,
            "force": args.force_run,
        },
    )
    run_data = require_success(response, "启动模拟")["data"]
    summary["run_start"] = run_data
    if args.no_wait_run:
        print("已启动模拟，按参数要求不等待完成")
    else:
        summary["run_result"] = poll_run(
            client,
            simulation_id=simulation_id,
            timeout_seconds=args.run_timeout,
            poll_interval=args.poll_interval,
        )

    if not args.skip_report and not args.no_wait_run:
        print_stage("6. 生成报告")
        response = client.post(
            "/api/report/generate",
            json={
                "simulation_id": simulation_id,
                "force_regenerate": args.force_report,
            },
        )
        report_data = require_success(response, "启动报告生成")["data"]
        report_id = report_data.get("report_id")
        report_task_id = report_data.get("task_id")
        summary["report_id"] = report_id
        summary["report_task_id"] = report_task_id
        if report_data.get("already_generated"):
            print(f"报告已存在: {report_id}")
        else:
            if not report_task_id:
                raise PipelineError(f"报告生成未返回 task_id: {report_data}")
            poll_task(
                client,
                task_id=report_task_id,
                status_endpoint="/api/report/generate/status",
                status_method="POST",
                label="报告生成",
                timeout_seconds=args.report_timeout,
                poll_interval=args.poll_interval,
                extra_payload={"simulation_id": simulation_id},
            )

        if report_id:
            from app.services.report_agent import ReportManager

            markdown_path = ReportManager._get_report_markdown_path(report_id)
            summary["report_markdown_path"] = markdown_path
            print(f"report_id: {report_id}")
            print(f"report_markdown_path: {markdown_path}")

    simulation_dir = BACKEND_DIR / "uploads" / "simulations" / simulation_id
    summary["simulation_dir"] = str(simulation_dir)
    summary["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    summary_path = Path(args.summary_path) if args.summary_path else simulation_dir / "cli_run_summary.json"
    write_summary(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    print(f"\nsummary_path: {summary_path}")

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MiroFish from one seed document and one prompt file."
    )
    parser.add_argument("--seed", default=DEFAULT_SEED_PATH, help="Reality seed document path.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT_PATH, help="Prompt file path.")
    parser.add_argument("--project-name", default="", help="Project name. Defaults to seed filename.")
    parser.add_argument("--graph-name", default="", help="Graph name. Defaults to project name.")
    parser.add_argument("--additional-context", default="", help="Extra context for ontology generation.")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--platform", choices=["twitter", "reddit", "parallel"], default="parallel")
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Optional max simulation rounds. By default, use the generated config without truncation.",
    )
    parser.add_argument("--parallel-profile-count", type=int, default=5)
    parser.add_argument("--no-llm-profiles", action="store_true", help="Disable LLM-enhanced profiles.")
    parser.add_argument("--twitter-only", action="store_true", help="Enable Twitter only.")
    parser.add_argument("--reddit-only", action="store_true", help="Enable Reddit only.")
    parser.add_argument("--enable-graph-memory-update", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force-run", action="store_true")
    parser.add_argument("--force-report", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--no-wait-run", action="store_true", help="Start simulation and exit without report.")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--graph-timeout", type=int, default=3600)
    parser.add_argument("--prepare-timeout", type=int, default=7200)
    parser.add_argument("--run-timeout", type=int, default=7200)
    parser.add_argument("--report-timeout", type=int, default=7200)
    parser.add_argument("--summary-path", default="", help="Where to write the run summary JSON.")
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.twitter_only and args.reddit_only:
        raise PipelineError("--twitter-only 和 --reddit-only 不能同时使用")
    args.enable_twitter = not args.reddit_only
    args.enable_reddit = not args.twitter_only
    if args.twitter_only:
        args.platform = "twitter"
    if args.reddit_only:
        args.platform = "reddit"
    if args.max_rounds is not None and args.max_rounds <= 0:
        raise PipelineError("--max-rounds 必须大于 0")
    if args.parallel_profile_count <= 0:
        raise PipelineError("--parallel-profile-count 必须大于 0")
    return args


def main() -> int:
    parser = build_parser()
    args = normalize_args(parser.parse_args())

    try:
        summary = run_pipeline(args)
    except KeyboardInterrupt:
        print("\n已中断")
        return 130
    except Exception as exc:
        print(f"\n运行失败: {exc}", file=sys.stderr)
        return 1

    print("\n运行完成")
    print(f"project_id: {summary.get('project_id')}")
    print(f"graph_id: {summary.get('graph_id')}")
    print(f"simulation_id: {summary.get('simulation_id')}")
    if summary.get("report_id"):
        print(f"report_id: {summary.get('report_id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
