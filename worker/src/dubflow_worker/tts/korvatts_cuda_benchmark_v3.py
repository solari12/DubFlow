from __future__ import annotations

import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any

from dubflow_worker.tts.benchmark_v1 import TTSBenchmarkFixture, _windows_peak_working_set_mib, validate_wav
from dubflow_worker.tts.cuda_benchmark_v1 import _GpuSampler, _environment_snapshot, classify_cuda_failure


PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"
VOICE = "gia_bao"
STEPS = 32
SPEED = 1.05


def resolve_korva_assets(assets_dir: Path | None = None) -> Path:
    from korvatts.assets import resolve_assets_dir

    if assets_dir is not None:
        return Path(resolve_assets_dir(assets_dir, auto_download=False))
    try:
        return Path(resolve_assets_dir(auto_download=False))
    except FileNotFoundError:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(
            repo_id="dogenthq/KorvaTTS",
            allow_patterns=["onnx/*", "voice_styles/*"],
            local_files_only=True,
        ))


def verify_session_providers(sessions: dict[str, Any]) -> dict[str, Any]:
    """Require CUDA to be the primary provider for every Korva ONNX graph."""
    providers = {name: list(session.get_providers()) for name, session in sessions.items()}
    if not providers or any(not values or values[0] != PROVIDER for values in providers.values()):
        raise RuntimeError(f"CUDA session verification failed; expected {PROVIDER} first for every graph, got {providers}")
    return {
        "provider_by_session": providers,
        "cuda_primary_for_every_session": True,
        "cpu_ep_registered_as_fallback": any(CPU_PROVIDER in values for values in providers.values()),
    }


def calculate_rtf(synthesis_seconds: float, duration_seconds: float) -> float:
    if duration_seconds <= 0:
        raise ValueError("Audio duration must be positive to calculate RTF")
    return synthesis_seconds / duration_seconds


def synthesize_case(tts: Any, text: str, path: Path, *, voice: str = VOICE, steps: int = STEPS, speed: float = SPEED) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    start_total = time.perf_counter()
    start_synthesis = time.perf_counter()
    wav_data, returned_duration = tts.synthesize(text, voice=voice, lang="vi", total_steps=steps, speed=speed)
    synthesis_seconds = time.perf_counter() - start_synthesis
    tts.save_audio(wav_data, str(path))
    metadata = validate_wav(path)
    total_wall_seconds = time.perf_counter() - start_total
    return {
        **metadata,
        "duration_seconds": metadata["duration_seconds"],
        "korva_returned_duration_seconds": float(returned_duration),
        "synthesis_seconds": synthesis_seconds,
        "total_wall_seconds": total_wall_seconds,
        "rtf": calculate_rtf(synthesis_seconds, metadata["duration_seconds"]),
        "file_size_bytes": path.stat().st_size,
        "wav_validation": "PASS",
        "output_wav": str(path.resolve()),
    }


def _create_tts(*, assets_dir: Path, profile: bool, profile_dir: Path | None = None):
    import onnxruntime as ort
    import korvatts.tts as korva_tts_module
    from korvatts import TTS
    from korvatts.session import ModelSessions

    ort.preload_dlls()
    original_sessions = korva_tts_module.ModelSessions
    if not profile:
        return TTS(assets_dir=str(assets_dir), device="gpu", auto_download=False)

    profile_dir = profile_dir or assets_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    class ProfileSessions(ModelSessions):
        @classmethod
        def load(cls, onnx_dir, device="auto", num_threads=None):
            if device != "gpu":
                raise RuntimeError(f"Provider verification requested unexpected device {device!r}")
            options = ort.SessionOptions()
            options.enable_profiling = True
            options.profile_file_prefix = str(profile_dir / "korva-session")
            if num_threads:
                options.intra_op_num_threads = num_threads
            providers = [PROVIDER, CPU_PROVIDER]

            def open_session(name):
                return ort.InferenceSession(str(onnx_dir / name), sess_options=options, providers=providers)

            return cls(
                duration_predictor=open_session("duration_predictor.onnx"),
                text_encoder=open_session("text_encoder.onnx"),
                vector_estimator=open_session("vector_estimator.onnx"),
                vocoder=open_session("vocoder.onnx"),
            )

    korva_tts_module.ModelSessions = ProfileSessions
    try:
        return TTS(assets_dir=str(assets_dir), device="gpu", auto_download=False)
    finally:
        korva_tts_module.ModelSessions = original_sessions


def _finish_profile(sessions: dict[str, Any], profile_dir: Path) -> dict[str, Any]:
    profile_files = []
    for name, session in sessions.items():
        path = Path(session.end_profiling())
        profile_files.append({"session": name, "path": path.resolve()})
    counts = {PROVIDER: 0, CPU_PROVIDER: 0, "other": {}}
    counts_by_session = {}
    for entry in profile_files:
        per_session = {PROVIDER: 0, CPU_PROVIDER: 0, "other": {}}
        path = entry["path"]
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
            for event in rows:
                if event.get("cat") != "Node":
                    continue
                provider = (event.get("args") or {}).get("provider")
                if provider == PROVIDER:
                    counts[PROVIDER] += 1
                    per_session[PROVIDER] += 1
                elif provider == CPU_PROVIDER:
                    counts[CPU_PROVIDER] += 1
                    per_session[CPU_PROVIDER] += 1
                elif provider:
                    counts["other"][provider] = counts["other"].get(provider, 0) + 1
                    per_session["other"][provider] = per_session["other"].get(provider, 0) + 1
        finally:
            counts_by_session[entry["session"]] = per_session
            # Keep only the compact provider counts in the artifact, not multi-megabyte ORT traces.
            if path.parent.resolve() == Path(profile_dir).resolve() and path.is_file():
                path.unlink()
    if counts[PROVIDER] <= 0:
        raise RuntimeError(f"ORT profile did not record any nodes executed by {PROVIDER}: {counts}")
    return {
        "status": "PASS" if counts[CPU_PROVIDER] == 0 and not counts["other"] else "CPU_EP_FALLBACK_DETECTED",
        "execution_node_event_counts": counts,
        "execution_node_event_counts_by_session": counts_by_session,
        "profile_trace_files_consumed": [path["path"].name for path in profile_files],
        "profile_traces_retained": False,
        "interpretation": "ORT node events are used to distinguish actual CUDAExecutionProvider execution from CPUExecutionProvider fallback. CPU node events cause CUDA_FAILED per phase requirements.",
    }


def verify_cuda_execution(assets_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Profile a short probe independently so timing runs stay unprofiled."""
    tts = _create_tts(assets_dir=assets_dir, profile=True, profile_dir=output_dir / "provider-verification")
    sessions = {
        name: getattr(tts.sessions, name)
        for name in ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")
    }
    verification: dict[str, Any] = verify_session_providers(sessions)
    try:
        tts.synthesize("Tôi đang kiểm tra nhà cung cấp CUDA.", voice=VOICE, lang="vi", total_steps=STEPS, speed=SPEED)
        verification["profile_probe_synthesis"] = "PASS"
    finally:
        verification.update(_finish_profile(sessions, output_dir / "provider-verification"))
        del sessions, tts
        gc.collect()
    verification["profile_probe_audio_written"] = False
    return verification


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def comparison_data(repo: Path, new_report: dict[str, Any]) -> dict[str, Any]:
    root = repo / "output/tts-benchmark-v1"
    piper_all = _read_json(root / "benchmark-results.json")
    piper = next((r for r in piper_all if r.get("provider") == "piper"), None) if isinstance(piper_all, list) else None
    korva_cpu = _read_json(root / "korvatts/report.json")
    kokoro = _read_json(repo / "output/tts-benchmark-v2/kokoro/benchmark.json")

    def original(row):
        return next((case for case in row.get("cases", []) if case.get("case") == "original"), {}) if isinstance(row, dict) else {}

    piper_case = original(piper)
    cpu_case = original(korva_cpu)
    kokoro_rows = kokoro.get("results", []) if isinstance(kokoro, dict) else []
    kokoro_summary = {}
    for device in ("cpu", "cuda"):
        vals = []
        for result in kokoro_rows:
            if result.get("device") != device:
                continue
            case = next((c for c in result.get("cases", []) if c.get("case") == "original"), {})
            if case.get("status") == "SUCCESS":
                vals.append({"voice": result.get("voice"), "rtf": case.get("rtf"), "duration_seconds": case.get("duration_seconds"), "synthesis_seconds": case.get("average_synthesis_seconds", case.get("synthesis_seconds"))})
        kokoro_summary[device] = vals

    def row(provider, device, case, voice=None):
        return {
            "provider": provider, "device": device, "voice": voice,
            "status": case.get("status", "N/A") if case else "N/A",
            "rtf": case.get("rtf") if case else None,
            "audio_duration_seconds": case.get("duration_seconds") if case else None,
            "wall_time_seconds": case.get("synthesis_seconds", case.get("average_synthesis_seconds")) if case else None,
        }

    current_case = next((case for case in new_report.get("cases", []) if case.get("case") == "original"), {})
    rows = [
        row("Piper VIVOS x_low", "CPU", piper_case),
        row("KorvaTTS", "CPU", cpu_case, (korva_cpu or {}).get("provider_configuration", {}).get("voice") if isinstance(korva_cpu, dict) else None),
        {
            "provider": "KorvaTTS", "device": "CUDA", "voice": new_report.get("settings", {}).get("voice"),
            "status": new_report.get("status", "N/A"),
            "rtf": new_report.get("original_repeated_runs", {}).get("average_rtf"),
            "audio_duration_seconds": statistics.mean([r["duration_seconds"] for r in new_report.get("original_repeated_runs", {}).get("runs", []) if r.get("success") and r.get("duration_seconds") is not None]) if any(r.get("success") and r.get("duration_seconds") is not None for r in new_report.get("original_repeated_runs", {}).get("runs", [])) else None,
            "wall_time_seconds": new_report.get("original_repeated_runs", {}).get("average_synthesis_seconds"),
            "wav_generation_status": current_case.get("status", "N/A"),
        },
    ]
    for device, vals in kokoro_summary.items():
        if vals:
            rows.append({
                "provider": "Kokoro Vietnamese", "device": device, "voice": "3 voices; per-voice data in benchmark-v2 artifact",
                "status": "COMPLETED", "rtf": {"min": min(v["rtf"] for v in vals), "mean": statistics.mean(v["rtf"] for v in vals), "max": max(v["rtf"] for v in vals)},
                "audio_duration_seconds": {"min": min(v["duration_seconds"] for v in vals), "mean": statistics.mean(v["duration_seconds"] for v in vals), "max": max(v["duration_seconds"] for v in vals)},
                "wall_time_seconds": {"min": min(v["synthesis_seconds"] for v in vals), "mean": statistics.mean(v["synthesis_seconds"] for v in vals), "max": max(v["synthesis_seconds"] for v in vals)},
                "source_voices": vals,
            })
        else:
            rows.append({"provider": "Kokoro Vietnamese", "device": device, "status": "N/A - previous benchmark artifact not found", "rtf": None, "audio_duration_seconds": None, "wall_time_seconds": None})
    return {"basis": "Descriptive comparison copied from existing benchmark artifacts; Kokoro rows summarize its three tested voices; no ranking or winner selection.", "rows": rows}


def _fmt(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, (float, int)) else "N/A"


def _format_summary(value: Any) -> str:
    if isinstance(value, dict):
        return f"mean {_fmt(value.get('mean'))} (min {_fmt(value.get('min'))}, max {_fmt(value.get('max'))})"
    return _fmt(value)


def render_report(report: dict[str, Any]) -> str:
    env, repeated = report["environment"], report["original_repeated_runs"]
    gpu = env.get("gpu") or {}
    memory = report.get("gpu_memory_measurements") or {}
    system = memory.get("system_wide_sampled") or {}
    process = memory.get("process_level_sampled") or {}
    lines = [
        "# Phase 1K.4.3 - KorvaTTS CUDA benchmark", "",
        f"Status: `{report['status']}`. No automatic quality score or winner is selected.", "",
        "## Environment", "",
        f"- GPU: {gpu.get('name') or 'N/A'}; total VRAM: {_fmt(gpu.get('total_vram_mib'))} MiB; driver: {gpu.get('driver_version') or 'N/A'}.",
        f"- Python: {env.get('python')}; ONNX Runtime: {env.get('onnxruntime', {}).get('version')}; KorvaTTS: {env.get('packages', {}).get('korvatts')}.",
        f"- Available ORT providers: {', '.join(env.get('onnxruntime', {}).get('available_providers') or []) or 'N/A'}.",
        f"- CUDA provider verification: `{report.get('cuda_execution_verification', {}).get('status', 'FAILED')}`; primary provider per graph: `{PROVIDER}`.",
        f"- Voice: `{report['settings']['voice']}`; language: `vi`; steps: {report['settings']['steps']}; speed: {report['settings']['speed']}; seed: {report['settings'].get('seed') or 'not specified by fixture'}.",
        f"- Fixture: `{report['fixture']['path']}`; SHA-256 `{report['fixture']['sha256']}`; used unchanged.", "",
        "## Original full-text performance", "",
        f"Initialization: {_fmt(report.get('model_initialization_seconds'))} seconds. Warm-up: {_fmt((repeated.get('warmup') or {}).get('synthesis_seconds'))} seconds (excluded from measured mean).",
        "| Run | Status | Synthesis wall seconds | Total wall seconds | Audio duration seconds | RTF (recomputed) | Sample rate | Channels | Samples | File bytes |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in repeated.get("runs", []):
        lines.append(f"| {run.get('run')} | {run.get('status')} | {_fmt(run.get('synthesis_seconds'))} | {_fmt(run.get('total_wall_seconds'))} | {_fmt(run.get('duration_seconds'))} | {_fmt(run.get('rtf'))} | {run.get('sample_rate', 'N/A')} | {run.get('channels', 'N/A')} | {run.get('frame_count', 'N/A')} | {run.get('file_size_bytes', 'N/A')} |")
    lines.extend([
        "", f"Measured mean: synthesis {_fmt(repeated.get('average_synthesis_seconds'))} s (min {_fmt(repeated.get('min_synthesis_seconds'))}, max {_fmt(repeated.get('max_synthesis_seconds'))}); total wall {_fmt(repeated.get('average_total_wall_seconds'))} s; mean RTF {_fmt(repeated.get('average_rtf'))} (min {_fmt(repeated.get('min_rtf'))}, max {_fmt(repeated.get('max_rtf'))}).",
        "", "## CUDA and memory verification", "",
        f"- CUDA nodes in independent profile probe: {(report.get('cuda_execution_verification', {}).get('execution_node_event_counts') or {}).get(PROVIDER, 0)}.",
        f"- CPU EP nodes in independent profile probe: {(report.get('cuda_execution_verification', {}).get('execution_node_event_counts') or {}).get(CPU_PROVIDER, 0)}.",
        f"- Coarse system-wide VRAM sampled peak: {_fmt(system.get('peak_sampled_mib'))} MiB; method: {system.get('method', 'N/A')}.",
        f"- Process-level VRAM: {_fmt(process.get('peak_sampled_mib'))} MiB; method: {process.get('method', 'N/A')}; {process.get('error') or 'sampled value is coarse, not exact peak'}.",
        f"- Process peak working set: {_fmt(report.get('peak_process_working_set_mib'))} MiB. ORT does not expose a PyTorch allocator metric.",
        "", "## Five fixture cases", "", "| Case | Status | Audio seconds | Wall seconds | RTF | Sample rate | Channels | Samples | WAV check | File bytes |", "|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ])
    for case in report.get("cases", []):
        lines.append(f"| {case['case']} | {case['status']} | {_fmt(case.get('duration_seconds'))} | {_fmt(case.get('synthesis_seconds'))} | {_fmt(case.get('rtf'))} | {case.get('sample_rate', 'N/A')} | {case.get('channels', 'N/A')} | {case.get('frame_count', 'N/A')} | {case.get('wav_validation', 'N/A')} | {case.get('file_size_bytes', 'N/A')} |")
        if case.get("error"):
            lines.append(f"|  | Error: `{case['error']}` |  |  |  |  |  |  |  |  |")
    lines.extend(["", "## Descriptive comparison", "", "No ranking or winner selection. Values come from the prior benchmark artifacts and this run; Kokoro summarizes three voices. The prior Korva CPU run used `khanh_vy`; this CUDA run used `gia_bao`, so this is not a same-voice comparison.", "", "| Provider | Device | Voice | RTF | Audio duration s | Wall time s | Status |", "|---|---|---|---:|---:|---:|---|"])
    for row in report["comparison"]["rows"]:
        lines.append(f"| {row.get('provider')} | {row.get('device')} | {row.get('voice') or 'N/A'} | {_format_summary(row.get('rtf'))} | {_format_summary(row.get('audio_duration_seconds'))} | {_format_summary(row.get('wall_time_seconds'))} | {row.get('status')} |")
    lines.extend([
        "", "## Reliability and human review", "",
        f"- CUDA initialization: `{report.get('initialization_status')}`; failed fixture cases: {report.get('failure_count')}; CUDA initialization failures: {report.get('cuda_initialization_failure_count')}; full-device CPU fallbacks: {report.get('cpu_device_fallback_occurrences')}.",
        "- CPU EP node executions in the separate profiling probe: " + str(report.get("cpu_ep_fallback_occurrences", 0)) + "; any CPU node execution marks CUDA compliance `CUDA_FAILED` (while retaining measured timings and successful WAVs for diagnosis).",
        f"- CUDA provider compliance failures: {report.get('provider_compliance_failure_count')}.",
        "- WAV validation: " + ("PASS" if report.get("wav_validation_failure_count") == 0 and report.get("success_count") == len(report.get("cases", [])) else "FAIL") + ".",
        "- All human quality items start `NOT_REVIEWED`; listen manually. No acoustic quality metric is computed.",
        "", "## Scope", "", "Benchmark-only. No production provider, translation, alignment, render, or API behavior changed. No full-video validation was run.", "",
    ])
    return "\n".join(lines)


def build_quality_review(cases: list[dict[str, Any]]) -> str:
    lines = ["# KorvaTTS CUDA listening review", "", "All entries require manual listening; no automatic score is produced.", "", "| Case | Voice | Status | Human quality | Notes | WAV |", "|---|---|---|---|---|---|"]
    for case in cases:
        lines.append(f"| {case['case']} | `{VOICE}` | {case['status']} | NOT_REVIEWED |  | `{case.get('output_wav', 'N/A')}` |")
    return "\n".join(lines) + "\n"


def run_benchmark(*, fixture_path: Path, output_dir: Path, repo: Path, assets_dir: Path | None = None, voice: str = VOICE, steps: int = STEPS, speed: float = SPEED) -> dict[str, Any]:
    fixture_path, output_dir, repo = map(Path, (fixture_path, output_dir, repo))
    fixture = TTSBenchmarkFixture.load(fixture_path)
    fixture_data = json.loads(fixture_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = resolve_korva_assets(Path(assets_dir) if assets_dir is not None else None)
    report: dict[str, Any] = {
        "benchmark_version": "3.0", "phase": "1K.4.3", "provider": "KorvaTTS", "model": "dogenthq/KorvaTTS",
        "status": "CUDA_FAILED", "device": "gpu", "initialization_status": "NOT_STARTED",
        "fixture": {"path": str(fixture_path.resolve()), "sha256": __import__("hashlib").sha256(fixture_path.read_bytes()).hexdigest(), "version": fixture_data.get("version"), "original_text": fixture.original_text, "normalized_text": fixture.normalized_text, "cases": fixture.cases()},
        "settings": {"voice": voice, "language": "vi", "steps": steps, "speed": speed, "seed": None, "seed_note": "The shared fixture does not define a seed; Korva's default nondeterministic seed behavior is retained."},
        "environment": _environment_snapshot(), "cuda_execution_verification": {}, "cases": [],
        "original_repeated_runs": {"warmup": None, "runs": [], "average_synthesis_seconds": None, "min_synthesis_seconds": None, "max_synthesis_seconds": None, "average_total_wall_seconds": None, "average_rtf": None, "min_rtf": None, "max_rtf": None},
        "gpu_memory_measurements": None, "peak_process_working_set_mib": None,
        "success_count": 0, "failure_count": 0, "cuda_initialization_failure_count": 0,
        "cpu_device_fallback_occurrences": 0, "cpu_ep_fallback_occurrences": 0,
        "provider_compliance_failure_count": 0, "wav_validation_failure_count": 0,
        "production_changed": False,
    }
    try:
        report["cuda_execution_verification"] = verify_cuda_execution(assets_dir, output_dir)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        report["cuda_execution_verification"] = {"status": "FAIL", "error": error}

    sampler = _GpuSampler()
    sampler.start()
    tts = None
    init_started = time.perf_counter()
    try:
        tts = _create_tts(assets_dir=assets_dir, profile=False)
        report["model_initialization_seconds"] = time.perf_counter() - init_started
        report["initialization_status"] = "LOADED"
        report["execution_providers"] = verify_session_providers({name: getattr(tts.sessions, name) for name in ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        report["initialization_status"] = "FAILED"
        report["initialization_error"] = error
        report["cuda_initialization_failure_count"] = 1
        report["status"] = classify_cuda_failure(error)

    if tts is not None:
        report["status"] = "CUDA_SUPPORTED"
        cases = fixture.cases()
        repeated = report["original_repeated_runs"]
        try:
            warm = synthesize_case(tts, fixture.original_text, output_dir / "repeated/warmup.wav", voice=voice, steps=steps, speed=speed)
            repeated["warmup"] = {"success": True, **warm}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            repeated["warmup"] = {"success": False, "error": error}
            report["status"] = classify_cuda_failure(error)
        for run_num in range(1, 4):
            if report["status"] != "CUDA_SUPPORTED":
                repeated["runs"].append({"run": run_num, "status": "SKIPPED_AFTER_WARMUP_FAILURE", "success": False, "error": repeated["warmup"].get("error")})
                continue
            path = output_dir / ("original.wav" if run_num == 1 else f"repeated/original-run-{run_num}.wav")
            try:
                repeated["runs"].append({"run": run_num, "success": True, "status": "SUCCESS", **synthesize_case(tts, fixture.original_text, path, voice=voice, steps=steps, speed=speed)})
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                repeated["runs"].append({"run": run_num, "success": False, "status": classify_cuda_failure(error), "error": error, "output_wav": str(path.resolve())})
                report["status"] = classify_cuda_failure(error)
        successful_runs = [run for run in repeated["runs"] if run.get("success")]
        if successful_runs:
            repeated.update({
                "average_synthesis_seconds": statistics.mean(r["synthesis_seconds"] for r in successful_runs),
                "min_synthesis_seconds": min(r["synthesis_seconds"] for r in successful_runs),
                "max_synthesis_seconds": max(r["synthesis_seconds"] for r in successful_runs),
                "average_total_wall_seconds": statistics.mean(r["total_wall_seconds"] for r in successful_runs),
                "min_total_wall_seconds": min(r["total_wall_seconds"] for r in successful_runs),
                "max_total_wall_seconds": max(r["total_wall_seconds"] for r in successful_runs),
                "average_rtf": statistics.mean(r["rtf"] for r in successful_runs),
                "min_rtf": min(r["rtf"] for r in successful_runs),
                "max_rtf": max(r["rtf"] for r in successful_runs),
            })
        original = next((r for r in repeated["runs"] if r.get("success")), None)
        report["cases"].append({"case": "original", "input_text": cases["original"], **(original or {"status": "FAILED", "error": repeated["warmup"].get("error", "No successful original run")})})
        for name, text in cases.items():
            if name == "original":
                continue
            path = output_dir / f"{name}.wav"
            try:
                report["cases"].append({"case": name, "input_text": text, "status": "SUCCESS", **synthesize_case(tts, text, path, voice=voice, steps=steps, speed=speed)})
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                report["cases"].append({"case": name, "input_text": text, "output_wav": str(path.resolve()), "status": classify_cuda_failure(error), "error": error, "wav_validation": "FAIL"})
                report["status"] = classify_cuda_failure(error)
        if report["cuda_execution_verification"].get("status") != "PASS":
            report["status"] = "CUDA_FAILED"
            report["cpu_ep_fallback_occurrences"] = (report["cuda_execution_verification"].get("execution_node_event_counts") or {}).get(CPU_PROVIDER, 0)
            report["provider_compliance_failure_count"] = 1
    else:
        report["cases"] = [{"case": name, "input_text": text, "status": report["status"], "error": report.get("initialization_error"), "wav_validation": "NOT_RUN"} for name, text in fixture.cases().items()]
        report["original_repeated_runs"]["warmup"] = {"success": False, "error": report.get("initialization_error")}
        report["original_repeated_runs"]["runs"] = [{"run": n, "success": False, "status": "SKIPPED", "error": "CUDA initialization failed"} for n in range(1, 4)]

    report["peak_process_working_set_mib"] = _windows_peak_working_set_mib()
    report["gpu_memory_measurements"] = sampler.finish()
    if tts is not None:
        del tts
        gc.collect()
    report["success_count"] = sum(case.get("status") == "SUCCESS" for case in report["cases"])
    report["failure_count"] = len(report["cases"]) - report["success_count"]
    report["wav_validation_failure_count"] = sum(case.get("wav_validation") not in {"PASS", "NOT_RUN"} for case in report["cases"])
    if report["cuda_execution_verification"].get("status") != "PASS" and report["initialization_status"] == "LOADED":
        report["status"] = "CUDA_FAILED"
        report["cpu_ep_fallback_occurrences"] = (report["cuda_execution_verification"].get("execution_node_event_counts") or {}).get(CPU_PROVIDER, 0)
        report["provider_compliance_failure_count"] = 1
    report["comparison"] = comparison_data(repo, report)
    return report
