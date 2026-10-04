import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from dubflow_worker.dev_api import main as api


def test_health_check():
    response = TestClient(api.app).get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "dubflow-dev-api"}


def test_job_lifecycle_and_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "JOBS_DIR", tmp_path)

    def fake_run(video, output, *, progress, **options):
        assert video.read_bytes() == b"video-bytes"
        assert options == {
            "source_language": "auto", "target_language": "vi", "voice": "gia_bao",
            "device": "gpu", "steps": 32,
        }
        progress("ASR", 15)
        artifact = output / "dubbed_video.mp4"
        artifact.write_bytes(b"mp4-bytes")
        return artifact

    monkeypatch.setattr(api, "run_dubbing", fake_run)
    api.jobs.clear()
    client = TestClient(api.app)
    created = client.post("/api/v1/jobs", files={"video": ("sample.mp4", b"video-bytes", "video/mp4")})
    assert created.status_code == 202
    job_id = created.json()["jobId"]
    status = client.get(f"/api/v1/jobs/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "COMPLETED"
    assert status.json()["stage"] == "COMPLETED"
    assert status.json()["progress"] == 100
    assert status.json()["artifactUrl"] == f"/api/v1/artifacts/{job_id}/video"
    artifact = client.get(status.json()["artifactUrl"])
    assert artifact.status_code == 200
    assert artifact.content == b"mp4-bytes"
    assert client.get("/api/v1/jobs/not-a-job").status_code == 404
    assert client.get("/api/v1/artifacts/not-a-job/video").status_code == 404


def test_rejects_unsupported_video(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "JOBS_DIR", tmp_path)
    response = TestClient(api.app).post(
        "/api/v1/jobs", files={"video": ("notes.txt", b"text", "text/plain")}
    )
    assert response.status_code == 415


def test_job_accepts_korva_voice_device_and_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "JOBS_DIR", tmp_path)
    captured = {}

    def fake_run(video, output, *, progress, **options):
        captured.update(options)
        artifact = output / "dubbed_video.mp4"
        artifact.write_bytes(b"mp4-bytes")
        return artifact

    monkeypatch.setattr(api, "run_dubbing", fake_run)
    response = TestClient(api.app).post(
        "/api/v1/jobs",
        files={"video": ("sample.mp4", b"video-bytes", "video/mp4")},
        data={"voice": "quynh_nhu", "device": "cpu", "steps": "20"},
    )

    assert response.status_code == 202
    assert captured == {
        "source_language": "auto", "target_language": "vi", "voice": "quynh_nhu",
        "device": "cpu", "steps": 20,
    }


def test_rejects_missing_or_empty_video(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "JOBS_DIR", tmp_path)
    client = TestClient(api.app)
    assert client.post("/api/v1/jobs").status_code == 422
    response = client.post(
        "/api/v1/jobs", files={"video": ("empty.mp4", b"", "video/mp4")}
    )
    assert response.status_code == 400
