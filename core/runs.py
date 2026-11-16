"""One output directory per run, with config, logging, and completion status."""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
import re
import tempfile
from uuid import uuid4
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Keep aligned with RESULTS_SUBDIR in .runpod.env.
RESULTS_SUBDIR = Path("outputs")


def write_json(path, data):
    """Atomically replace a JSON artifact; callers keep paths inside their run."""
    path = Path(path)
    text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class PacificFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return datetime.fromtimestamp(record.created, ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %I:%M:%S %p %Z")


@dataclass(frozen=True)
class Run:
    directory: Path
    logger: logging.Logger


@contextmanager
def run_context(experiment, config, resume_directory=None):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", experiment):
        raise ValueError("Experiment names must contain lowercase letters, digits, or underscores")
    started = datetime.now(timezone.utc)
    local = started.astimezone(ZoneInfo("America/Los_Angeles"))
    label = config.get("run_label") or ("attention" if config.get("attention") else "run")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", label):
        raise ValueError("Run label must be a short lowercase slug")
    synthetic = bool(config.get("smoke")) or experiment == "smoke"
    run_id = local.strftime("%Y-%m-%d_%H-%M-%S_%Z") + "--" + label + "--" + uuid4().hex[:8]
    base = PROJECT_ROOT / RESULTS_SUBDIR
    directory = (base / "_smoke" / experiment if synthetic else base / experiment) / run_id
    previous_status = None
    if resume_directory is not None:
        directory = Path(resume_directory).resolve()
        if not any(directory.is_relative_to(p.resolve()) for p in [base / experiment, base / "_smoke" / experiment]):
            raise ValueError("Resume directory must be inside this experiment's output tree")
        previous_status = json.loads((directory / "status.json").read_text())
        if previous_status["state"] not in {"failed", "interrupted"}:
            raise ValueError("Resume requires a failed/interrupted run; do not overlap active or completed work")
        if json.loads((directory / "config.json").read_text()) != config:
            raise ValueError("Resume configuration differs; start a new run for changed parameters")
        run_id = directory.name
    else:
        directory.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger(f"runs.{experiment}.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.FileHandler(directory / "run.log", encoding="utf-8"), logging.StreamHandler()]
    for handler in handlers:
        handler.setFormatter(PacificFormatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    status = {"run_id": run_id, "experiment": experiment, "state": "running",
              "started_at": started.isoformat(), "python": platform.python_version(),
              "platform": platform.platform()}
    if previous_status is not None:
        status.update(started_at=previous_status["started_at"], resumed_at=started.isoformat(),
                      resume_count=previous_status.get("resume_count", 0)+1)
    from core.run_index import pacific_time
    status["started_at_pacific"] = pacific_time(status["started_at"])
    status["run_kind"] = "synthetic_smoke" if synthetic else "research"
    try:
        write_json(directory / "config.json", config)
        write_json(directory / "status.json", status)
        logger.info("Run directory: %s", directory.resolve().relative_to(PROJECT_ROOT.resolve()))
        yield Run(directory, logger)
    except BaseException as exc:
        status.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error_type=type(exc).__name__)
        logger.exception("Run did not complete")
        raise
    else:
        status["state"] = "completed"
        logger.info("Run completed")
    finally:
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        status["finished_at_pacific"] = pacific_time(status["finished_at"])
        try:
            write_json(directory / "status.json", status)
            if experiment == "ism":
                try:
                    from core.reporting import generate_evidence
                    generate_evidence(directory)
                except Exception:
                    # Preserve saved measurements and the original experiment outcome.
                    logger.exception("Evidence generation failed; regenerate with python -m core.reporting %s", directory)
            try:
                from core.run_index import build_index
                build_index(PROJECT_ROOT / RESULTS_SUBDIR)
            except Exception:
                logger.exception("Run index refresh failed; run python -m core.run_index")
        finally:
            for handler in handlers:
                logger.removeHandler(handler)
                handler.close()
