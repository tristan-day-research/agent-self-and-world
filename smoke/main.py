from core.device import resolve_device
from core.runs import run_context, write_json


def main(args):
    # This experiment is always small; --smoke is also the contract for future runs.
    with run_context(args.experiment, vars(args)) as run:
        device = resolve_device(args.device)
        run.logger.info("Selected device: %s; this check performs only file I/O", device)
        write_json(run.directory / "outputs.json", {
            "kind": "plumbing_check",
            "selected_device": device,
            "model_loaded": False,
            "sae_loaded": False,
            "message": "Experiment dispatch and artifact writing succeeded.",
        })
    return 0
