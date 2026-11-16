"""Optional R2 uploads. Nothing uploads unless an experiment calls this helper."""

import os
from pathlib import Path


def sync_to_r2(local_dir, *, prefix):
    """Upload a run's artifact directory; return file count, or 0 without credentials.

    Reads the names supplied by the user's RunPod workflow. Incomplete credentials,
    missing dependencies, and upload errors fail visibly instead of pretending to
    have backed up results. Symlinks are excluded from directory uploads.
    """
    names = ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
    values = {name: os.environ.get(name) for name in names}
    if not any(values.values()):
        return 0
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError("Incomplete R2 configuration: " + ", ".join(missing))
    directory = Path(local_dir)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("R2 source must be a real artifact directory")
    prefix = prefix.strip("/")
    if not prefix or any(part in {".", "..", ""} for part in prefix.split("/")):
        raise ValueError("Use a nonempty R2 prefix unique to this project and run")
    import boto3

    client = boto3.client("s3", endpoint_url=values["R2_ENDPOINT"],
                          aws_access_key_id=values["R2_ACCESS_KEY_ID"],
                          aws_secret_access_key=values["R2_SECRET_ACCESS_KEY"],
                          region_name="auto")
    count = 0
    for path in sorted(directory.rglob("*")):
        if path.is_file() and not path.is_symlink():
            key = f"{prefix}/{path.relative_to(directory).as_posix()}"
            client.upload_file(str(path), values["R2_BUCKET"], key)
            count += 1
    return count
