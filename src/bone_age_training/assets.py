"""Versioned preprocessing assets used by the published bone-age recipe."""

import hashlib
from pathlib import Path

RELEASE_REVISION = "2ab81275b84e9f518f04584177221a1d8c1dc1a5"
CROP_REVISION = "43c044ed192d1f952af1ae34f57ab881edd5dd9d"
REFERENCE_SHA256 = "fc243a431d60e5486449d2fa0b52627b633f80fe5512599c0f9ba8ca8ca578b6"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def data_identity(manifest, reference):
    return {"manifest_sha256": sha256(manifest), "reference_sha256": sha256(reference)}
