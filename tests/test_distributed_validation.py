import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def test_two_process_validation_reduction_and_checkpoint(tmp_path):
    if sys.platform != "linux" or not torch.distributed.is_gloo_available():
        pytest.skip("CPU process-group probe requires Linux/Gloo")
    script = Path(__file__).with_name("distributed_validation_probe.py")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in [0, 1]:
        metrics = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert metrics["world_size"] == 2
        assert metrics["val_mae_reg"] == pytest.approx(80.0)
        assert metrics["val_mae_cls"] == pytest.approx((99.5 + 39.5 + 20.5) / 3)
    checkpoint = torch.load(tmp_path / "distributed.ckpt", weights_only=True)
    assert checkpoint["training_fold"] == 0
    assert checkpoint["training_data"]["manifest_sha256"]
