"""Test that all packages and modules can be imported correctly."""
import pytest


class TestDependencies:
    """Verify that all required third-party packages are installed."""

    def test_torch(self):
        import torch
        assert torch.__version__ is not None

    def test_einops(self):
        import einops
        assert einops.__version__ is not None

    def test_transformers(self):
        import transformers
        assert transformers.__version__ is not None

    def test_peft(self):
        import peft
        assert peft.__version__ is not None

    def test_accelerate(self):
        import accelerate
        assert accelerate.__version__ is not None

    def test_numpy(self):
        import numpy
        assert numpy.__version__ is not None

    def test_scipy(self):
        import scipy
        assert scipy.__version__ is not None

    def test_sklearn(self):
        import sklearn
        assert sklearn.__version__ is not None

    def test_nibabel(self):
        import nibabel
        assert nibabel.__version__ is not None

    def test_pandas(self):
        import pandas
        assert pandas.__version__ is not None

    def test_h5py(self):
        import h5py
        assert h5py.__version__ is not None


class TestRADRATEImports:
    """Verify that RAD-RATE package modules can be imported."""

    def test_import_rad_rate_package(self):
        from rad_rate import RADRATE
        assert RADRATE is not None

    def test_import_rad_rate_submodules(self):
        from rad_rate import RADRATE, SimpleAttnPool, CrossAttnPool, GatedAttnPool
        assert all(cls is not None for cls in [RADRATE, SimpleAttnPool, CrossAttnPool, GatedAttnPool])

    def test_import_rad_rate_helpers(self):
        from rad_rate import exists, l2norm, cast_tuple, all_gather_batch
        assert exists(1) is True
        assert exists(None) is False


class TestVisionEncoderImports:
    """Verify that vision_encoder package modules can be imported."""

    def test_import_vision_encoder_package(self):
        from vision_encoder import VJEPA2Encoder, ResidualTemporalDownsample, get_optimizer
        assert VJEPA2Encoder is not None
        assert ResidualTemporalDownsample is not None
        assert get_optimizer is not None

    def test_import_vision_encoder_submodules(self):
        from vision_encoder.vjepa_encoder import VJEPA2Encoder, ResidualTemporalDownsample
        from vision_encoder.optimizer import get_optimizer
        assert all(obj is not None for obj in [VJEPA2Encoder, ResidualTemporalDownsample,
                                               get_optimizer])
