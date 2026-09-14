import h5py
import numpy as np
import pytest
import torch

from evaluate import export_predictions


class DummyModel(torch.nn.Module):
    def shared_step(self, batch):
        return batch


def test_export_real_targets_and_variable_prongs(tmp_path):
    batches = [
        (
            torch.tensor([0, 1]),
            torch.tensor([[0, -1], [1, 0]]),
            torch.tensor([[0.0, 5.0], [5.0, 0.0]]),
            torch.zeros(2, 2, 2),
        ),
        (torch.tensor([1]), torch.tensor([[1]]), torch.tensor([[5.0, 0.0]]), torch.zeros(1, 1, 2)),
    ]
    path = export_predictions(DummyModel(), batches, tmp_path / "predictions.h5")
    with h5py.File(path) as output:
        np.testing.assert_array_equal(output["event_targets"][:], [0, 1, 1])
        np.testing.assert_array_equal(output["event_predictions"][:], [1, 0, 0])
        np.testing.assert_array_equal(output["prong_targets"][:], [0, 1, 0, 1])
        np.testing.assert_array_equal(output["prong_event_index"][:], [0, 1, 1, 2])
        assert output.attrs["num_events"] == 3
    with pytest.raises(FileExistsError):
        export_predictions(DummyModel(), batches, path)
