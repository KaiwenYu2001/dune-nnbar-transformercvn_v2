import h5py
import numpy as np
import pytorch_lightning as pl
import torch

from evaluate import export_predictions
from transformercvn.network.trainers.neutrino_full_dense_trainer import NeutrinoFullDenseTrainer
from transformercvn.options import Options


def test_synthetic_training_validation_and_export(tmp_path):
    torch.set_num_threads(1)
    data_path = tmp_path / "sample.h5"
    n = 24
    rng = np.random.default_rng(3)
    with h5py.File(data_path, "w") as f:
        f["features"] = rng.normal(size=(n, 2, 3)).astype("float32")
        f["extra"] = rng.normal(size=(n, 2)).astype("float32")
        f["prong_mask"] = np.ones((n, 2), dtype=bool)
        f["event_target"] = np.tile([0, 1, 2, 3], n // 4).astype("int64")
        f["prong_target"] = np.tile([0, 1], (n, 1)).astype("int64")
        f["full_pixels_shape"] = [1, 16, 16]
        for prefix, prongs in [("event", 1), ("prong", 2)]:
            offsets = np.arange(n + 1) * prongs
            f[f"{prefix}_compressed_index"] = np.stack([offsets[:-1], offsets[1:]], axis=1)
            f[f"{prefix}_pixels_shape"] = [prongs, 1, 16, 16]
            f[f"{prefix}_pixels_coordinates"] = np.tile(
                np.array([[p, 8, 8] for p in range(prongs)], dtype="int32"), (n, 1)
            )
            f[f"{prefix}_pixels_values"] = np.full((n * prongs, 1), 128, dtype="float32")
    options = Options(training_file=str(data_path))
    options.update_options(
        {
            "hidden_dim": 16,
            "initial_feature_dim": 8,
            "initial_pixel_dim": 8,
            "feature_embedding_dim": 8,
            "pixel_embedding_dim": 8,
            "position_embedding_dim": 8,
            "num_encoder_layers": 1,
            "num_attention_heads": 2,
            "num_prong_decoder_layers": 1,
            "densenet_structure": [1],
            "densenet_growth_rate": 4,
            "num_gpu": 0,
            "num_dataloader_workers": 0,
            "batch_size": 4,
            "train_validation_split": 0.75,
            "epochs": 1,
        }
    )
    model = NeutrinoFullDenseTrainer(options)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)
    assert trainer.global_step == 1
    assert torch.isfinite(trainer.callback_metrics["train_loss"])
    assert "val_epoch_nnbar_ovr_tpr" in trainer.callback_metrics
    loader_options = dict(model.dataloader_options, drop_last=False)
    loader = model.dataloader(model.validation_dataset, **loader_options)
    result = export_predictions(model, loader, tmp_path / "predictions.h5")
    with h5py.File(result) as f:
        np.testing.assert_array_equal(f["event_targets"][:], model.validation_dataset.event_targets)
        assert len(f["event_targets"]) == len(model.validation_dataset)
