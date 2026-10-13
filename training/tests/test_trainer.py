"""Tests for the training loop: metrics, clipping, best-checkpoint selection."""

from __future__ import annotations

import pytest
import torch

from src.data.dataset import LabelMap
from src.trainer import CHECKPOINT_FILE, Trainer, resolve_device, set_seed


@pytest.fixture
def loaders(random_batch):
    """Three fixed batches reused as both train and val data."""
    return [random_batch(batch_size=4, seq_len=16) for _ in range(3)]


def make_trainer(model, loaders, tmp_path=None, lr=1e-3, scheduler=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    config = {"max_grad_norm": 1.0, "label_map": LabelMap()}
    if tmp_path is not None:
        config["checkpoint_dir"] = str(tmp_path)
    return Trainer(
        model=model,
        train_loader=loaders,
        val_loader=loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        config=config,
    )


# ---- helpers ---------------------------------------------------------- #


def test_set_seed_makes_torch_deterministic():
    set_seed(7)
    a = torch.randn(5)
    set_seed(7)
    assert torch.allclose(a, torch.randn(5))


def test_resolve_device_defaults_to_cpu_without_cuda():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device().type in ("cpu", "cuda")


# ---- train_epoch / eval_epoch ----------------------------------------- #


def test_train_epoch_returns_the_documented_keys(tiny_model, loaders):
    metrics = make_trainer(tiny_model, loaders).train_epoch()
    assert set(metrics) == {"loss", "accuracy", "macro_f1"}
    assert metrics["loss"] > 0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_train_epoch_updates_the_weights(tiny_model, loaders):
    before = tiny_model.classifier[-1].weight.detach().clone()
    make_trainer(tiny_model, loaders).train_epoch()
    assert not torch.allclose(before, tiny_model.classifier[-1].weight)


def test_eval_epoch_returns_the_documented_keys(tiny_model, loaders):
    metrics = make_trainer(tiny_model, loaders).eval_epoch()
    for key in ("loss", "accuracy", "macro_f1", "per_class_f1"):
        assert key in metrics, key
    assert set(metrics["per_class_f1"]) == set(LabelMap().labels)


def test_eval_epoch_leaves_the_weights_untouched(tiny_model, loaders):
    before = tiny_model.classifier[-1].weight.detach().clone()
    make_trainer(tiny_model, loaders).eval_epoch()
    assert torch.allclose(before, tiny_model.classifier[-1].weight)


def test_eval_epoch_restores_nothing_but_still_sets_eval_mode(tiny_model, loaders):
    trainer = make_trainer(tiny_model, loaders)
    trainer.model.train()
    trainer.eval_epoch()
    assert not trainer.model.training


def test_eval_epoch_accepts_an_explicit_loader(tiny_model, loaders, random_batch):
    other = [random_batch(batch_size=2, seq_len=16)]
    assert make_trainer(tiny_model, loaders).eval_epoch(other)["loss"] > 0


def test_eval_epoch_rejects_unlabelled_batches(tiny_model, random_batch):
    unlabelled = [random_batch(with_labels=False)]
    with pytest.raises(ValueError, match="needs labelled batches"):
        make_trainer(tiny_model, unlabelled).eval_epoch()


# ---- fit -------------------------------------------------------------- #


def test_fit_returns_one_record_per_epoch(tiny_model, loaders):
    history = make_trainer(tiny_model, loaders).fit(3)
    assert len(history) == 3
    assert [row["epoch"] for row in history] == [1, 2, 3]
    for row in history:
        for key in ("train_loss", "train_accuracy", "train_macro_f1",
                    "val_loss", "val_accuracy", "val_macro_f1", "lr"):
            assert key in row, key


def test_fit_rejects_zero_epochs(tiny_model, loaders):
    with pytest.raises(ValueError, match="num_epochs"):
        make_trainer(tiny_model, loaders).fit(0)


def test_fit_tracks_the_best_epoch_by_val_macro_f1(tiny_model, loaders):
    trainer = make_trainer(tiny_model, loaders)
    history = trainer.fit(3)
    best = max(history, key=lambda row: row["val_macro_f1"])
    assert trainer.best_epoch == best["epoch"]
    assert trainer.best_macro_f1 == pytest.approx(best["val_macro_f1"])


def test_fit_leaves_the_model_holding_the_best_weights(tiny_model, loaders):
    """Not the last epoch's -- the restore is the point of tracking a best."""
    trainer = make_trainer(tiny_model, loaders)
    trainer.fit(3)
    for key, value in trainer.best_state.items():
        assert torch.allclose(trainer.model.state_dict()[key].cpu(), value), key


def test_fit_writes_a_checkpoint_when_a_dir_is_configured(tiny_model, loaders, tmp_path):
    trainer = make_trainer(tiny_model, loaders, tmp_path=tmp_path)
    trainer.fit(2)
    checkpoint_path = tmp_path / CHECKPOINT_FILE
    assert checkpoint_path.is_file()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == trainer.best_epoch
    assert checkpoint["val_macro_f1"] == pytest.approx(trainer.best_macro_f1)
    assert set(checkpoint["state_dict"]) == set(tiny_model.state_dict())


def test_fit_writes_no_checkpoint_without_a_dir(tiny_model, loaders, tmp_path):
    make_trainer(tiny_model, loaders).fit(1)
    assert not (tmp_path / CHECKPOINT_FILE).exists()


def test_fit_steps_the_scheduler_once_per_batch(tiny_model, loaders):
    calls = {"n": 0}

    class CountingScheduler:
        def step(self):
            calls["n"] += 1

    make_trainer(tiny_model, loaders, scheduler=CountingScheduler()).fit(2)
    assert calls["n"] == 2 * len(loaders)


def test_fit_records_the_per_class_f1_each_epoch(tiny_model, loaders):
    history = make_trainer(tiny_model, loaders).fit(1)
    assert set(history[0]["val_per_class_f1"]) == set(LabelMap().labels)


def test_fit_stores_the_best_validation_metrics(tiny_model, loaders):
    trainer = make_trainer(tiny_model, loaders)
    trainer.fit(2)
    metrics = trainer.best_val_metrics
    assert metrics["evaluation_split"] == "val"
    assert metrics["n"] == 12
    assert "confusion_matrix" in metrics


# ---- gradient clipping ------------------------------------------------ #


def test_gradient_clipping_is_applied(tiny_model, loaders, monkeypatch):
    """A huge LR would diverge without clipping; assert the call happens."""
    seen: list[float] = []
    original = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm, *args, **kwargs):
        seen.append(max_norm)
        return original(params, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    make_trainer(tiny_model, loaders).train_epoch()

    assert seen == [1.0] * len(loaders)


def test_clipping_can_be_disabled(tiny_model, loaders, monkeypatch):
    seen: list[float] = []
    monkeypatch.setattr(
        torch.nn.utils, "clip_grad_norm_", lambda p, n, *a, **k: seen.append(n)
    )
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    Trainer(
        model=tiny_model, train_loader=loaders, val_loader=loaders,
        optimizer=optimizer, device="cpu",
        config={"max_grad_norm": 0, "label_map": LabelMap()},
    ).train_epoch()
    assert seen == []


def test_clipping_keeps_gradients_within_the_norm(tiny_model, loaders):
    """With a pathological LR the loop must still produce finite weights."""
    trainer = make_trainer(tiny_model, loaders, lr=5.0)
    trainer.fit(1)
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())


# ---- learning ---------------------------------------------------------- #


def test_the_loop_can_fit_a_tiny_batch(tiny_model, random_batch):
    """Overfit one batch: if loss does not fall, the wiring is broken."""
    batch = random_batch(batch_size=4, seq_len=16)
    trainer = make_trainer(tiny_model, [batch], lr=5e-3)
    first = trainer.train_epoch()["loss"]
    for _ in range(25):
        last = trainer.train_epoch()["loss"]
    assert last < first
