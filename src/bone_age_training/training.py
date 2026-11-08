"""Lightning trainer with explicit iteration-based distributed sampling."""

import json
import math
import os
from dataclasses import replace

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from torchmetrics import MeanAbsoluteError

from .assets import data_identity
from .config import TrainConfig
from .data import BoneAgeDataset, check_manifest
from .model import BoneAgeNet, double_mae, expected_age
from .schedule import LinearWarmupCosineAnnealingLR


class IterationSampler(Sampler):
    """Shuffle repeated dataset passes, then shard an exact number of global samples."""

    def __init__(self, size, samples_per_rank, rank=0, world_size=1, seed=42):
        if size <= 0 or samples_per_rank <= 0 or not 0 <= rank < world_size:
            raise ValueError("Invalid sampler dimensions")
        self.size, self.samples = size, samples_per_rank
        self.rank, self.world_size, self.seed, self.epoch = rank, world_size, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.samples

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        total = self.samples * self.world_size
        indices = torch.cat(
            [
                torch.randperm(self.size, generator=generator)
                for _ in range(math.ceil(total / self.size))
            ]
        )[:total]
        return iter(indices[self.rank :: self.world_size].tolist())


class BoneAgeModule(L.LightningModule):
    def __init__(self, config, initialize_pretrained=True):
        super().__init__()
        self.save_hyperparameters({"config": config})
        self.recipe = TrainConfig(**config)
        initialization = (
            self.recipe if initialize_pretrained else replace(self.recipe, pretrained=False)
        )
        self.model = BoneAgeNet(initialization)
        self.mae_reg, self.mae_cls = MeanAbsoluteError(), MeanAbsoluteError()

    def training_step(self, batch, batch_idx):
        reg, logits = self.model(batch["image"])
        loss = double_mae(reg, logits, batch["age"])
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, batch_size=batch["image"].shape[0]
        )
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"], on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        reg, logits = self.model(batch["image"])
        self.mae_reg.update(reg.float(), batch["age"].float())
        self.mae_cls.update(expected_age(logits), batch["age"].float())
        self.log("val_mae_reg", self.mae_reg, on_epoch=True, prog_bar=True)
        self.log("val_mae_cls", self.mae_cls, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        c = self.recipe
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=c.learning_rate,
            weight_decay=c.weight_decay,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            max_lr=c.learning_rate,
            total_steps=c.epochs * c.steps_per_epoch,
            pct_start=c.warmup_fraction,
            div_factor=c.div_factor,
            final_div_factor=c.final_div_factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_train_epoch_start(self):
        self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["training_fold"] = self.trainer.datamodule.fold
        checkpoint["training_data"] = self.trainer.datamodule.identity


class BoneAgeDataModule(L.LightningDataModule):
    def __init__(self, config, manifest, data_root, reference, fold):
        super().__init__()
        self.config, self.manifest, self.data_root = config, manifest, data_root
        self.reference, self.fold = reference, fold
        self.identity = data_identity(manifest, reference)

    def setup(self, stage=None):
        c = self.config
        self.training = BoneAgeDataset(
            self.manifest,
            self.data_root,
            self.reference,
            self.fold,
            True,
            c.image_size,
        )
        self.validation = BoneAgeDataset(
            self.manifest,
            self.data_root,
            self.reference,
            self.fold,
            False,
            c.image_size,
        )

    def train_dataloader(self):
        c = self.config
        sampler = IterationSampler(
            len(self.training),
            c.batch_size * c.steps_per_epoch,
            self.trainer.global_rank,
            self.trainer.world_size,
            c.seed,
        )
        return DataLoader(
            self.training,
            batch_size=c.batch_size,
            sampler=sampler,
            num_workers=c.workers,
            pin_memory=True,
            persistent_workers=c.workers > 0,
        )

    def val_dataloader(self):
        c = self.config
        sampler = DistributedSampler(
            self.validation,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
            shuffle=False,
        )
        return DataLoader(
            self.validation,
            batch_size=c.batch_size * 2,
            sampler=sampler,
            num_workers=c.workers,
            pin_memory=True,
            persistent_workers=c.workers > 0,
        )


def train(
    config,
    manifest,
    data_root,
    reference,
    fold,
    devices,
    accelerator,
    output,
    limit_val_batches=1.0,
    resume=None,
):
    config.validate()
    if fold not in (0, 1, 2, 3, 4):
        raise ValueError("fold must be between 0 and 4")
    if devices < 1:
        raise ValueError("devices must be positive")
    check_manifest(manifest, data_root)
    if not reference.is_file():
        raise FileNotFoundError(reference)
    if accelerator == "cpu" and devices != 1:
        raise ValueError("Use one CPU device; distributed training is configured for GPUs")
    if resume is not None:
        previous = torch.load(resume, map_location="cpu", weights_only=True)
        if previous.get("training_fold") != fold:
            raise ValueError("Resume checkpoint belongs to another fold")
        if previous["hyper_parameters"]["config"] != config.to_dict():
            raise ValueError("Resume requires the same effective training configuration")
        if previous.get("training_data") != data_identity(manifest, reference):
            raise ValueError("Resume requires the same training manifest and histogram reference")
    launcher_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if launcher_rank == 0 and output.exists() and resume is None and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory or --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    L.seed_everything(config.seed, workers=True)
    effective = {
        "config": config.to_dict(),
        "fold": fold,
        "manifest": str(manifest.resolve()),
        "data_root": str(data_root.resolve()),
        "reference": str(reference.resolve()),
        "devices": devices,
        "accelerator": accelerator,
        "training_data": data_identity(manifest, reference),
    }
    if launcher_rank == 0:
        (output / "effective_config.json").write_text(json.dumps(effective, indent=2) + "\n")
    checkpoint = ModelCheckpoint(
        dirpath=output,
        filename="best",
        monitor="val_mae_reg",
        mode="min",
        save_last=True,
        save_top_k=1,
        save_weights_only=False,
    )
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy="ddp" if devices > 1 else "auto",
        max_epochs=config.epochs,
        precision="32-true" if accelerator == "cpu" else config.precision,
        sync_batchnorm=config.sync_batchnorm and devices > 1,
        benchmark=True,
        use_distributed_sampler=False,
        logger=CSVLogger(str(output), name="logs"),
        callbacks=[checkpoint],
        log_every_n_steps=min(50, config.steps_per_epoch),
        limit_val_batches=limit_val_batches,
    )
    trainer.fit(
        BoneAgeModule(config.to_dict(), initialize_pretrained=resume is None),
        datamodule=BoneAgeDataModule(
            config,
            manifest,
            data_root,
            reference,
            fold,
        ),
        ckpt_path=resume,
    )
