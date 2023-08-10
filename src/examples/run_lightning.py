import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from functools import partial
from pytorch_lightning.loggers.wandb import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


# Data Preprocessing
def preprocess(batch, tokenizer):
    sentences = [case["text"] for case in batch]
    labels = torch.LongTensor([case["label"] for case in batch])

    encoded_sent = tokenizer(
        sentences,
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    out = {}
    out["input_ids"] = encoded_sent["input_ids"].cuda()
    out["attention_mask"] = encoded_sent["attention_mask"].cuda()
    out["label"] = labels
    return out


# Lightning Module definition
class TextClassifier(pl.LightningModule):
    def __init__(self, lr=2e-5, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.num_classes = 2
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-cased", num_labels=self.num_classes
        )
        self.predictions = []
        self.references = []

    def forward(self, batch):
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits

    def training_step(self, batch, batch_idx):
        labels = batch["label"]
        logits = self.forward(batch)
        loss = F.cross_entropy(logits.view(-1, self.num_classes), labels)
        self.log("train_loss", loss)
        return loss

    def on_validation_epoch_start(self):
        self.predictions.clear()
        self.references.clear()

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        logits = self.forward(batch)
        preds = torch.argmax(logits, dim=1)
        self.predictions.append(preds)
        self.references.append(labels)

    def on_validation_epoch_end(self):
        predictions = torch.concat(self.predictions).view(-1)
        references = torch.concat(self.references).view(-1)
        accuracy = (predictions == references).sum() / len(predictions)
        self.log_dict({"accuracy": accuracy.item()}, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, eps=self.eps)


if __name__ == "__main__":
    # Prepare Data
    hf_ds = load_dataset("tweet_eval", "irony")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    collate_fn = partial(preprocess, tokenizer=tokenizer)
    train_dataloader = DataLoader(hf_ds["train"], batch_size=32, collate_fn=collate_fn)
    val_dataloader = DataLoader(hf_ds["test"], batch_size=32, collate_fn=collate_fn)

    ## Experiment tracking
    wandb_logger = WandbLogger(
        name="test",
        project="ray-train-cuj",
        id="unique_id",
        save_dir=f"./wandb",
        offline=True,
    )

    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        save_on_train_epoch_end=False, monitor="accuracy", mode="max", save_top_k=1
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=3,
        accelerator="cpu",
        devices="auto",
        log_every_n_steps=1,
    )

    model = TextClassifier(lr=5e-5, eps=1e-8)

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
