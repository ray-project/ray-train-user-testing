import os
import wandb
import mlflow
import torch
import torch.nn.functional as F
from datasets import load_dataset
from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


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
    out["label"] = labels.cuda()
    return out


if __name__ == "__main__":
    # Prepare Data
    hf_ds = load_dataset("tweet_eval", "irony")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    collate_fn = partial(preprocess, tokenizer=tokenizer)
    train_dataloader = DataLoader(hf_ds["train"], batch_size=32, collate_fn=collate_fn)
    val_dataloader = DataLoader(hf_ds["test"], batch_size=32, collate_fn=collate_fn)

    configs = {
        "lr": 2e-4,
        "eps": 1e-8,
        "num_labels": 2,
        "num_epochs": 4,
        "checkpoint_dir": "./exp_results/torch_checkpoints",
    }

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=configs["num_labels"]
    ).to("cuda")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=configs["lr"], eps=configs["eps"]
    )

    global_steps = 0
    for epoch in range(configs["num_epochs"]):
        # Training
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Train Epoch {epoch}"):
            input_ids, attention_mask, labels = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["label"],
            )
            outputs = model(input_ids, attention_mask=attention_mask)
            optimizer.zero_grad()
            logits = outputs.logits
            loss = F.cross_entropy(logits.view(-1, configs["num_labels"]), labels)
            loss.backward()
            optimizer.step()
            global_steps += 1

        # Evaluation
        predictions = []
        references = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch}"):
                input_ids, attention_mask, labels = (
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["label"],
                )
                outputs = model(input_ids, attention_mask=attention_mask)
                labels = batch["label"]
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.append(preds)
                references.append(labels)

        predictions = torch.concat(predictions).view(-1)
        references = torch.concat(references).view(-1)
        accuracy = (predictions == references).sum() / len(predictions)
        print(f"Epoch {epoch}: Evaluation accuracy = {accuracy}.")

        # Saving checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "accuracy": accuracy,
        }

        os.makedirs(configs["checkpoint_dir"], exist_ok=True)
        torch.save(
            checkpoint, os.path.join(configs["checkpoint_dir"], f"ckpt_{epoch}.pth")
        )
