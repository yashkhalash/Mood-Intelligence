import os
import sys
from pathlib import Path
import json
from datetime import datetime

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Ensure project root is on sys.path when running as `python3 scripts/train_rafdb.py`
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mood_detector.model import EmotionNet, ModelConfig


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _compute_class_weights(targets: list[int], num_classes: int) -> torch.Tensor:
    # Inverse-frequency weights (simple + effective for RAF-DB imbalance).
    counts = torch.bincount(torch.tensor(targets, dtype=torch.int64), minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = (counts.sum() / counts) / float(num_classes)
    return weights


def _freeze_backbone(model: EmotionNet) -> None:
    # Freeze everything, then unfreeze classifier head.
    for p in model.parameters():
        p.requires_grad = False

    # torchvision MobileNetV2 and EfficientNet both use `.classifier`
    if hasattr(model.backbone, "classifier"):
        for p in model.backbone.classifier.parameters():
            p.requires_grad = True


def _unfreeze_backbone_last_blocks(model: EmotionNet, *, last_n: int) -> None:
    # Keep classifier trainable
    if hasattr(model.backbone, "classifier"):
        for p in model.backbone.classifier.parameters():
            p.requires_grad = True

    # Unfreeze last N blocks of feature extractor (MobileNetV2 has `.features`).
    features = getattr(model.backbone, "features", None)
    if features is None:
        # EfficientNet-B0 in torchvision uses `.features` too; if not found, unfreeze all.
        for p in model.parameters():
            p.requires_grad = True
        return

    blocks = list(features.children())
    if last_n <= 0:
        return
    for block in blocks[-last_n:]:
        for p in block.parameters():
            p.requires_grad = True


def main():
    base_dir = os.environ.get("RAF_DB_DIR", "/home/ns-44/Desktop/Mood Detector/raf_db_extracted")
    train_dir = os.path.join(base_dir, "train_data")
    val_dir = os.path.join(base_dir, "test_data")

    os.makedirs("models", exist_ok=True)
    weights_out = os.path.join("models", "raf_db_emotion_model.pth")

    img_size = (224, 224)
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))
    epochs = int(os.environ.get("EPOCHS", "30"))
    freeze_epochs = int(os.environ.get("FREEZE_EPOCHS", "8"))
    unfreeze_last_blocks = int(os.environ.get("UNFREEZE_LAST_BLOCKS", "4"))
    lr_head = float(os.environ.get("LR_HEAD", "0.001"))
    lr_finetune = float(os.environ.get("LR_FINETUNE", "0.00005"))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "0.01"))
    seed = int(os.environ.get("SEED", "42"))
    log_every = int(os.environ.get("LOG_EVERY", "100"))
    num_workers = int(os.environ.get("NUM_WORKERS", str(min(4, os.cpu_count() or 1))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _set_seed(seed)

    print("=== RAF-DB Training (PyTorch) ===")
    print(f"base_dir={base_dir}")
    print(f"train_dir={train_dir}")
    print(f"val_dir={val_dir}")
    print(f"device={device}")
    print(
        f"epochs={epochs} freeze_epochs={freeze_epochs} unfreeze_last_blocks={unfreeze_last_blocks} "
        f"batch_size={batch_size} num_workers={num_workers}"
    )
    print(f"lr_head={lr_head} lr_finetune={lr_finetune} weight_decay={weight_decay} seed={seed}")

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15)], p=0.7),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("Loading datasets...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    print(f"train_images={len(train_dataset)} val_images={len(val_dataset)} classes={train_dataset.classes}")

    # Save label mapping used by ImageFolder
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    with open("label_map.json", "w") as f:
        json.dump(idx_to_class, f)

    num_classes = len(train_dataset.classes)
    class_weights = _compute_class_weights(train_dataset.targets, num_classes=num_classes).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    cfg = ModelConfig(model_type=os.environ.get("MODEL_TYPE", "mobilenet_v2"), num_classes=num_classes)
    print(f"Building model: {cfg.model_type} (num_classes={cfg.num_classes})")
    model = EmotionNet(cfg).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Stage 1: train only the classification head (faster convergence + less overfitting).
    _freeze_backbone(model)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_head, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, freeze_epochs))

    best_val_acc = 0.0
    for epoch in range(epochs):
        # Switch to fine-tuning after head-only warmup.
        if epoch == freeze_epochs:
            _unfreeze_backbone_last_blocks(model, last_n=unfreeze_last_blocks)
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr_finetune,
                weight_decay=weight_decay,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - freeze_epochs))

        stage = "head" if epoch < freeze_epochs else "finetune"
        print(f"\n--- Epoch {epoch+1}/{epochs} ({stage}) ---")
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0

        for step, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            pred = logits.argmax(dim=1)
            total += labels.size(0)
            correct += int((pred == labels).sum().item())
            if log_every > 0 and (step % log_every == 0):
                train_acc_so_far = 100.0 * correct / max(total, 1)
                avg_loss = running_loss / float(step)
                print(f"step {step}/{len(train_loader)} - loss={avg_loss:.4f} acc={train_acc_so_far:.2f}%")

        train_acc = 100.0 * correct / max(total, 1)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += float(loss.item())
                pred = logits.argmax(dim=1)
                val_total += labels.size(0)
                val_correct += int((pred == labels).sum().item())

        val_acc = 100.0 * val_correct / max(val_total, 1)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs}: "
            f"loss={running_loss/max(1,len(train_loader)):.4f} train_acc={train_acc:.2f}% "
            f"val_loss={val_loss/max(1,len(val_loader)):.4f} val_acc={val_acc:.2f}%"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), weights_out)
            print(f"Saved best weights: {weights_out} (val_acc={val_acc:.2f}%)")

    metadata = {
        "validation_accuracy": round(best_val_acc, 2),
        "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": cfg.model_type,
        "framework": "pytorch",
        "classes": train_dataset.classes,
        "seed": seed,
        "epochs": epochs,
        "freeze_epochs": freeze_epochs,
        "unfreeze_last_blocks": unfreeze_last_blocks,
    }
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    main()

