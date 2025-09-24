from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from Config import train_and_validate, ImageDataset
from Metric import plot_history
import pandas as pd


DATA_DIR = "Data/DataSet"
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda:1"
MODEL_NAME = "resnet50"
n_runs = 5
histories = {}


default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


dataset = ImageDataset(DATA_DIR, default_transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)





for i in range(n_runs):
    print(f"\n===== Run {i+1}/{n_runs} =====")

    
    num_classes = len(dataset.class_to_idx)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Treina e retorna histórico
    history = train_and_validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        epochs=EPOCHS,
        save_path=f"{MODEL_NAME}_run{i}.pth",
        num_classes=num_classes
    )

    histories[f"run_{i}"] = history

    # Salva histórico em .pt
    torch.save(history, f"history_{MODEL_NAME}_run{i}.pt")

    # Salva histórico em CSV
    df = pd.DataFrame(history)
    csv_path = f"{MODEL_NAME}_run{i}_history.csv"
    df.to_csv(csv_path, index=False)
    print(f"Histórico salvo em CSV em {csv_path}")

    # Plota e salva gráficos
    plot_history(history, model_name=f"{MODEL_NAME}_run{i}")
