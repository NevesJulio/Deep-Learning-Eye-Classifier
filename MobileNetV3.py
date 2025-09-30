from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from Config import train_and_validate, ImageDataset, CustomHead
from Metric import plot_history
import pandas as pd
from torch.optim.lr_scheduler import OneCycleLR



DATA_DIR = "Data/DataSet"
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda:1"
MODEL_NAME = "mobilenetv3_large"
n_runs = 1
histories = {}
num_classes = 2
DROPOUT_RATE = 0.4



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

    # Backbone MobileNetV3  
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)  

    # Congela todas as camadas exceto o último bloco  
    for name, param in model.named_parameters():  
        if "classifier" not in name:  
            param.requires_grad = False  

    # Substitui o classificador padrão pelo CustomHead  
    in_features = model.classifier[0].in_features  
    model.classifier = CustomHead(in_features, num_classes, DROPOUT_RATE)  
    model = model.to(DEVICE)  

    # Loss, otimizador e scheduler  
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(  
        filter(lambda p: p.requires_grad, model.parameters()),  
        lr=LR,  
        weight_decay=0.02  
    )  
    scheduler = OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)  

    # Treinamento e validação  
    history = train_and_validate(  
        model=model,  
        train_loader=train_loader,  
        val_loader=val_loader,  
        criterion=criterion,  
        optimizer=optimizer,  
        scheduler=scheduler,  
        device=DEVICE,  
        epochs=EPOCHS,  
        save_path=f"{MODEL_NAME}_run{i}.pth",  
        num_classes=num_classes  
    )  

    histories[f"run_{i}"] = history  

    # Salva histórico  
    torch.save(history, f"history_{MODEL_NAME}_run{i}.pt")  

    df = pd.DataFrame(history)  
    csv_path = f"{MODEL_NAME}_run{i}_history.csv"  
    df.to_csv(csv_path, index=False)  
    print(f"Histórico salvo em CSV em {csv_path}")  

    plot_history(history, model_name=f"{MODEL_NAME}_run{i}")  

