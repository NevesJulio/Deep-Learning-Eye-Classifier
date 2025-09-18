import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch


#Definindo a classe do dataset
class ImageDataset(Dataset):
    
    def __init__(self, root_dir, transform = None):
        self.root_dir = Path(root_dir)
        self.transform = transform  # se não passar transform, usa padrão
        self.samples = []  # [(img_path, label_idx), ...]
        self.class_to_idx = {}  # {"classe": idx}

        # Cria lista de imagens + label
        for i, class_dir in enumerate(sorted(p for p in self.root_dir.iterdir() if p.is_dir())):
            self.class_to_idx[class_dir.name] = i
            for img_path in class_dir.glob('*'):
                self.samples.append((img_path, i))

    def __len__(self):
        return len(self.samples)

    def __str__(self):
        return f"ImageDataset with {len(self)} samples from {len(self.class_to_idx)} classes."


    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Abre a imagem de forma segura
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')  # garante 3 canais

        # Aplica transformações (pré-processamento/resnet normalization)
        img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)
    


#Função de treino e validação + tqdm

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, epochs, save_path):
    for epoch in range(1, epochs + 1):

        # ---- Treino ----
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False, colour="green")
        for images, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=f"{correct/total:.4f}")

        train_loss /= total
        train_acc = correct / total

        # ---- Validação ----
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        loop = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]  ", leave=False, colour="blue")
        with torch.no_grad():
            for images, labels in loop:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                loop.set_postfix(loss=loss.item(), acc=f"{correct/total:.4f}")

        val_loss /= total
        val_acc = correct / total

        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # ---- Salvar modelo ----
    torch.save(model.state_dict(), save_path)
    print(f"Modelo salvo em {save_path}")
