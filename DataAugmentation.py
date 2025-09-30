import os
from PIL import Image
from torchvision import transforms

# Caminhos
input_dir = "Data/DataSet"
output_dir = "Data/DataAugmentation"
os.makedirs(output_dir, exist_ok=True)

# Define data augmentation
augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
])

# Fator de aumento por classe (ex: {"normal": 3, "doente": 10})
augmentation_factors = {
    "classe1": 5,  # quantidade de cópias aumentadas por imagem
    "classe2": 10
}

for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    output_class_path = os.path.join(output_dir, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    factor = augmentation_factors.get(class_name, 1)  # se não tiver no dicionário, usa 1

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = Image.open(img_path).convert("RGB")

        # Salva a imagem original no novo dataset
        img.save(os.path.join(output_class_path, img_name))

        # Gera imagens aumentadas proporcionalmente
        for i in range(factor):
            aug_img = augmentations(img)
            aug_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"
            aug_img.save(os.path.join(output_class_path, aug_name))
