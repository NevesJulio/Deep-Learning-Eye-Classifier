import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_history(history, model_name="modelo", save_dir="plots"):
    """
    Gera gráficos de métricas a partir do dicionário `history` retornado pelo treino
    e salva os gráficos em arquivos PNG.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # --- Loss ---
    plt.figure(figsize=(10,5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title(f"Loss - {model_name}")
    plt.legend()
    plt.savefig(save_path / f"{model_name}_loss.png")
    plt.show()

    # --- Acurácia ---
    plt.figure(figsize=(10,5))
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.title(f"Acurácia - {model_name}")
    plt.legend()
    plt.savefig(save_path / f"{model_name}_accuracy.png")
    plt.show()

    # --- Precisão / Recall / F1 ---
    plt.figure(figsize=(10,5))
    plt.plot(history["precision"], label="Precisão")
    plt.plot(history["recall"], label="Recall")
    plt.plot(history["f1"], label="F1-score")
    plt.xlabel("Época")
    plt.ylabel("Score")
    plt.title(f"Métricas - {model_name}")
    plt.legend()
    plt.savefig(save_path / f"{model_name}_metrics.png")
    plt.show()

    # --- ROC final ---
    if any(history["roc_curves"]):
        # pega a última curva (última época)
        last_fpr, last_tpr = history["roc_curves"][-1]
        plt.figure(figsize=(6,6))
        plt.plot(last_fpr, last_tpr, label="ROC final")
        plt.plot([0,1], [0,1], 'k--', label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Curva ROC final - {model_name}")
        plt.legend()
        plt.savefig(save_path / f"{model_name}_roc_curve.png")
        plt.show()                                                                                  

    # --- Matriz de confusão ---
    if "val_labels" in history and "val_preds" in history:
        cm = confusion_matrix(history["val_labels"], history["val_preds"])
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Matriz de Confusão - {model_name}")
        plt.savefig(save_path / f"{model_name}_confusion_matrix.png")
        plt.show()

    print(f"Todos os gráficos salvos na pasta: {save_path.resolve()}")
