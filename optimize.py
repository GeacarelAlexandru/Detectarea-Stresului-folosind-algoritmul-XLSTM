import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from main import *

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = StressDetector(VOCAB_SIZE, EMBED_DIM, hidden_dim, OUTPUT_DIM).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    EPOCHS = 3
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            predictions = model(input_ids)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            predictions = model(input_ids)
            pred_labels = (predictions > 0).float()
            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)
            
    accuracy = correct / total
    return accuracy
print("Starting Evolutionary Optimization...")


sampler = optuna.samplers.NSGAIIISampler()

study = optuna.create_study(direction="maximize", sampler=sampler)

study.optimize(objective, n_trials=20)

print("\n Optimization Finished!")
print(f"Best Accuracy achieved: {study.best_value * 100:.2f}%")
print("Best Hyperparameters found:", study.best_params)