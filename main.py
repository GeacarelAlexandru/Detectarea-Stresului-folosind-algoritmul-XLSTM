import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# Importurile corecte pentru librăria xLSTM
try:
    from xlstm import (
        xLSTMBlockStack,
        xLSTMBlockStackConfig,
        mLSTMBlockConfig,
        mLSTMLayerConfig,
        sLSTMBlockConfig,
        sLSTMLayerConfig,
        FeedForwardConfig
    )
except ImportError:
    print("Eroare: Asigură-te că ai instalat xlstm corect (pip install xlstm)")

# 1. Încărcare și Filtrare Date
# Citim fișierul și filtrăm după încredere (confidence > 0.6)
df = pd.read_csv('dreaddit-train.csv')
df = df[df['confidence'] > 0.6].reset_index(drop=True)
print(f"Dataset filtrat: {len(df)} rânduri rămase (unde confidence > 0.6).")

# 2. Dataset și Preprocesare
class StressDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        encoding = self.tokenizer.encode_plus(
            str(self.texts[item]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'label': torch.tensor(self.labels[item], dtype=torch.long)
        }

# Setări model
MAX_LEN = 128
BATCH_SIZE = 16
EMB_DIM = 128

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.15, random_state=42)

train_loader = DataLoader(StressDataset(X_train.values, y_train.values, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(StressDataset(X_test.values, y_test.values, tokenizer, MAX_LEN), batch_size=BATCH_SIZE)

# 3. Definirea Arhitecturii xLSTM
class StressXLSTMModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMB_DIM)
        
        # Configurare xLSTM (folosind sLSTM pentru performanță pe text)
        cfg = xLSTMBlockStackConfig(
            mlstm_block=None, # Putem folosi doar sLSTM pentru început
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla", # Folosim vanilla dacă nu ai setup de nuclee Triton/CUDA complexe
                    num_heads=4,
                    conv1d_kernel_size=4
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu")
            ),
            context_length=MAX_LEN,
            num_blocks=2,
            embedding_dim=EMB_DIM,
            slstm_at=[0, 1] # Aplicăm sLSTM pe ambele layere
        )
        
        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(EMB_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2) # 0 = Fără stres, 1 = Stres
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.xlstm_stack(x)
        # Global Average Pooling (media pe toată secvența de text)
        x = torch.mean(x, dim=1)
        return self.classifier(x)

# Inițializare model și optimizator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StressXLSTMModel(len(tokenizer.vocab)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 4. Antrenare
def train():
    model.train()
    for epoch in range(3): # 3 epoci sunt suficiente pentru un test inițial
        total_loss = 0
        for batch in train_loader:
            ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoca {epoch+1} finalizată. Loss mediu: {total_loss/len(train_loader):.4f}")

    # Validare finală
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            preds = model(ids).argmax(dim=1)
            correct += (preds == labels).sum().item()
    
    print(f"Acuratețe pe setul de test: {correct/len(X_test):.2%}")

if __name__ == "__main__":
    train()