import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
# Asigură-te că ai instalat: pip install xlstm
from xlstm import xLSTMStack, xLSTMStackConfig

# 1. Încărcare și Filtrare Date
df = pd.read_csv('dreaddit-train.csv')

# Filtrare: păstrăm doar datele cu încredere ridicată
initial_count = len(df)
df = df[df['confidence'] > 0.6]
print(f"Am eliminat {initial_count - len(df)} rânduri cu încredere scăzută.")

# Păstrăm coloanele necesare
df = df[['text', 'label']].reset_index(drop=True)

# 2. Configurare Model și Tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class StressDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.values
        self.labels = labels.values
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

# Pregătire Dataloaders
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.15, random_state=42)

train_loader = DataLoader(StressDataset(X_train, y_train, tokenizer), batch_size=16, shuffle=True)
test_loader = DataLoader(StressDataset(X_test, y_test, tokenizer), batch_size=16)

# 3. Arhitectura xLSTM

class StressXLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        
        # Configurație pentru sLSTM (versiune scalabilă a LSTM)
        cfg = xLSTMStackConfig(
            context_length=128,
            embedding_dim=emb_dim,
            num_layers=2,
            slstm_config={'backend': 'vanilla', 'num_heads': 4}
        )
        self.xlstm = xLSTMStack(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.xlstm(x)
        # Pooling: luăm media reprezentărilor pentru tot textul
        x = torch.mean(x, dim=1) 
        return self.classifier(x)

model = StressXLSTM(len(tokenizer.vocab)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 4. Bucla de Antrenare cu validare
def train_model(epochs=5):
    for epoch in range(epochs):
        model.train()
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
        
        # Evaluare rapidă
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in test_loader:
                ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                preds = model(ids).argmax(dim=1)
                correct += (preds == labels).sum().item()
        
        accuracy = correct / len(X_test)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Acc: {accuracy:.2%}")

train_model()