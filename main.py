import pandas as pd 
import re
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# Clean the text data
def clean_text(text):
    text = str(text).lower()  
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load the dataset
df_train = pd.read_csv('dreaddit-train.csv')
print("distributia:")
print(df_train['label'].value_counts())
df_test = pd.read_csv('dreaddit-test.csv')

df_train = df_train.dropna(subset=['text', 'label'])
df_test = df_test.dropna(subset=['text', 'label'])

df_train['text'] = df_train['text'].apply(clean_text)
df_test['text'] = df_test['text'].apply(clean_text)

# Class Dataset
from torch.utils.data import Dataset
import torch

class StressDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

        cleaned_text = clean_text(text)

        encoding = self.tokenizer(
            cleaned_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

    
    
TOKENIZER_NAME = 'bert-base-uncased'
MAX_LENGTH = 128
BATCH_SIZE = 16

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

train_dataset = StressDataset(df_train['text'], df_train['label'], tokenizer, MAX_LENGTH)
test_dataset = StressDataset(df_test['text'], df_test['label'], tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Set de antrenament: {len(train_dataset)} texte.")
print(f"Set de testare: {len(test_dataset)} texte.")

try:
    from xlstm.xlstm_large.model import xLSTMLargeConfig,xLSTMLarge
    HAS_XLSTM = False
except ImportError:
    print("Biblioteca xLSTM nu a fost gasita.")
    HAS_XLSTM = False

class StressDetector(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(StressDetector, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        
        if HAS_XLSTM:
            config = xLSTMLargeConfig(
                vocab_size=vocab_size,
                embedding_dim=embed_dim,
                num_heads=4,
                num_blocks=2
            )
            self.sequence_model = xLSTMLarge(config)
            
            fc_input_size = embed_dim
        else:
            self.sequence_model = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                batch_first=True
            )
            fc_input_size = hidden_dim 
            
        self.fc = nn.Linear(fc_input_size, output_dim)

    def forward(self, input_ids):
        if HAS_XLSTM:
            output = self.sequence_model(input_ids)
            if isinstance(output, tuple):
                output = output[0]
            last_hidden_state = torch.mean(output, dim=1)
        else: 
            x = self.embedding(input_ids)
            output, (hidden, cell) = self.sequence_model(x)
            last_hidden_state = torch.mean(output, dim=1)
            
        logits = self.fc(last_hidden_state)
        return logits.squeeze(-1)

VOCAB_SIZE = tokenizer.vocab_size
EMBED_DIM = 128
HIDDEN_DIM = 256 
OUTPUT_DIM = 1 
model = StressDetector(VOCAB_SIZE,EMBED_DIM,HIDDEN_DIM,OUTPUT_DIM)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"Modelul a fost initializat si mutat pe {device}")

EPOCHS = 5 
LEARNING_RATE = 1e-3

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr = LEARNING_RATE)

print("\n Incepe antrenarea modelului")

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_loader, desc = f'Epoca {epoch + 1}/{EPOCHS}')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        predictions = model(input_ids)
        loss = criterion(predictions,labels)
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix({'Loss':f'{loss.item():.4f}'})

    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_test_loss = 0 
    corecte = 0
    total_texte = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            predictions = model(input_ids)
            loss = criterion(predictions, labels)
            total_test_loss += loss.item()

            pred_labels = (predictions > 0).float()
            corecte += (pred_labels == labels).sum().item()
            total_texte += labels.size(0)
    
    avg_test_loss = total_test_loss / len(test_loader)
    acuratete = (corecte / total_texte) * 100

    print(f"\n Rezumat Epoca {epoch + 1}:")
    print(f"   - Eroare Antrenare (Loss): {avg_train_loss:.4f}")
    print(f"   - Eroare Testare (Loss):   {avg_test_loss:.4f}")
    print(f"   - Acuratete pe Testare:    {acuratete:.2f}%\n")    

torch.save(model.state_dict(), 'stress_detector_model.pth')
print("Antrenare completa, modelul a fost salvat.")
           
def predict_stress(text, model, tokenizer, device, max_len=128):
    model.eval() 
    
    cleaned_text = clean_text(text)
    
    encoding = tokenizer(
        cleaned_text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids)
        probability = torch.sigmoid(logits).item()
        
    prediction = 1 if probability > 0.5 else 0
    return prediction, probability

print(" TESTEAZĂ DETECTORUL DE STRES ")
print("Scrie 'exit' pentru a opri programul.")

while True:
    user_input = input("Scrie un text in engleza: ")
    
    if user_input.lower() == 'exit':
        print("Oprire program. O zi fara stres! ")
        break
        
    if not user_input.strip():
        print("Te rog sa scrii un text valid.")
        continue
        
    pred, prob = predict_stress(user_input, model, tokenizer, device, MAX_LENGTH)
    
    if pred == 1:
        print(f" Rezultat: STRES (Probabilitate: {prob*100:.1f}%)")
    else:
        print(f" Rezultat: FARA STRES / RELAXAT (Probabilitate: {(1-prob)*100:.1f}%)")
    