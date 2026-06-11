import pandas as pd 
import re
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from xlstm_pure import Pure_xLSTM

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
BATCH_SIZE = 8

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

train_dataset = StressDataset(df_train['text'], df_train['label'], tokenizer, MAX_LENGTH)
test_dataset = StressDataset(df_test['text'], df_test['label'], tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Set de antrenament: {len(train_dataset)} texte.")
print(f"Set de testare: {len(test_dataset)} texte.")

HAS_XLSTM = True   
print("Algoritmul xLSTM a fost gasit!")

class StressDetector(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(StressDetector, self).__init__()

        # Stratul care transformă cuvintele în numere (comun pentru ambele)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        
        if HAS_XLSTM:
            # Folosim inovația NOASTRĂ: Pure_xLSTM
            self.sequence_model = Pure_xLSTM(
                input_size=embed_dim, 
                hidden_size=hidden_dim, 
                batch_first=True
            )
        else:
            # Folosim modelul clasic
            self.sequence_model = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                batch_first=True
            )
            
        # Ambele scot un output de dimensiunea hidden_dim
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids):
        # 1. Trecem cuvintele prin Embedding
        x = self.embedding(input_ids)
        
        # 2. Trecem datele prin modelul ales
        if HAS_XLSTM:
            output, _ = self.sequence_model(x)
        else: 
            output, (hidden, cell) = self.sequence_model(x)
            
        # 3. Calculăm maximul (Despachetăm direct în 2 variabile separate)
        valori_maxime, indecsi = torch.max(output, dim=1)
        
        # Folosim doar valorile maxime pentru clasificare
        last_hidden_state = valori_maxime
            
        # 4. Clasificarea finală
        logits = self.fc(last_hidden_state)
        return logits.squeeze(-1)
    
# --- ADAUGĂ ACESTE LINII AICI ---

# 1. Definim pe ce hardware rulăm (pe Windows va alege automat CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Rulăm antrenamentul pe: {device}")

# 2. Definim dimensiunile modelului
VOCAB_SIZE = tokenizer.vocab_size  # Câte cuvinte știe tokenizer-ul
EMBED_DIM = 128                    # Dimensiunea vectorilor de cuvinte
HIDDEN_DIM = 128                   # Cât de mare este "creierul" xLSTM
OUTPUT_DIM = 1                     # Un singur output (1 = Stres, 0 = Relaxat)

# 3. Inițializăm modelul și îl mutăm pe device (CPU)
model = StressDetector(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)
model.to(device)

print(f"Modelul a fost inițializat cu succes!")

# --------------------------------

EPOCHS = 10
LEARNING_RATE = 0.0024485979712466923

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- ÎNTRERUPĂTORUL TĂU ---
# Pune True ca să antrenezi, Pune False ca să testezi direct ce ai salvat
VREAU_SA_ANTRENEZ = False


NUME_FISIER_MODEL = 'cel_mai_bun_model_xlstm.pth'

if VREAU_SA_ANTRENEZ:
    print("\n--- Începe antrenarea modelului ---")
    cea_mai_buna_acuratete = 0.0  # Ținem minte recordul
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoca {epoch + 1}/{EPOCHS}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            predictions = model(input_ids)
            loss = criterion(predictions, labels)
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

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
        print(f"   - Acuratete pe Testare:    {acuratete:.2f}%")    
        
        # --- SALVAREA CAMPIONULUI ---
        # Dacă a depășit recordul anterior, îl salvăm!
        if acuratete > cea_mai_buna_acuratete:
            cea_mai_buna_acuratete = acuratete
            torch.save(model.state_dict(), NUME_FISIER_MODEL)
            print(f"   [!] NOU RECORD! Modelul a fost salvat în '{NUME_FISIER_MODEL}'\n")

    print(f"Antrenare completă! Cea mai bună acuratețe a fost: {cea_mai_buna_acuratete:.2f}%")

else:
    # --- ÎNCĂRCAREA MODELULUI SALVAT (CÂND SĂRIM PESTE ANTRENAMENT) ---
    print("\n--- Sărim peste antrenament. Încărcăm modelul salvat... ---")
    try:
        model.load_state_dict(torch.load(NUME_FISIER_MODEL, map_location=device))
        print("Modelul a fost încărcat cu succes! E gata de teste.")
    except FileNotFoundError:
        print(f"Eroare: Nu am găsit fișierul '{NUME_FISIER_MODEL}'. Trebuie să antrenezi modelul măcar o dată cu VREAU_SA_ANTRENEZ = True.")
        exit()

# =====================================================================
# SECȚIUNEA DE TESTARE MANUALĂ (INTERACTIVĂ)
# =====================================================================

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
print("Scrie 'exit' pentru a opri programul.\n")

while True:
    user_input = input("Scrie un text în engleză: ")
    
    if user_input.lower() == 'exit':
        print("Oprire program. O zi fără stres!")
        break
        
    if not user_input.strip():
        print("Te rog să scrii un text valid.")
        continue
        
    pred, prob = predict_stress(user_input, model, tokenizer, device, MAX_LENGTH)
    
    if pred == 1:
        print(f"Rezultat: STRES (Probabilitate: {prob*100:.1f}%)")
    else:
        print(f"Rezultat: FĂRĂ STRES / RELAXAT (Probabilitate: {(1-prob)*100:.1f}%)")