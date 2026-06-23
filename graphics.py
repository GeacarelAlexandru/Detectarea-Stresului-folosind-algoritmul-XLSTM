import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import re
from Pure_xLSTM import Pure_xLSTM

print("Se încarcă datele și modelul pentru generarea graficelor...")

# 1. GRAFICUL DISTRIBUȚIEI DATELOR
def deseneaza_distributie():
    df_train = pd.read_csv('dreaddit-train.csv')
    df_train = df_train.dropna(subset=['label'])
    
    # Numărăm câte texte sunt cu stres (1) și câte relaxate (0)
    valori = df_train['label'].value_counts()
    etichete = ['Stres (1)', 'Relaxat (0)']
    culori = ['#ff9999', '#66b3ff']
    
    plt.figure(figsize=(8, 6))
    plt.pie(valori, labels=etichete, colors=culori, autopct='%1.1f%%', startangle=140, explode=(0.05, 0))
    plt.title('Distribuția Claselor în Setul de Date Dreaddit (Antrenament)', fontsize=14)
    
    # Salvăm imaginea în folder
    plt.savefig('grafic_1_distributie_date.png', bbox_inches='tight')
    plt.close()
    print("-> S-a salvat 'grafic_1_distributie_date.png'")

# 2. GRAFICUL EVOLUȚIEI (LOSS CURVE)
def deseneaza_loss_curve():
    # AICI TREBUIE SĂ PUI TU NUMERELE DIN TERMINAL!
    # Uită-te în consolă la rularea ta norocoasă și trece erorile (Loss) de la fiecare epocă:
    epoci = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Exemplu: înlocuiește aceste numere cu valorile "Eroare Antrenare" reale din terminalul tău
    train_loss = [0.6800, 0.6200, 0.5800, 0.5100, 0.4700, 0.4300, 0.3900, 0.3500, 0.3200, 0.2900]
    
    # Exemplu: înlocuiește aceste numere cu valorile "Eroare Testare" reale din terminalul tău
    test_loss = [0.6600, 0.6100, 0.5900, 0.5400, 0.5100, 0.4900, 0.4700, 0.4800, 0.4900, 0.5100]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epoci, train_loss, marker='o', linestyle='-', color='blue', label='Eroare Antrenare (Train Loss)')
    plt.plot(epoci, test_loss, marker='s', linestyle='--', color='red', label='Eroare Testare (Test Loss)')
    
    plt.title('Evoluția Erorii (Loss) pe parcursul antrenamentului', fontsize=14)
    plt.xlabel('Epoca', fontsize=12)
    plt.ylabel('Valoare Loss', fontsize=12)
    plt.xticks(epoci)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('grafic_2_loss_curve.png', bbox_inches='tight')
    plt.close()
    print("-> S-a salvat 'grafic_2_loss_curve.png'")

# 3. MATRICEA DE CONFUZIE
# Reproducem rapid clasele esențiale ca să putem încărca modelul salvat
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class StressDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, index):
        encoding = self.tokenizer(clean_text(str(self.texts[index])), add_special_tokens=True,
                                  max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].flatten(), 'label': torch.tensor(self.labels[index], dtype=torch.float)}

class StressDetector(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(StressDetector, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.sequence_model = Pure_xLSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        output, _ = self.sequence_model(x)
        valori_maxime, _ = torch.max(output, dim=1)
        return self.fc(valori_maxime).squeeze(-1)

def deseneaza_matrice_confuzie():
    # Setările modelului (ASIGURĂ-TE CĂ SUNT ACELEAȘI CA ÎN MAIN.PY)
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    EMBED_DIM = 128
    HIDDEN_DIM = 128  # Pune valoarea ta de la Optuna
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Încărcăm doar datele de test
    df_test = pd.read_csv('dreaddit-test.csv').dropna(subset=['text', 'label'])
    test_dataset = StressDataset(df_test['text'], df_test['label'], tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Încărcăm Campionul
    model = StressDetector(tokenizer.vocab_size, EMBED_DIM, HIDDEN_DIM, 1).to(device)
    try:
        model.load_state_dict(torch.load('cel_mai_bun_model_xlstm.pth', map_location=device))
    except Exception as e:
        print("Nu am găsit fișierul modelului! Rulează main.py cu antrenamentul pornit mai întâi.")
        return
        
    model.eval()
    y_adevarat = []
    y_prezis = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            predictions = model(input_ids)
            pred_labels = (predictions > 0).float()
            
            y_adevarat.extend(labels.cpu().numpy())
            y_prezis.extend(pred_labels.cpu().numpy())
            
    # Generăm matricea
    cm = confusion_matrix(y_adevarat, y_prezis)
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Relaxat (0)', 'Stres (1)'], yticklabels=['Relaxat (0)', 'Stres (1)'])
    plt.title('Matricea de Confuzie (Setul de Testare)', fontsize=14)
    plt.ylabel('Eticheta Adevărată', fontsize=12)
    plt.xlabel('Predicția Modelului xLSTM', fontsize=12)
    
    plt.savefig('grafic_3_matrice_confuzie.png', bbox_inches='tight')
    plt.close()
    print("-> S-a salvat 'grafic_3_matrice_confuzie.png'")

# EXECUTAREA SCRIPTULUI
if __name__ == "__main__":
    print("Generăm graficele pentru documentația de licență...\n")
    deseneaza_distributie()
    deseneaza_loss_curve()
    deseneaza_matrice_confuzie()
    print("\nToate graficele au fost create cu succes! Le găsești în folderul proiectului.")