import pandas as pd 
import re
import torch
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
df_test = pd.read_csv('dreaddit-test.csv')

df_train = df_train.dropna(subset=['text', 'label'])
df_test = df_test.dropna(subset=['text', 'label'])

df_train['text'] = df_train['text'].apply(clean_text)
df_test['text'] = df_test['text'].apply(clean_text)

# Class Dataset
class StressDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Final DataLoaders
TOKENIZER_NAME = 'bert-base-uncased'
MAX_LENGTH = 128
BATCH_SIZE = 16

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# Create objects of the StressDataset class for training and testing
train_dataset = StressDataset(df_train['text'], df_train['label'], tokenizer, MAX_LENGTH)
test_dataset = StressDataset(df_test['text'], df_test['label'], tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Set de antrenament: {len(train_dataset)} texte.")
print(f"Set de testare: {len(test_dataset)} texte.")