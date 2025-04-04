import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import BertTokenizerFast, BertForTokenClassification

from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

def read_ner_data(file_path, tokens_per_sentence=10):
    df = pd.read_csv(file_path)
    sentences = []
    labels = []
    
    tokens = df['token'].tolist()
    tags = df['tag'].tolist()
    
    for i in range(0, len(tokens), tokens_per_sentence):
        sentences.append(tokens[i:i+tokens_per_sentence])
        labels.append(tags[i:i+tokens_per_sentence])
    
    return sentences, labels


file_path = "ner_dataset.csv" 
sentences, labels = read_ner_data(file_path)

# Получаем уникальные метки
unique_labels = set(label for sent_labels in labels for label in sent_labels)
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}
num_labels = len(unique_labels)

class NERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        
        # Токенизация
        encoding = self.tokenizer(
            sentence,
            is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_offsets_mapping=True
        )
        
        word_ids = encoding.word_ids()  
        aligned_labels = [-100] * self.max_len  
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:  
                continue
            if i < len(label):  
                aligned_labels[i] = label2id[label[word_idx]]
        
        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(aligned_labels, dtype=torch.long)
        }
        return item
    

tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
model = BertForTokenClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    sentences, labels, test_size=0.5, random_state=42 
)

train_dataset = NERDataset(train_sentences, train_labels, tokenizer)
test_dataset = NERDataset(test_sentences, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

# Обучение
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        predictions = torch.argmax(logits, dim=2).cpu().numpy()
        true_labels = labels.cpu().numpy()
        
        for pred, true in zip(predictions, true_labels):
            pred_labels = [id2label[p] for p, t in zip(pred, true) if t != -100]
            true_labels = [id2label[t] for t in true if t != -100]
            all_preds.append(pred_labels)
            all_labels.append(true_labels)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds))