import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, Bidirectional, TimeDistributed
from tensorflow_addons.layers import CRF
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

# Read the CSV file properly
df = pd.read_csv("ner_dataset.csv", keep_default_na=False)

# Initialize sentence storage
sentences = []
current_sentence = []
labels = []
current_labels = []

# Process each row
for _, row in df.iterrows():
    token = str(row['token']).strip()  # Convert to string and strip whitespace
    tag = row['tag']
    
    # Check for sentence boundary (empty token)
    if token == '':
        if current_sentence:  # Only add if we have a sentence
            sentences.append(current_sentence)
            labels.append(current_labels)
            current_sentence = []
            current_labels = []
    else:
        current_sentence.append(token)
        current_labels.append(tag)

# Add the last sentence if exists
if current_sentence:
    sentences.append(current_sentence)
    labels.append(current_labels)

# Verify we have multiple sentences
print(f"Found {len(sentences)} sentences")
print(f"Example sentence: {' '.join(sentences[0])}")
print(f"Corresponding tags: {' '.join(labels[0])}")

# Create vocabularies
all_words = set(word for sentence in sentences for word in sentence)
all_tags = set(tag for label_seq in labels for tag in label_seq)

word2idx = {word: idx+2 for idx, word in enumerate(all_words)}
word2idx["PAD"] = 0
word2idx["UNK"] = 1

tag2idx = {tag: idx for idx, tag in enumerate(all_tags)}
tag2idx["PAD"] = 0

# Convert to indices
X = [[word2idx.get(word, 1) for word in sent] for sent in sentences]
y = [[tag2idx[tag] for tag in label_seq] for label_seq in labels]

# Padding
MAXLEN = max(len(seq) for seq in X)
X = pad_sequences(X, maxlen=MAXLEN, padding="post", value=0)
y = pad_sequences(y, maxlen=MAXLEN, padding="post", value=tag2idx["PAD"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Model architecture
model = Sequential()
model.add(Embedding(input_dim=len(word2idx), output_dim=50, input_length=MAXLEN))
model.add(Bidirectional(LSTM(units=100, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)))
model.add(Dropout(0.1))
model.add(TimeDistributed(Dense(len(tag2idx))))
crf = CRF(len(tag2idx))
model.add(crf)

# Compile
model.compile(optimizer="adam", loss=crf.loss_function, metrics=[crf.accuracy])

# Train
print("Training model...")
history = model.fit(
    X_train, np.array(y_train),
    batch_size=32,
    epochs=5,
    validation_data=(X_test, np.array(y_test)),
    verbose=1
)

# Save model
model.save("ner_model.h5")

# Evaluation
score = model.evaluate(X_test, np.array(y_test), batch_size=32)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")

# Prediction function
def predict(sentence):
    X_pred = [word2idx.get(word, 1) for word in sentence]
    X_pred = pad_sequences([X_pred], maxlen=MAXLEN, padding="post", value=0)
    y_pred = model.predict(X_pred)
    y_pred = np.argmax(y_pred, axis=-1)[0]
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    pred_tags = [idx2tag[idx] for idx in y_pred if idx in idx2tag]
    return pred_tags[:len(sentence)]  # Return only tags for input words

# Example prediction
test_sentence = ["Россия", "одержала", "победу", "в", "войне"]
predicted_tags = predict(test_sentence)
print(f"Sentence: {' '.join(test_sentence)}")
print(f"Predicted tags: {' '.join(predicted_tags)}")