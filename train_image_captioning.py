import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Embedding, Dropout, add
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import os

# Load image features
features_path = "train_image_features.npz"
data = np.load(features_path)
image_features = {key: data[key] for key in data.files}

# Load captions CSV
captions_df = pd.read_csv("image_captions.csv")  # Columns: filename, caption

# Create mapping: filename -> list of captions
captions_map = {}
for _, row in captions_df.iterrows():
    fname, cap = row["image"], row["caption"]
    captions_map.setdefault(fname, []).append("startseq " + cap + " endseq")

# Flatten all captions for tokenizer
all_captions = [cap for caps in captions_map.values() for cap in caps]

# Tokenize captions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = 30

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Create training data
X1, X2, y = [], [], []

for fname, captions in captions_map.items():
    feature = image_features.get(fname)
    if feature is None:
        continue
    for cap in captions:
        seq = tokenizer.texts_to_sequences([cap])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)

X1 = np.array(X1)
X2 = np.array(X2)
y = np.array(y)

print("Training samples:", X1.shape[0])
print("Feature shape:", X1.shape, "Sequence shape:", X2.shape, "Output shape:", y.shape)

# Build model
def build_model(vocab_size, max_length):
    # Image feature input
    inputs1 = Input(shape=(4608,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)  # 👈 Changed from 512 to 256

    # Sequence input
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256, return_sequences=True)(se2)
    se4 = GRU(256)(se3)

    # Merge
    decoder1 = add([fe2, se4])  # 👈 Now both are 256-dim
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=Adam())
    return model

model = build_model(vocab_size, max_length)
model.summary()

# Train the model
model.fit([X1, X2], y, epochs=20, batch_size=64)

# Save the model
model.save("model.keras")
print("✅ Image captioning model saved as model.keras")
