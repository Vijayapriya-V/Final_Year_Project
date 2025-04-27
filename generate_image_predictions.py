import numpy as np
import pickle
import csv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths
model_path = 'model.keras'
tokenizer_path = 'tokenizer.pkl'
feature_path = 'train_image_features.npz'
output_csv = 'predicted_image_captions.csv'
max_length = 30
beam_width = 3

# Load model and tokenizer
model = load_model(model_path)
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

# Beam search
def beam_search_caption(feature, tokenizer, max_length, beam_width=3):
    sequences = [[list(), 0.0]]
    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            in_text = 'startseq ' + ' '.join(seq)
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat = model.predict([np.expand_dims(feature, axis=0), sequence], verbose=0)[0]
            top_indices = np.argsort(yhat)[-beam_width:]
            for idx in top_indices:
                word = next((w for w, i in tokenizer.word_index.items() if i == idx), None)
                if word is None:
                    continue
                candidate = [seq + [word], score - np.log(yhat[idx])]
                all_candidates.append(candidate)
        sequences = sorted(all_candidates, key=lambda tup: tup[1])[:beam_width]
    return ' '.join([w for w in sequences[0][0] if w != 'endseq'])

# Load features
features = np.load(feature_path)

# Generate and save captions
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'caption'])
    for key in features.files:
        caption = beam_search_caption(features[key], tokenizer, max_length, beam_width)
        writer.writerow([key, caption])
        print(f"✅ {key}: {caption}")
