import cv2
import numpy as np
import os
import pickle
import random
import math
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths
model_path = 'model_video.keras'
tokenizer_path = 'tokenizer_video.pkl'
feature_path = 'train_video_features.npz'
video_base_dir = 'dataset/videos'
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Load model and tokenizer
model = load_model(model_path)
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)
print("🧠 Vocabulary size:", len(tokenizer.word_index))
print("📚 Top 10 tokens:", list(tokenizer.word_index.items())[:10])
max_length = 30
beam_width = 3
repetition_threshold = 2

# Clean consecutive repeated words
def clean_caption(caption):
    words = caption.split()
    cleaned = []
    for i in range(len(words)):
        if i == 0 or words[i] != words[i - 1]:
            cleaned.append(words[i])
    return ' '.join(cleaned)

# Beam Search Captioning
def beam_search_caption(feature, tokenizer, max_length, beam_width=3, repetition_threshold=2):
    sequences = [[[], 0.0]]  # [sequence, score]
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
                if word is None or word == 'startseq':
                    continue
                if word == 'endseq':
                    candidate = [seq, score - math.log(yhat[idx])]
                    all_candidates.append(candidate)
                    continue

                if seq.count(word) >= repetition_threshold:
                    continue

                candidate = [seq + [word], score - math.log(yhat[idx])]
                all_candidates.append(candidate)

        if not all_candidates:
            break

        sequences = sorted(all_candidates, key=lambda tup: tup[1])[:beam_width]

    final_caption = ' '.join([w for w in sequences[0][0] if w != 'endseq'])
    return clean_caption(final_caption.strip())

def greedy_caption(feature, tokenizer, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([np.expand_dims(feature, axis=0), sequence], verbose=0)[0]
        yhat = np.argmax(yhat)
        word = next((w for w, i in tokenizer.word_index.items() if i == yhat), None)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return clean_caption(in_text.replace('startseq', '').strip())

# Load features and pick 5 random video keys
feature_data = np.load(feature_path)
all_keys = list(feature_data.files)
selected_keys = random.sample(all_keys, 5)

print("🎯 Selected Videos:")
for k in selected_keys:
    print(f" - {k}")

# Caption and overlay
for key in selected_keys:
    feature = feature_data[key].flatten()

    # Safety check to ensure uniqueness
    if np.all(feature == 0):
        print(f"⚠️ Feature for {key} seems empty or zero!")
        continue

    caption = beam_search_caption(feature, tokenizer, max_length)

    print(f"\n📝 Caption for {key}: {caption}")

    # Extract genre and filename
    genre = key.split('_')[0].capitalize()
    index = key.split('_')[-1].split('.')[0]
    video_name = f"{genre.lower()}_{index}.mp4"
    video_path = os.path.join(video_base_dir, genre, video_name)
    output_path = os.path.join(output_dir, f'captioned_{video_name}')

    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        continue

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        continue

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print(f"🎞️ Playing: {video_name} from genre '{genre}' with caption overlay...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw caption box and text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (0, 255, 0)
        text_pos = (10, 40)
        (text_w, text_h), _ = cv2.getTextSize(caption, font, font_scale, thickness)
        cv2.rectangle(frame, (text_pos[0] - 5, text_pos[1] - text_h - 5),
                      (text_pos[0] + text_w + 5, text_pos[1] + 5), (0, 0, 0), -1)
        cv2.putText(frame, caption, text_pos, font, font_scale, color, thickness, cv2.LINE_AA)

        # Display and write video
        cv2.imshow('Video Captioning', frame)
        out.write(frame)

        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            print("⏭️ Skipping to next video...")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Saved: {output_path}")
