import numpy as np
import pickle
import os
import cv2
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model('model.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_length = 30
beam_width = 3
image_root = 'dataset/images/'
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

# Beam Search Caption Generation
def generate_caption_beam_search(feature, tokenizer, max_length, beam_width=3):
    start_seq = [tokenizer.word_index['startseq']]
    sequences = [[start_seq, 0.0]]

    while len(sequences[0][0]) < max_length:
        all_candidates = []
        for seq, score in sequences:
            padded_seq = pad_sequences([seq], maxlen=max_length).astype('int32')
            preds = model.predict([np.expand_dims(feature, axis=0), padded_seq], verbose=0)
            top_preds = np.argsort(preds[0])[-beam_width:]

            for word_idx in top_preds:
                word_prob = preds[0][word_idx]
                new_seq = seq + [word_idx]
                new_score = score + np.log(word_prob + 1e-9)
                all_candidates.append([new_seq, new_score])

        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_width]

    final_seq = sequences[0][0]
    caption = ''
    for idx in final_seq:
        word = next((w for w, i in tokenizer.word_index.items() if i == idx), None)
        if word == 'endseq' or word is None:
            break
        if word != 'startseq':
            caption += ' ' + word
    return caption.strip()


# Load extracted features
data = np.load('train_image_features.npz')
image_keys = list(data.files)
random.shuffle(image_keys)
sample_keys = image_keys[:5]

# Generate and display captions for 5 images
for key in sample_keys:
    feature = data[key]
    print(f"\n🔍 Processing: {key} | Feature shape: {feature.shape}")
    
    caption = generate_caption_beam_search(feature, tokenizer, max_length, beam_width)
    if not caption.strip():
        caption = "(No caption generated)"
    print(f"📝 Caption: {caption}")

    # Search image in subfolders
    found = False
    for genre in os.listdir(image_root):
        genre_path = os.path.join(image_root, genre)
        if not os.path.isdir(genre_path):
            continue

        image_path = os.path.join(genre_path, key)
        if os.path.exists(image_path):
            found = True
            img = cv2.imread(image_path)
            if img is None:
                print(f"⚠️ Couldn't read image file: {image_path}")
                break

            img = cv2.resize(img, (640, 480))

            # Draw caption
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            color = (0, 255, 0)
            text_position = (10, 40)

            (text_w, text_h), _ = cv2.getTextSize(caption, font, font_scale, thickness)
            cv2.rectangle(img, (text_position[0] - 5, text_position[1] - text_h - 5),
                          (text_position[0] + text_w + 5, text_position[1] + 5),
                          (0, 0, 0), -1)

            cv2.putText(img, caption, text_position, font, font_scale, color, thickness, cv2.LINE_AA)

            # Show & save
            cv2.imshow(f'Captioned: {key}', img)
            cv2.imwrite(os.path.join(output_folder, f'captioned_{key}'), img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break

    if not found:
        print(f"❌ Image not found for key: {key}")
