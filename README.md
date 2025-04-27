🖼️🎥 Automatic Image and Video Captioning using Deep Learning
This project presents a unified deep learning framework for automatic caption generation for both images and videos, using hybrid CNN feature extraction and a combined LSTM-GRU decoder architecture.

✨ Features
* Hybrid feature extraction using InceptionV3, VGG16, and ResNet152.

* Single unified model for both image and video captioning.

* Keyframe extraction and feature fusion for video processing.

* Hybrid decoder combining LSTM and GRU for better sequential modeling.

* Evaluation using standard metrics: BLEU, METEOR, and CIDEr.

* Genre-wise analysis using confusion matrices and metric-based scoring.

🗂️ Project Structure

├── caption_image.py              # Generate captions for images
├── caption_video.py              # Generate captions for videos
├── dashboard.py                  # (Optional) Streamlit-based dashboard
├── evaluate_model.py             # BLEU, METEOR, CIDEr evaluation
├── feature_extractor.py          # Extract hybrid features (InceptionV3, VGG16, ResNet152)
├── generate_image_predictions.py # Generate batch image caption predictions
├── generate_video_predictions.py # Generate batch video caption predictions
├── train_image_captioning.py     # Train image captioning model
├── train_video_captioning.py     # Train video captioning model
├── README.md

📊 Results
* Achieved strong BLEU, METEOR, and CIDEr scores across five genres: Indoor, Nature, Social, Sports, and Urban.

* High genre classification accuracy based on generated captions.

* Genre-wise confusion matrices indicate robust semantic understanding.

🚀 How to Run

1. Clone the Repository

git clone https://github.com/yourusername/automatic-image-video-captioning.git
cd automatic-image-video-captioning

3. Install Dependencies

Basic Requirements:

TensorFlow 2.x

OpenCV

scikit-learn

NLTK

NumPy

Matplotlib

3. Prepare Your Dataset
⚡ Note:
The dataset used is private and cannot be shared due to confidentiality.

To run the project:

* Prepare your own images and videos.

* Create ground truth caption CSV files:

=> image_captions.csv

=> video_captions.csv

Dataset Folder Structure:

datasets/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
├── videos/
│   ├── video1.mp4
│   ├── video2.mp4
├── image_captions.csv
├── video_captions.csv

4. Train the Models
Image captioning:
python train_image_captioning.py

Video captioning:
python train_video_captioning.py

5. Generate Captions
For an image:
python caption_image.py --input_path path_to_image.jpg

For a video:
python caption_video.py --input_path path_to_video.mp4

6. Evaluate Model Performance
python evaluate_model.py

Evaluates using BLEU, METEOR, CIDEr scores and confusion matrices.

📋 Requirements
tensorflow>=2.8.0
opencv-python
scikit-learn
nltk
numpy
matplotlib


🛠️ Future Work
Incorporate attention mechanisms for better region-based captioning.

Fine-tune CNN encoders for better feature extraction.

Expand dataset for broader generalization, especially for videos.

Deploy an improved web interface (Streamlit dashboard).



👨‍💻 Project Members and Contributions
Name	         Contribution
Vijayapriya V	 Model Development, Evaluation
Sakthi Shree R Data Collection, Manual Caption Labeling
Jeeviha A	     Feature Extraction, Keyframe Extraction, Documentation
Jeevakaruny R	 Caption Generation, Dashboard Creation

