import os
import subprocess
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from torch.utils.model_zoo import load_url
from scipy.special import expit
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Use an absolute path for better reliability
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure uploads folder exists

# Install dependencies (if missing)
try:
    import efficientnet_pytorch
except ImportError:
    subprocess.run(["pip", "install", "efficientnet-pytorch"])

try:
    import albumentations
except ImportError:
    subprocess.run(["pip", "install", "-U", "git+https://github.com/albu/albumentations"], stdout=subprocess.DEVNULL)

# Clone repo if not exists
if not os.path.exists("icpr2020dfdc"):
    subprocess.run(["git", "clone", "https://github.com/polimi-ispl/icpr2020dfdc"])

# Change directory to notebook
repo_path = os.path.join(os.getcwd(), "icpr2020dfdc", "notebook")
if os.path.exists(repo_path):
    os.chdir(repo_path)
else:
    raise FileNotFoundError("Repository not found. Please check the clone process.")

sys.path.append('..')

from blazeface import FaceExtractor, BlazeFace, VideoReader
from architectures import fornet, weights
from isplutils import utils

# Parameters
net_model = 'EfficientNetAutoAttB4'
train_db = 'DFDC'
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
face_policy = 'scale'
face_size = 224
frames_per_video = 32

# Model Initialization
model_url = weights.weight_url[f"{net_model}_{train_db}"]
net = getattr(fornet, net_model)().eval().to(device)
net.load_state_dict(load_url(model_url, map_location=device, check_hash=True))

# Face Detection & Preprocessing
transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)
facedet = BlazeFace().to(device)
facedet.load_weights("../blazeface/blazeface.pth")
facedet.load_anchors("../blazeface/anchors.npy")
videoreader = VideoReader(verbose=False)
video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn=video_read_fn, facedet=facedet)

# File type validation
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']

        if not allowed_file(video_file.filename):
            return jsonify({'error': 'Invalid file type. Only video files are allowed.'}), 400

        # Save the file to the absolute path
        video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
        print(f"Saving video to: {video_path}")  # Debugging line
        video_file.save(video_path)

        # Process the video
        video_faces = face_extractor.process_video(video_path)

        if not video_faces:
            return jsonify({'error': 'No faces detected in the video'}), 400

        faces_t = torch.stack([
            transf(image=frame['faces'][0])['image']
            for frame in video_faces if len(frame['faces'])
        ])

        with torch.no_grad():
            faces_pred = net(faces_t.to(device)).cpu().numpy().flatten()

        # Calculate Deepfake score as percentage
        average_score = expit(faces_pred.mean()) * 100
        confidence_level = average_score  # Confidence is now the same as the score percentage
        accuracy_rate = (100 - average_score)  # Accuracy is the complement of the Deepfake score

        # Return the results as a JSON response
        return jsonify({
            'average_score': float(average_score),
            'confidence': confidence_level,
            'accuracy_rate': accuracy_rate,
            'media_type': video_file.mimetype
        })


    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
