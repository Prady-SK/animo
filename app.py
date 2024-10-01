import os
import cv2
import numpy as np
from flask import Flask, request, send_file, render_template

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def anime_filter_advanced(img):
    # Convert to float and normalize
    img_float = img.astype(np.float32) / 255.0

    # Bilateral filtering
    bilateral = cv2.bilateralFilter(img_float, 5, 0.1, 5)

    # Sharpening
    gaussian = cv2.GaussianBlur(bilateral, (7, 7), 2)
    unsharp_mask = bilateral - gaussian
    sharpened = bilateral + 0.7 * unsharp_mask

    # Edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    edge = edge.astype(np.float32) / 255.0

    # Combine edge with sharpened image
    result = np.clip(sharpened + 0.2 * edge, 0, 1)

    # Color quantization
    indices = (result * 4).astype(np.uint8)
    quantized = (indices * 64).astype(np.uint8)

    # Emphasize outlines
    outline = 255 - cv2.Canny(quantized, 100, 200)
    outline = cv2.cvtColor(outline, cv2.COLOR_GRAY2BGR)
    outline = outline.astype(np.float32) / 255.0
    result = np.clip(result + 0.1 * outline, 0, 1)

    # Increase saturation
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 1)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Convert back to uint8
    result = (result * 255).astype(np.uint8)

    return result

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        return 'File uploaded successfully', 200
    return 'File type not allowed', 400

@app.route('/process/<filename>', methods=['POST'])
def process_video(filename):
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], f'anime_{filename}')
    
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Apply advanced anime filter
            anime_frame = anime_filter_advanced(frame)
            out.write(anime_frame)
        else:
            break
    
    # Release everything
    cap.release()
    out.release()
    
    return 'Video processed successfully', 200

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    app.run(debug=True)