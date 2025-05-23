import os
import numpy as np
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flash messages

# Load the trained model
MODEL_PATH = "glaucoma_detection_model.h5"
model = load_model(MODEL_PATH)

# Define image dimensions
IMG_WIDTH, IMG_HEIGHT = 224, 224

# Set up upload directory
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to predict on an image
def predict_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        return "Glaucoma Positive" if preds[0][0] > 0.5 else "Glaucoma Negative"
    except Exception as e:
        return str(e)

# Route for home page
@app.route("/")
def home():
    return render_template('home.html')

# Route for about page
@app.route("/about")
def about():
    return render_template('about.html')

# Route for contact page
@app.route("/contact", methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        # Here you would typically handle the form submission (e.g., send email)
        flash('Thank you for your message! We will get back to you soon.', 'success')
        return redirect(url_for('contact'))
    return render_template('contact.html')

# Route for eye check page
@app.route("/eye-check", methods=['GET', 'POST'])
def eye_check():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            prediction = predict_image(filepath)
            return render_template('result.html', 
                                prediction=prediction, 
                                image_path=filename)
        
        flash('Invalid file type. Please upload a valid image (PNG, JPG, JPEG).', 'error')
        return redirect(request.url)
    
    return render_template('eye_check.html')

# API endpoint for predictions
@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        prediction = predict_image(filepath)
        return jsonify({"filename": filename, "prediction": prediction})
    
    return jsonify({"error": "Invalid file type"})

# API for batch testing
@app.route("/batch_test", methods=["GET"])
def batch_test():
    results = []
    for label, directory in [("Positive", positive_test_dir), ("Negative", negative_test_dir)]:
        for i, filename in enumerate(os.listdir(directory)[:10]):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(directory, filename)
                prediction = predict_image(image_path)
                results.append({"filename": filename, "actual": label, "prediction": prediction})
    return jsonify(results)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=False)
