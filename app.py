from flask import Flask, request, render_template, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load model
model = load_model('model/BrainTumorDetectionEfficientNetB0.h5')

# Define class labels
class_labels = ['Glioma', 'No Tumor', 'Meningioma', 'Pituitary']

def predict_tumor(file_path):
    img = load_img(file_path, target_size=(150, 150))  # Adjust to your model's input size
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_class = class_labels[np.argmax(predictions)]
    predicted_proba = {class_labels[i]: float(predictions[0][i]) for i in range(len(class_labels))}
    return predicted_class, predicted_proba

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            predicted_class, predicted_proba = predict_tumor(file_path)
            return render_template('index.html', result=predicted_class, probabilities=predicted_proba, filename=file.filename)
    return render_template('index.html', result=None, probabilities=None)

if __name__ == '__main__':
    app.run(debug=True)
