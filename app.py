from flask import Flask, render_template, request, jsonify, send_file
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)


model = tf.keras.models.load_model('model\model.h5')

# Define class names
class_names = ['Ragdoll', 'Siamese', 'American Shorthair', 'Bengal', 'Birman',
       'Russian Blue', 'Maine Coon', 'Abyssinian', 'Sphynx',
       'British Shorthair', 'Bombay', 'Tuxedo', 'Egyptian Mau', 'Persian',
       'American Bobtail']


def preprocess_image(image):
   
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  
    return image

def predict_image(image, model, class_names):
    img = preprocess_image(image)
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    class_name = class_names[class_index]  
    print("Predicted class:", class_name)

    return class_name

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        uploaded_image = request.files['image']
        
        if uploaded_image.filename != '':
            image = Image.open(uploaded_image)
            
            class_name = predict_image(image, model, class_names)

            temp_buffer = io.BytesIO()
            image.save(temp_buffer, format="JPEG")

            return jsonify({
                'image_url': '/uploads/' + uploaded_image.filename,
                'class_name': class_name
            })

        else:
            return jsonify({'error': 'No file uploaded'})
    except Exception as e:
        return jsonify({'error': str(e)})

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(f'./uploads/{filename}', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
