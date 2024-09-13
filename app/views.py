from django.shortcuts import render
from django.http import HttpResponseRedirect
from PIL import Image
import io
from django.urls import reverse
import numpy as np
import tensorflow as tf
from .forms import ImageUploadForm

# Load the pre-trained model
model = tf.keras.models.load_model('./saved_model/best_best_model.h5')

def classify_image(image):
    """Classify the image using the pre-trained model."""
    # Preprocess the image
    image_array = preprocess_image(image)
    
    if image_array is None:
        return ["Error processing image", 0]

    # Predict using the model
    predictions = model.predict(image_array)
    
    # Get the index of the class with the highest confidence
    class_index = np.argmax(predictions)
    
    # Get the confidence of that class
    confidence = predictions[0][class_index] * 100
    
    # List of class names
    class_names = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
        'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
        'Tomato___healthy'
    ]
    
    # Validate class_index
    if 0 <= class_index < len(class_names):
        class_name = class_names[class_index]
    else:
        class_name = "Unknown"

    return [class_name, confidence]

def preprocess_image(image):
    """Preprocess the image for model input."""
    try:
        image_data = io.BytesIO(image.read())
        pil_image = Image.open(image_data).convert('RGB')
        pil_image = pil_image.resize((128, 128))  # Adjust size to match model input
        image_array = np.array(pil_image)
        image_array = image_array  # Normalize the image to [0, 1] range
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def start(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            # Perform image classification
            predicted_result = classify_image(image)
            print(predicted_result)
            return HttpResponseRedirect(reverse('result', kwargs={'result': predicted_result}))
    else:
        form = ImageUploadForm()
    
    return render(request, 'res.html', {'form': form})

def result(request, result):
    return render(request, 'result.html', {'result': result})

'''from django.shortcuts import render
from django.http import HttpResponseRedirect
from PIL import Image
import io
from django.urls import reverse
import numpy as np
import tensorflow as tf
from .forms import ImageUploadForm
from django.conf import settings
import os

# Load the pre-trained model
model = tf.keras.models.load_model('./saved_model/cat_model.h5')

def classify_image(image):
    """Classify the image using the pre-trained model."""
    image_array = preprocess_image(image)
    if image_array is None:
        return None  # Early return if image preprocessing failed
    
    predictions = model.predict(image_array)
    
    # Get the index of the class with the highest confidence
    class_index = np.argmax(predictions)
    
    # Get the confidence of that class
    confidence = predictions[0][class_index] * 100
    
    # List of class names
    class_names = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
        'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
        'Tomato___healthy'
    ]
    class_names=['Cat','Dog']
    
    # Get the class name
    class_name = class_names[class_index]
    
    return [class_name, confidence]

def preprocess_image(image):
    """Preprocess the image for model input."""
    try:
        # Convert bytes to a PIL image directly
        pil_image = Image.open(io.BytesIO(image)).convert('RGB')
        pil_image = pil_image.resize((128, 128))  # Adjust size to match model input
        image_array = np.array(pil_image)
        image_array = image_array  # Normalize the image
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def change_background_color_to_match_reference(image_file, reference_image_path):
    """Change the background color of the image to match the color of the reference image."""
    try:
        pil_image = Image.open(image_file).convert('RGBA')
        pil_reference_image = Image.open(reference_image_path).convert('RGBA')
        
        # Calculate the average color of the reference image
        total_r, total_g, total_b = 0, 0, 0
        width, height = pil_reference_image.size
        for x in range(width):
            for y in range(height):
                r, g, b, a = pil_reference_image.getpixel((x, y))
                total_r += r
                total_g += g
                total_b += b
        avg_r = total_r // (width * height)
        avg_g = total_g // (width * height)
        avg_b = total_b // (width * height)
        reference_color = (avg_r, avg_g, avg_b)
        
        # Change the color of the background pixels in the input image to match the reference color
        width, height = pil_image.size
        for x in range(width):
            for y in range(height):
                r, g, b, a = pil_image.getpixel((x, y))
                if a == 0:  # only change background pixels
                    pil_image.putpixel((x, y), reference_color + (a,))
        
        # Convert PIL image back to bytes for further processing
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return img_byte_arr
    except Exception as e:
        print(f"Error changing background color: {e}")
        return None

def start(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = request.FILES['image']
            
            # Load the reference image from the 'saved_model' folder
            reference_image_path = os.path.join('./saved_model', 'AppleCedarRust3.jpg')
            
            # Change the background color of the uploaded image to match the reference image
            processed_image_data = change_background_color_to_match_reference(uploaded_image, reference_image_path)
            
            if processed_image_data is None:
                # Handle error case
                return render(request, 'error.html', {'error_message': 'Error processing the image. Please try again.'})
            
            # Classify the processed image
            result = classify_image(processed_image_data)
            
            if result is None:
                # Handle error case
                return render(request, 'error.html', {'error_message': 'Error classifying the image. Please try again.'})
            
            return HttpResponseRedirect(reverse('result', kwargs={'result': result}))
    else:
        form = ImageUploadForm()
        return render(request, 'res.html', {'form': form})
    
def result(request, result):
    return render(request, 'result.html', {'result': result})'''



