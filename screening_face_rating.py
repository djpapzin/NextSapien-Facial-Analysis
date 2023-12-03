import os
import cv2
import dlib
import numpy as np
import gdown
from datetime import datetime
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa

# Constants for the model, dataset, and output settings.
MODEL_URL = 'https://drive.google.com/file/d/1b_tgLV-m7kiVQarKXHpth6paQjwQA89y'  # URL of the pre-trained model
MODEL_PATH = "pre-trained_weights/model_inception_facial_keypoints.h5"  # Path to save the model
DATASET_FOLDER = 'dataset'  # Folder containing images for processing
OUTPUT_FOLDER = './output'  # Folder to save output files
COLOR_GREEN = (0, 255, 0)  # Color for drawing rectangles (Green)
LINE_WIDTH = 3  # Line width for drawing rectangles
THRESHOLD = 0.4  # Threshold for attribute detection
FONT = cv2.FONT_HERSHEY_PLAIN  # Font for text on image
FONT_SCALE = 0.6  # Font scale for text on image

# Labels for facial attributes
labels = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 
    'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 
    'Wearing_Necktie', 'Young'
]


# Attribute weights for calculating face rating.
attribute_weights = {
    'male': {
        'positive': {
            'Arched_Eyebrows': 0.2, 'Oval_Face': 0.3, 'Straight_Hair': 0.2, 'Wavy_Hair': 0.2,
            'Smiling': 0.5, 'Mouth_Slightly_Open': 0.2, 'High_Cheekbones': 0.4, 'Young': 0.6
        },
        'negative': {
            '5_o_Clock_Shadow': -0.2, 'Big_Nose': -0.2, 'Pointy_Nose': -0.2, 'Receding_Hairline': -0.3, 
            'Bags_Under_Eyes': -0.3, 'Blurry': -0.1, 'Gray_Hair': -0.3, 'Chubby': -0.3, 
            'Double_Chin': -0.4, 'Narrow_Eyes': -0.2, 'Eyeglasses': -0.1
        }
    },
    'female': {
        'positive': {
            'Arched_Eyebrows': 0.2, 'Oval_Face': 0.3, 'Straight_Hair': 0.2, 'Wavy_Hair': 0.2, 
            'Smiling': 0.5, 'Mouth_Slightly_Open': 0.2, 'High_Cheekbones': 0.4, 'Young': 0.6
        },
        'negative': {
            '5_o_Clock_Shadow': -2, 'Big_Nose': -0.2, 'Pointy_Nose': -0.2, 'Receding_Hairline': -0.3, 
            'Bags_Under_Eyes': -0.3, 'Blurry': -0.1, 'Gray_Hair': -0.3, 'Chubby': -0.3, 
            'Double_Chin': -0.4, 'Narrow_Eyes': -0.2, 'Eyeglasses': -0.1, 'Male': -2, 'No_Beard': 2
        }
    }
}

def download_model(url, output_path):
    """Downloads a model from a URL to a specified path."""
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path))  # Create directory if it doesn't exist
    gdown.download(url, output_path, quiet=False)  # Download the model file

def download_model(url, output_path):
    """Downloads a model from a URL to a specified path."""
    dir_path = os.path.dirname(output_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)  # Create directory only if it doesn't exist
    gdown.download(url, output_path, quiet=False)  # Download the model file

def load_face_model():
    """Loads the face model, downloading it if not present."""
    if not os.path.isfile(MODEL_PATH):
        download_model(MODEL_URL, MODEL_PATH)  # Download model if not present
    return load_model(MODEL_PATH, custom_objects={"Adamw": tfa.optimizers.AdamW}, compile=False)  # Load and return model

# Load the model and initialize the face detector.
model = load_face_model()  # Load the face model
detector = dlib.get_frontal_face_detector()  # Initialize the Dlib face detector

def calculate_face_rating(attributes, gender):
    """Calculates a face rating based on attributes and gender."""
    gender_key = 'male' if gender == 'male' else 'female'  # Determine gender key
    score = 5  # Start with a base score
    for attribute in attributes:  # Add/subtract attribute weights to/from the score
        score += attribute_weights[gender_key]['positive'].get(attribute, 0)
        score += attribute_weights[gender_key]['negative'].get(attribute, 0)
    return max(1, min(10, score))  # Ensure score is within 1-10

def draw_results(input_image, det, output, file_path, face_number):
    """Draws detection results on an image and writes them to a file."""
    high_prob_positions = np.where(output[0] > THRESHOLD)[0]  # Identify high probability attributes
    detected_attributes = [labels[pos] for pos in high_prob_positions if labels[pos] != 'Male']  # Filter attributes
    male_index = labels.index('Male')  # Get index of 'Male' attribute
    gender = "male" if output[0][male_index] > 0.5 else "female"  # Determine gender
    rating = calculate_face_rating(detected_attributes, gender)  # Calculate face rating

    with open(file_path, 'a') as file:  # Write results to the output file
        file.write(f"Face {face_number}:\n")
        file.write(f"  Gender: {gender.capitalize()}\n")
        file.write(f"  Rating: {rating:.1f}/10\n")
        for attr in detected_attributes:  # Write attributes with non-zero weight
            weight = attribute_weights[gender]['positive'].get(attr, 0) + attribute_weights[gender]['negative'].get(attr, 0)
            if weight != 0:
                file.write(f"    {attr}: {weight}\n")
        file.write('\n')

    x, y, w, h = det.left(), det.top(), det.width(), det.height()  # Get coordinates of the face
    # Draw face number, gender, and rating on the image
    cv2.putText(input_image, f"Face {face_number}", (x, y - 30), FONT, FONT_SCALE, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(input_image, f"{gender.capitalize()}", (x, y - 15), FONT, FONT_SCALE, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(input_image, f"{rating:.1f}/10", (x, y + h + 15), FONT, FONT_SCALE, (0, 0, 255), 1, cv2.LINE_AA)

    return input_image, gender, rating

def process_faces(image, dets, output_path, output_file_path):
    """Processes detected faces in an image and saves the output."""
    for face_number, det in enumerate(dets, start=1):  # Process each detected face
        crop_image = image[det.top():det.bottom(), det.left():det.right()]  # Crop face image
        image_batch = np.zeros((1, 128, 128, 3))  # Prepare batch for prediction
        image_batch[0] = cv2.resize(crop_image, (128, 128)) / 256  # Resize and normalize face image
        output = model.predict(image_batch)  # Predict attributes

        # Draw results and bounding box on the image
        image, gender, rating = draw_results(image, det, output, output_file_path, face_number)
        cv2.rectangle(image, (det.left(), det.top()), (det.right(), det.bottom()), COLOR_GREEN, LINE_WIDTH)
        print(f"Face {face_number}: Gender: {gender.capitalize()}, Rating: {rating:.1f}/10")

    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Save the processed image
    return image

def process_image(image_path, model, detector):
    """Processes a single image for face detection and attribute analysis."""
    img = cv2.imread(image_path)  # Load the image
    if img is None:
        print(f"Error: Image not found at {image_path}.")
        return

    resized_image = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))  # Resize image
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    dets = detector(rgb_image, 1)  # Detect faces
    print(f"Processing {image_path}, detected {len(dets)} faces")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate timestamp
    file_name = os.path.splitext(os.path.basename(image_path))[0]  # Extract file name
    output_image_path = os.path.join(OUTPUT_FOLDER, f'{file_name}_{timestamp}.png')  # Output image path
    output_file_path = os.path.join(OUTPUT_FOLDER, f'{file_name}_{timestamp}.txt')  # Output file path

    processed_image = process_faces(rgb_image, dets, output_image_path, output_file_path)  # Process faces

    cv2.imwrite(output_image_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))  # Save final image
    print(f"Processed image saved to {output_image_path}")

def main():
    """Main function to process all images in the dataset folder."""
    model = load_face_model()  # Load the face model
    detector = dlib.get_frontal_face_detector()  # Initialize the face detector

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)  # Create output folder if it doesn't exist

    for filename in os.listdir(DATASET_FOLDER):  # Process each image in the dataset folder
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(DATASET_FOLDER, filename)
            process_image(image_path, model, detector)

    print("Processing complete.")

if __name__ == "__main__":
    main()