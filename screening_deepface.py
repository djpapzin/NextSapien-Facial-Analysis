import os
import cv2
import dlib
import numpy as np
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model
from datetime import datetime
from deepface import DeepFace
import gdown
from PIL import Image
import keras

# Function to download the model file
def download_model(model_url, model_dir, model_file):
    """
    Downloads a model file from a given URL to a specified directory.
    Args:
        model_url (str): The URL of the model file to download.
        model_dir (str): The directory to save the downloaded model file.
        model_file (str): The name of the model file.
    Returns:
        None
    """
    
    # Create the model directory if it does not exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Download the model file if it does not exist
    if not os.path.isfile(model_file):
        print("Model file not found, downloading...")
        gdown.download(model_url, model_file, quiet=False)

# URL of the model in Google Drive
model_url = 'https://drive.google.com/file/d/1b_tgLV-m7kiVQarKXHpth6paQjwQA89y'
model_dir = 'pre-trained_weights'
model_file = os.path.join(model_dir, 'model_inception_facial_keypoints.h5')

# Download the model if it doesn't exist
download_model(model_url, model_dir, model_file)

# Load the pre-trained model
model = load_model(model_file, custom_objects={"Adamw": tfa.optimizers.AdamW}, compile=False)

# Initialize the Dlib face detector
detector = dlib.get_frontal_face_detector()

# Load height estimation model
height_model = keras.models.load_model('pre-trained_weights/face_to_height_model.h5')

# Define attribute weights with adjustments for both genders
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

# Function to calculate face rating based on detected attributes and gender
def calculate_face_rating(attributes, gender):
    """    
    Calculates the face rating based on the given attributes and gender.
    Args:
        attributes (list): A list of attributes detected from the face.
        gender (str): The gender of the face.
    Returns:
        int: The calculated face rating.
    """
    
    # Convert DeepFace gender result to match attribute_weights keys
    gender_key = 'male' if gender.lower() == 'man' else 'female'

    # Start with a base score
    score = 5

    # Calculate the score based on detected attributes
    for attribute in attributes:
        if attribute in attribute_weights[gender_key]['positive']:
            score += attribute_weights[gender_key]['positive'][attribute]
        elif attribute in attribute_weights[gender_key]['negative']:
            score += attribute_weights[gender_key]['negative'][attribute]

    # Normalize the score to be within 1-10
    score = max(1, min(10, score))

    return score

# Function to draw gender, face number, rating, and height on the image
def draw_gender_and_box(image, face_analysis, det, rating, height, face_number):
    """ 
     Draw a bounding box around a face in an image, along with the face number, gender, rating, and height.
    Args:
        image (numpy.ndarray): The image to draw on.
        face_analysis (dict): Analysis of the face.
        det (dlib.rectangle): The bounding box of the face.
        rating (float): The rating of the face.
        face_number (int): The number of the face.
    Returns:
        numpy.ndarray: The image with the bounding box, face number, gender, and rating drawn.
    """
    
    # Set up font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 0, 0)  # Blue color in BGR

    # Get coordinates and dimensions of the bounding box
    x, y, w, h = det.left(), det.top(), det.width(), det.height()

    # Draw bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw face number, gender, and rating
    gender_text = "Man" if face_analysis['dominant_gender'].lower() == 'man' else "Woman"
    cv2.putText(image, f"Face {face_number}", (x, y - 30), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(image, gender_text, (x, y - 15), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(image, f"{height:.2f}cm", (x, y + h + 15), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(image, f"{rating:.1f}/10", (x, y + h + 30), font, font_scale, color, thickness, cv2.LINE_AA)

    return image

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

# Threshold for attribute detection
threshold = 0.4

# Function to draw results and calculate face rating
def draw_results(input_image, det, output, file_path, face_number, deepface_data):
    """
    Draws the results of face detection and analysis on the input image.

    Args:
        input_image (numpy.ndarray): The input image on which to draw the results.
        det (dlib.rectangle): The detected face rectangle.
        output (numpy.ndarray): The output array from the face attribute prediction model.
        file_path (str): The file path to which the results will be written.
        face_number (int): The number of the detected face.
        deepface_data (dict): A dictionary containing the deepface analysis results.

    Returns:
        tuple: A tuple containing the modified input image and the list of detected attributes.
    """
    # Set up font properties
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 0.6

    # Determine positions of attributes with high probability
    high_prob_positions = np.where(output[0] > threshold)[0]
    detected_attributes = [labels[pos] for pos in high_prob_positions if labels[pos] != 'Male']

    # Convert DeepFace gender result to match attribute_weights keys
    gender_key = 'male' if deepface_data['dominant_gender'].lower() == 'man' else 'female'

    # Calculate face rating
    rating = calculate_face_rating(detected_attributes, gender_key)

    # Write results to the file
    with open(file_path, 'a') as file:
        file.write(f"Face {face_number}:\n")
        file.write(f"  Gender: {deepface_data['dominant_gender']}\n")
        file.write(f"  Age: {deepface_data['age']}\n")
        file.write(f"  Race: {deepface_data['dominant_race']}\n")
        file.write(f"  Emotion: {deepface_data['dominant_emotion']}\n")
        file.write(f"  Rating: {rating}/10\n")
        for attr in detected_attributes:
            weight = attribute_weights[gender_key]['positive'].get(attr, 0) + attribute_weights[gender_key]['negative'].get(attr, 0)
            file.write(f"    {attr}: {weight}\n")
        file.write('\n')

    # Draw rating on the image
    x, y, w, h = det.left(), det.top(), det.width(), det.height()
    cv2.putText(input_image, f"Face {face_number}", (x, y - 20), font, font_scale, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(input_image, f"Gender: {deepface_data['dominant_gender']}", (x, y + h + 15), font, font_scale, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(input_image, f"Rating: {rating}/10", (x, y + h + 30), font, font_scale, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(input_image, f"Height: {deepface_data['height']}m", (x, y + h + 45), font, font_scale, (255, 0, 0), 1, cv2.LINE_AA)

    return input_image, detected_attributes

# Function to estimate height
def estimate_height(detected_face):
    region_image = Image.fromarray(detected_face[:, :, ::-1])  # Convert BGR to RGB
    resized_img = region_image.resize((128, 128), Image.Resampling.LANCZOS)
    output_array = np.array(resized_img) / 255.0
    output_array = np.expand_dims(output_array, axis=0)
    predicted_height = height_model.predict(output_array)[0][0]
    return predicted_height


# Function to process a face
def process_face(img, det, deepface_data, face_number, output_file_path):
    """
    Processes a face image and performs attribute analysis.

    Args:
        img (ndarray): The input image containing the face.
        det (dlib.rectangle): The bounding box coordinates of the face.
        deepface_data (dict): A dictionary containing deepface data for the face.
        face_number (int): The number of the face in the image.
        output_file_path (str): The path to the output file.

    Returns:
        tuple: A tuple containing the processed image and the face rating.

    This function crops the face image, resizes it to 128x128 pixels, and normalizes the pixel values.
    It then uses a pre-trained model to predict the attributes of the face.
    The function determines the positions of attributes with high probability and filters out the 'Male' attribute.
    The face rating is calculated based on the detected attributes and the dominant gender.
    The function draws the gender, face number, and rating on the input image.
    Finally, the results are written to the output file, including the face number, gender, age, race, emotion, rating,
    and the weights of the detected attributes.

    Note: The image, bounding box, and deepface data should be preprocessed before passing them to this function.

    Example usage:
        img, rating = process_face(img, det, deepface_data, face_number, output_file_path)
    """
    # Crop and process the face for attribute analysis
    cropImage = img[det.top():det.bottom(), det.left():det.right()]
    image_batch = np.zeros((1, 128, 128, 3))
    image_batch[0] = cv2.resize(cropImage, (128, 128), interpolation=cv2.INTER_CUBIC) / 256
    output = model.predict(image_batch)
    
    # Height estimation
    cropImage = img[det.top():det.bottom(), det.left():det.right()]
    height = estimate_height(cropImage)
    deepface_data['height'] = height

    # Determine positions of attributes with high probability
    high_prob_positions = np.where(output[0] > threshold)[0]
    detected_attributes = [labels[pos] for pos in high_prob_positions if labels[pos] != 'Male']

    # Calculate the rating
    gender_key = 'male' if deepface_data['dominant_gender'].lower() == 'man' else 'female'
    rating = calculate_face_rating(detected_attributes, gender_key)

    # Draw gender, face number, rating, and height on the image
    img = draw_gender_and_box(img, deepface_data, det, rating, height, face_number)


    # Write results to the file with rating rounded to one decimal place
    with open(output_file_path, 'a') as file:
        file.write(f"Face {face_number}:\n")
        file.write(f"  Gender: {deepface_data['dominant_gender']}\n")
        file.write(f"  Age: {deepface_data['age']}\n")
        file.write(f"  Height: {height:.2f}cm\n")
        # file.write(f"  Race: {deepface_data['dominant_race']}\n")
        # file.write(f"  Emotion: {deepface_data['dominant_emotion']}\n")
        file.write(f"  Rating: {rating:.1f}/10\n")
        for attr in detected_attributes:
            weight = attribute_weights[gender_key]['positive'].get(attr, 0) + attribute_weights[gender_key]['negative'].get(attr, 0)
            if weight != 0:
                file.write(f"    {attr}: {weight}\n")
        file.write('\n')

    return img, rating


# Main function
def main():
    input_folder = 'dataset'
    output_folder = './output'

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over each file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, file_name)

            # Read the image using OpenCV
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image {file_name}. Skipping.")
                continue
            resizedImage = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
            rgb_image = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2RGB)

            # Prepare for output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_image_path = os.path.join(output_folder, f'{os.path.splitext(file_name)[0]}_{timestamp}.png')
            output_file_path = os.path.join(output_folder, f'{os.path.splitext(file_name)[0]}_{timestamp}.txt')

            # Analyze the image using DeepFace for demographic information
            deepface_analysis = DeepFace.analyze(img_path, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False, detector_backend='retinaface')

            # Initialize face number
            face_number = 1

            # Process the analysis results
            if isinstance(deepface_analysis, list):
                for face_analysis in deepface_analysis:
                    x, y, w, h = (face_analysis['region'][k] for k in ['x', 'y', 'w', 'h'])
                    det = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)
                    img, rating = process_face(img, det, face_analysis, face_number, output_file_path)
                    face_number += 1
            else:
                x, y, w, h = (deepface_analysis['region'][k] for k in ['x', 'y', 'w', 'h'])
                det = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)
                img, rating = process_face(img, det, deepface_analysis, face_number, output_file_path)

            # Save and display the final image
            cv2.imwrite(output_image_path, img)
            # cv2.imshow("Processed Image", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

# Call the main function
if __name__ == "__main__":
    main()
