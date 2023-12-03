# Advanced Facial Analysis Project for NextSapien

Welcome to the Advanced Facial Analysis Project, a groundbreaking initiative designed for NextSapien. This project represents a significant leap in AI and computer vision, focusing on the intricate task of human facial analysis. It aligns perfectly with NextSapien's ethos of pushing the boundaries of AI/ML applications, showcasing innovative approaches in gender identification, face rating, and providing comprehensive demographic insights.

![Demo Image](demo.png)

## Key Achievements
- Developed a unique face rating system, integrating 40 facial attributes with a focus on Facial Symmetry, Skin Texture, and Expression Analysis.
- Successfully integrated the DeepFace framework for comprehensive demographic analysis, enhancing the depth of facial analysis.
- Overcame technical challenges, including the incompatibility with mxnet library, by switching to facenet for facial attribute extraction.

## Technical Overview
This project employs advanced algorithms for face detection and attribute recognition, demonstrating a deep level of analysis. The unique face rating system, a highlight of this project, exemplifies creativity and problem-solving skills. It rates faces on a scale of 1-10, considering various attributes like facial symmetry, skin texture, and expression analysis. These technical feats not only meet but set new standards in AI/ML applications, reflecting NextSapien's commitment to pioneering new frontiers in technology.

## Project Structure
- `pre-trained_weights/`: Contains pre-trained model weights.
- `screening_face_rating.py`: Core script for face rating and gender analysis.
- `screening_deepface.py`: Alternative Script for advanced facial analysis using DeepFace library.
- `dataset/`: Input images for processing.
- `output/`: Stores processed images and analysis reports.
- `demo.png`: Screenshot of expected output image.

## Scripts Overview
### `screening_face_rating.py`
Uses the Dlib library to detect faces in an image and then uses a pre-trained TensorFlow model to predict the facial attributes of each detected face. The script also includes code to save the results to a file and display the final image with the detected faces and attributes.

### `screening_deepface.py`
Leverages the DeepFace framework for comprehensive facial analysis, providing demographic insights such as age, gender, race, and emotion. It enhances the depth of facial analysis by integrating advanced AI techniques.

## Model Download
- Pre-trained Model: Automatically downloaded when running `screening_face_rating.py` alternatively, download from [Google Drive Link](https://drive.google.com/file/d/1b_tgLV-m7kiVQarKXHpth6paQjwQA89y) and place it inside `pre-trained_weights` folder.

## Running the Project
1. **Install Dependencies**: `pip install -r requirements.txt`.
2. **Prepare Data**: Place images in `dataset/`.
3. **Execute**: Run `python screening_face_rating.py` or `python screening_deepface.py`.
4. **Results**: Check `output/` for images and reports.

## Development of Face Rating Methodology
### Conceptualization
- **Initial Challenge**: The task was to rate faces on a 1-10 scale with a goal of 90% accuracy, a challenging objective considering the subjective nature of facial aesthetics.
- **Research and Analysis**: Extensive research was conducted on various facial attributes that could be quantitatively analyzed. This included studying facial symmetry, skin texture, expression analysis, and eye clarity, among others.

### Methodology Development
- **Attribute Selection and Grouping**: Based on research, 40 facial attributes were identified. These attributes were then grouped according to the nodes specified by NextSapien: Facial Symmetry, Skin Texture, Facial Proportions, Expression Analysis, and Eye Clarity. Some attributes, while present, did not align directly with these nodes but could become relevant in future tasks with additional nodes.
- **Weight Assignment**: Each attribute was assigned a weight based on its perceived impact on overall facial aesthetics. This step involved iterative testing and refinement to align the system with the project's accuracy goals.
- **Integration with AI Models**: The selected attributes were then integrated with AI models capable of detecting these features in facial images. This integration was key to automating the face rating process.

### Example of Score Calculation
- **Sample Calculation**: For a face detected with 'High_Cheekbones' (0.4), 'Smiling' (0.5), and 'Big_Nose' (-0.2), the initial score of 5 would be adjusted as follows: 5 + 0.4 (High_Cheekbones) + 0.5 (Smiling) - 0.2 (Big_Nose) = 5.7. This score is then normalized to ensure it falls within the 1-10 range.

### Refinement and Testing
- **Threshold Adjustment**: To improve accuracy, the threshold for attribute detection was carefully adjusted. This ensured that only the most prominent features were considered in the rating, reducing the likelihood of false positives.
- **Continuous Refinement**: The system underwent continuous refinement, incorporating feedback and results from multiple test iterations. This iterative process was vital in fine-tuning the methodology to meet the desired accuracy levels.

### Outcome
- **Innovative Solution**: The developed face rating system is a testament to innovative thinking and technical proficiency. It stands as a unique solution in the realm of AI-driven facial analysis.
- **Alignment with NextSapien's Vision**: This methodology aligns with NextSapien's vision for pioneering new AI applications, demonstrating the potential for AI to venture into areas that require a blend of quantitative analysis and creative thinking.

## Challenges and Solutions
- **Incompatibility with mxnet library**: Switched to facenet for facial attribute extraction.
- **Objective Face Rating System**: Developed a subjective system, emphasizing continuous refinement.
- **Accuracy Improvement**: Adjusted the threshold for attribute detection to enhance accuracy.

## Current Status (02/12/2023)
- Finalizing project for submission.
- Uploaded files to GitHub and Google Drive.
- Added `demo.png` to showcase expected output.

## Future Enhancements
- **View Image Post-Processing**: Implement functionality to view images immediately after processing.
- **Bias Mitigation**: Address the current bias towards female faces in the rating system.
- **Height Detection**: Integrate height detection into the script.
- **CelebA Dataset Testing**: Test models on the CelebA dataset for broader validation.
- **Additional Attributes**: Incorporate more attributes from other nodes, particularly for height and age analysis.
- **Age and Height Determination**: Focus on task 2, which involves determining age (already achievable with DeepFace) and integrating a model for height detection.

## Project Timeline
### Initial Phase: Understanding and Setup
- **Day 1 (2023-11-10)**: Received the assignment, emphasizing the importance of advanced human face detection in AI/ML, a key area for NextSapien.
- **Day 2 (2023-11-11)**: Researched acclaimed models for gender identification, aligning with NextSapien's focus on cutting-edge AI solutions.

### DeepFace Integration and Model Exploration
- **Day 3-4 (2023-11-12 to 2023-11-13)**: Overcame dependency issues by integrating Deep

Face, showcasing adaptability and technical skill, crucial for NextSapien's dynamic AI projects.
- **Day 5 (2023-11-14)**: Explored face rating models, addressing NextSapien's requirement for innovative approaches in AI/ML.

### Gender Identification and YOLOv5 Implementation
- **Day 6-7 (2023-11-15 to 2023-11-16)**: Focused on gender identification and integrated YOLOv5, demonstrating proficiency in utilizing advanced AI tools, a core requirement for NextSapien.

### Development of Face Rating System
- **Day 8-9 (2023-11-17 to 2023-11-18)**: Developed a unique face rating system, reflecting creativity and problem-solving skills, aligning with NextSapien's ethos of innovation in AI.

### Advanced Feature Integration and Script Refinement
- **Day 10-16 (2023-11-19 to 2023-11-25)**: Explored additional attribute detection and refined scripts, showcasing the ability to enhance AI models, crucial for NextSapien's evolving projects.

### Integration of Facial Attributes Analysis
- **Day 17-20 (2023-11-26 to 2023-11-29)**: Successfully integrated a comprehensive facial attributes analysis system, demonstrating the capability to merge different AI technologies, a skill highly valued at NextSapien.

### Finalization and Refinement
- **Day 21-23 (2023-11-30 to 2023-12-02)**: Refined the face rating system and finalized the project, showcasing a commitment to delivering high-quality, innovative AI solutions, in line with NextSapien's mission.

## Acknowledgments
Gratitude to NextSapien for inspiring this project and to the developers of the tools and libraries used.

## Contact Information
L.fanampe@gmail.com | [GitHub](https://github.com/djpapzin) | [LinkedIn](https://www.linkedin.com/in/letlhogonolo-fanampe-32ba9540/)

## License
Under MIT License - [LICENSE.md](./LICENSE.md).

## Personal Note
This project has been a journey of growth in AI and computer vision, aligning with my professional aspirations and NextSapien's innovative goals. I look forward to contributing to NextSapien's vision and being part of a team that transforms the technological landscape.