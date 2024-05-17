# Autonomous Bot Development for Counter-Strike: Global Offensive
<sub>**Note**: This project is for educational and research purposes only. It is not intended for use in competitive gaming environments.</sub>

## Overview
This project explores the integration of computer vision and machine learning models to develop an autonomous gaming bot for Counter-Strike: Global Offensive (CS:GO). Utilizing Python, YOLOv8, OpenCV, and TensorFlow, the bot interprets real-time visual data to navigate and engage within the game environment without accessing internal game data.

## Tech Stack
- **Python**: Primary programming language used for development.
- **YOLOv8**: Object detection model.
- **OpenCV**: Library used for image processing and computer vision task.
- **TensorFlow**: Machine learning framework used to enhance the bot's object detection.

## Project Details
### Objective
The goal of this project is to demonstrate advanced applications of artificial intelligence in real-time strategy settings, specifically within the context of a popular first-person shooter game.

### Implementation

1. **Data Collection**: Captured in-game footage to create a dataset for training the object detection model.
2. **Model Training**: Trained the YOLOv8 model on a dataset to identify in-game elements such as teammates, enemies, and different utility types.
3. **Bot Development**: Implemented the bot using Python, integrating the trained YOLOv8 model,OpenCV, and win32api for real-time image processing and player movement.
4. **Decision Making**: Utilized TensorFlow to develop a decision-making system, allowing the bot to react dynamically to the game environment.

### Results
The bot successfully demonstrated the ability to identify teammates, enemies, and utility types. However, the bot appears to be overfitted when it comes to enemy players; it often detects teammates as enemies. This issue may be due to the similar colors within the game, necessitating more training. Additionally, some utility is confused with other types. For instance, when there is fire on the ground and smoke starts to rise, the bot confuses the smoke with a normal smoke grenade. Unfortunately, this project has ended due to the release of Counter-Strike: Global Offensive 2, as the new game does not allow internal input or external image capturing.
