**Food Classification Using Deep Learning**

This project is a complete end-to-end food image classification system built using Deep Learning and Flask.
The goal is to identify the type of food from an uploaded image and display the predicted food category along with its nutrition values.

The system supports three different deep learning models, which users can select directly from the UI dropdown:

Custom CNN Model

VGG16 (Pre-trained)

ResNet50 (Pre-trained)

Project Overview:
This project aims to solve a real-world problem:
Recognizing food items from images and estimating their nutrition values.
Many health, diet, and fitness applications need accurate food classification. Instead of manually entering food names, users can simply take a picture and get:

Food Name
Calories
Protein
Fat
Carbohydrates
Other nutrition details
My project automates this entire process using deep learning and image recognition.

Deep Learning Models Used
The project offers flexibility by allowing the user to choose any model from the UI.

1.Custom CNN Model
This is a Convolutional Neural Network built from scratch.
Key Features:
Lightweight
Fast inference
Good accuracy for custom datasets
Uses Convolution + Pooling + Dense layers

Why included?
To show understanding of building a model without relying on pre-trained networks.

2️ VGG16
VGG16 is a popular pre-trained model trained on ImageNet.

Strengths:
Deep architecture
Very high accuracy
Extracts features extremely well

How used?
Loaded with ImageNet weights
Removed top layers
Added custom layers for food classification
Fine-tuned on the food dataset

3️ ResNet50
ResNet50 is another high-performance pre-trained model.
Special Feature: Residual Blocks
Residual blocks solve vanishing gradient problems, making training deep networks easier.

 Advantages:
Excellent performance
Very stable training
High accuracy on complex food items

User Interface (UI)
The web interface is built using HTML/CSS/Bootstrap + JavaScript.

UI Features:
Model Selection Dropdown
File Upload Box
Prediction Result Box
Nutrition Table
Error Handling Messages

The user simply:
Chooses a model
Uploads a food image
Clicks "Predict"
Receives the food name + nutrition values
Nutrition Value System
Nutrition values are loaded dynamically from a JSON file (nutrition.json).

How it works:
After prediction, the model returns "food_name"
The app checks nutrition.json for matching nutrition info
If found → show nutrition table
If not → show “Nutrition not available”
Admin can also update nutrition values using a secure API key.
Prediction Pipeline (Step-by-Step)

Here is the entire flow explained clearly:

1️ Image Upload
The user uploads an image (JPG/PNG).

2️ Model Selection
User selects either:
Custom Model
VGG16
ResNet50

3️ Preprocessing
The backend performs:
Resize to required size (e.g., 224×224)
Convert to array
Normalize pixels
Expand dimensions

4️ Prediction
The selected model predicts probabilities for each class.
The highest probability is chosen as the predicted food label.

5️ Nutrition Lookup
The predicted label is used to fetch nutrition data from nutrition.json.

6️ Final Output Displayed
The UI displays:
Predicted Food Name
Selected Model
Nutrition Table
Confidence Score (Optional)

Example Output

Input: Image of Pizza
Model Selected: ResNet50
Predicted: Pizza
Calories: 285 kcal
Protein: 12 g
Fat: 10 g

