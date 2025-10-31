# 🍎 FruityVision — Fruit Recognition & Nutrition Analysis
FruityVision is an AI-powered application that identifies fruits from an image and provides a detailed breakdown of their nutritional value. This system uses a deep learning model built with TensorFlow/Keras for image classification and is deployed as an interactive web app using Streamlit.

# 🧠 How It Works
The system is built in two main parts: the deep learning model and the front-end application.

1. Modeling (Deep Learning):

    Data Preparation: The model was trained on a dataset of fruit images, which were augmented using ImageDataGenerator to create variations in rotation, zoom, and flips. This makes the model more robust.

    Transfer Learning: It uses a pre-trained ResNet50V2 model, a powerful Convolutional Neural Network (CNN), as its base. The initial layers are frozen to leverage knowledge from the massive ImageNet dataset.

    Custom Layers: A custom classification head was added on top of ResNet50V2, consisting of BatchNormalization, Dropout (to prevent overfitting), and Dense layers with ReLU activation. The final output layer uses softmax for multi-class fruit classification.

    Training: The model was compiled with the Adam optimizer and trained to recognize 10 different types of fruits.

2. Frontend (Streamlit Application):

    Interface: Built with Streamlit to provide a simple and user-friendly web interface.

    User Input: Allows users to either upload an image file (JPG, PNG) or use their desktop camera to capture a photo of a fruit.

    Prediction: The app preprocesses the input image (resizing to 224x224, normalizing pixel values) and feeds it into the loaded .h5 model to get a prediction.

    Nutrition Data: After identifying the fruit, the app looks up its nutritional information from an included CSV file and displays key values like energy, protein, vitamins, and minerals in an easy-to-read table.

# 🗂️ Project Structure

I've renamed demo(5).py to app.py and the notebook for clarity, which is a common practice.

├── app.py
├── Fruity_Nutrition_Model_Training.ipynb 
├── my_model.h5                      
├── fruitsnutrition.csv                
├── requirements.txt                  
└── README.md                        

# ▶️ Getting Started

    Clone the Repository
    Bash

git clone https://github.com/your-username/fruityvision.git
cd fruityvision

Create an Environment and Install Requirements
Bash

pip install -r requirements.txt

Run the App Locally
Bash

    streamlit run app.py

# 🧪 Model Notebook (Fruity_Nutrition_Model_Training.ipynb)

Explore the full model development process in the Jupyter Notebook. Key steps include:

    Loading and augmenting the image dataset.

    Building the deep learning model using TensorFlow/Keras with a ResNet50V2 base.

    Training and validating the model's accuracy.

    Testing the model on sample images to verify its performance.

    Saving the final model to my_model.h5.

# 💡 Features

    Fruit Recognition: Identifies 10 different fruits from an image.

    Dual Input: Supports both file uploads and live camera captures.

    Nutritional Info: Displays key nutritional data for the predicted fruit.

    Confidence Score: Shows the model's confidence in its prediction.

    Interactive UI: Clean, simple, and responsive user interface built with Streamlit.

# 📌 Dependencies

    streamlit

    tensorflow

    pandas

    numpy

    opencv-python

    Pillow

Install all dependencies with:
Bash

pip install -r requirements.txt
