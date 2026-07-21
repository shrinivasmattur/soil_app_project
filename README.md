



a8e2e62d-0666-4b32-bf11-6c3d164bb47c.png

Based on your project (Smart Soil Testing with Plant Recommendations and Leaf Disease Detection), here are polished answers you can use for your hackathon submission.

Problem Statement
Farmers often struggle to identify the right crops for their soil and detect plant diseases at an early stage. Traditional soil testing requires laboratory analysis, which is time-consuming, expensive, and not always accessible in rural areas. Similarly, farmers may not have access to agricultural experts for timely disease diagnosis, leading to crop loss and reduced productivity.

This problem primarily affects small and medium-scale farmers who rely on accurate soil and crop recommendations to improve yield and income. Providing an AI-powered, accessible solution can help farmers make informed decisions, reduce losses, and promote sustainable cultivation of indigenous fruit crops.

Solution Description
Our solution is an AI-powered web application that performs soil analysis, recommends suitable indigenous fruit crops, and detects leaf diseases from uploaded images.

How the solution works
The user uploads a soil image.

The system analyzes soil characteristics using image processing techniques.

The user selects a location on the Karnataka map.

Soil properties are compared with typical soil types of different districts.

Based on soil characteristics, location, and seasonal information, the application recommends the most suitable fruit crop among:

Ramphal (Annona reticulata)

Lakshmanaphala (Annona muricata)

Wood Apple (Limonia acidissima)

The user can upload a leaf image.

A trained AI model detects possible leaf diseases and provides treatment recommendations.

Key Features
AI-based soil image analysis

Indigenous fruit crop recommendation

Karnataka district-based soil comparison

Interactive location selection

Leaf disease detection

Disease cure recommendations

Simple and user-friendly Streamlit interface

How AI is used
Image classification for soil analysis

Deep Learning model for leaf disease detection

AI-based recommendation logic combining:

Soil characteristics

Geographic location

Seasonal suitability

Crop requirements

Why the solution is effective
Eliminates the need for preliminary laboratory soil testing.

Enables early disease detection.

Promotes cultivation of indigenous fruit crops.

Supports sustainable agriculture.

Easy to use from any internet-connected device.

Tech Stack / AI Models / Tools Used
Frontend

Streamlit

Backend

Python

Libraries

TensorFlow

OpenCV

NumPy

Pandas

Pillow

Geopy

Scikit-learn

AI / Machine Learning

Convolutional Neural Network (CNN) for leaf disease detection

Image Processing

TensorFlow/Keras

Development Tools

Git

GitHub

VS Code

GitHub Repository Link
Paste your public repository, for example:

https://github.com/shrinivasmattur/soil_app_project
Ensure it includes:

README.md

Installation instructions

requirements.txt

Project screenshots

Live Deployment Link
Paste your Streamlit Cloud link, for example:

https://your-app-name.streamlit.app
Demo Video
Upload a 2–3 minute video to YouTube (Unlisted) or Google Drive.

Video Structure
0:00–0:20

Introduce yourself.

Explain the problem.

0:20–0:50

Explain your solution.

0:50–1:50

Live demonstration:

Upload soil image

Show crop recommendation

Select district

Upload leaf image

Show disease detection

1:50–2:20

Explain the AI components:

Soil image analysis

CNN disease detection

Recommendation logic

2:20–3:00

Expected impact:

Helps farmers

Encourages indigenous fruit cultivation

Early disease detection

Sustainable agriculture

This content is suitable for hackathon submissions and clearly highlights the AI aspects of your project.


how to add screen shots

If you want to add screenshots to your GitHub README, follow these steps.

Step 1: Take screenshots
Capture screenshots of:

Home page

Soil analysis page

Plant recommendation result

Leaf disease detection result

Save them as:

home.png
soil_analysis.png
recommendation.png
disease_detection.png
Step 2: Create a folder
Inside your project, create a folder named:

soil_app_project/
│── screenshots/
│    ├── home.png
│    ├── soil_analysis.png
│    ├── recommendation.png
│    └── disease_detection.png
│── README.md
Step 3: Edit README.md
Add this section:

# Screenshots

## Home Page

![Home](screenshots/home.png)

## Soil Analysis

![Soil Analysis](screenshots/soil_analysis.png)

## Plant Recommendation

![Recommendation](screenshots/recommendation.png)

## Leaf Disease Detection

![Disease Detection](screenshots/disease_detection.png)
GitHub will automatically display the images.

Step 4: Push to GitHub
git add .
git commit -m "Added project screenshots"
git push origin main
For your hackathon form
You don't upload screenshots separately. They should be included in your public GitHub repository README so judges can see them.

If you don't have a README yet
I can generate a professional README.md with:

Project description

Features

Installation steps

Tech stack

Screenshots section

Demo video link

Live deployment link

Future scope

It will be ready to upload directly to GitHub.


k

Here's a professional README.md for your project. You can copy and paste it into a file named README.md in your GitHub repository.

# 🌱 Smart Soil Testing with Plant Recommendations and Leaf Disease Detection

An AI-powered web application that analyzes soil images, recommends suitable indigenous fruit crops, and detects leaf diseases using Deep Learning.

---

## 📖 Overview

This project helps farmers identify suitable indigenous fruit crops based on soil characteristics and location. It also detects leaf diseases from uploaded images and provides treatment recommendations.

The application promotes sustainable agriculture by encouraging the cultivation of indigenous fruit crops like:

- 🌿 Ramphal (*Annona reticulata*)
- 🌿 Lakshmanaphala (*Annona muricata*)
- 🌿 Wood Apple (*Limonia acidissima*)

---

## ✨ Features

- 📷 Soil Image Analysis
- 🌍 Karnataka District Soil Comparison
- 📍 Interactive Location Selection
- 🌱 Smart Plant Recommendation
- 🍃 Leaf Disease Detection
- 💊 Disease Cure Suggestions
- 📊 User-Friendly Streamlit Interface

---

## 🧠 AI Features

- Image Processing
- CNN-based Leaf Disease Detection
- Soil Analysis
- AI-based Crop Recommendation
- Seasonal Crop Recommendation

---

## 🛠 Tech Stack

### Frontend
- Streamlit

### Backend
- Python

### AI & ML
- TensorFlow
- Keras
- OpenCV

### Libraries
- NumPy
- Pandas
- Pillow
- Geopy
- Scikit-learn

### Tools
- Git
- GitHub
- VS Code

---

## 📂 Project Structure

```
soil_app_project/
│
├── soil_app.py
├── app_info.py
├── disease_detection.py
├── requirements.txt
├── README.md
├── models/
├── images/
└── screenshots/
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/shrinivasmattur/soil_app_project.git
```

Go to project folder

```bash
cd soil_app_project
```

Create virtual environment

```bash
python -m venv venv
```

Activate virtual environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run soil_app.py
```

---

## 📷 Screenshots

### Home Page

![Home](screenshots/home.png)

### Soil Analysis

![Soil Analysis](screenshots/soil_analysis.png)

### Plant Recommendation

![Recommendation](screenshots/recommendation.png)

### Leaf Disease Detection

![Disease Detection](screenshots/disease_detection.png)

---

## 🎥 Demo Video

Add your video link here

```
https://youtu.be/your-video-link
```

---

## 🌐 Live Demo

Add your Streamlit Cloud link here

```
https://your-app.streamlit.app
```

---

## 📈 Future Enhancements

- AI Chatbot for farmers
- Live Weather API Integration
- Market Price Prediction
- Fertilizer Recommendation
- Mobile Application
- Multi-language Support
- AI Agent for Personalized Farming Advice

---

## 👨‍💻 Developer

**Shrinivas Mattur**

Information Science & Engineering

National Institute of Engineering (NIE), Mysuru

---

## 📄 License

This project is developed for educational and hackathon purposes.
Add screenshots
Create a folder named screenshots in your project.

Save screenshots with these names:

home.png

soil_analysis.png

recommendation.png

disease_detection.png

Commit and push the changes:

git add .
git commit -m "Added README and screenshots"
git push origin main
This will make your GitHub repository look polished and ready for hackathon judging.


