# Video KYC Automation using Computer Vision & LLMs

This project is an automated Know Your Customer (KYC) verification system that processes a video feed to verify a user's identity. The application uses computer vision models to detect the user's face and their Aadhar card (an Indian identification card), compares the two for a match, and employs a Large Language Model (LLM) to extract and parse the information from the ID card.

The entire application is wrapped in a user-friendly web interface built with Streamlit.

## ✨ Features

- **End-to-End Video Processing**: Simply upload a video of a user presenting their ID card and face.
- **AI-Powered ID Detection**: Utilizes a YOLO model to automatically detect the front and back of the ID card in video frames.
- **Face Detection & Verification**: Employs a YOLO face detector and the DeepFace library to perform high-accuracy face matching between the user and their ID photo.
- **Smart Information Extraction**: Leverages Google's Gemini LLM to accurately parse and structure the data (Name, DOB, Aadhar No., Address) from the ID card images.
- **Interactive Web UI**: A clean and simple interface powered by Streamlit for video uploads, processing, and viewing results.
- **Verification History**: Automatically saves each verification run in a local SQLite database for auditing and review.
- **PDF Report Generation**: Download a comprehensive PDF report for each KYC check, including captured images, verification status, and extracted data.

## ⚙️ How It Works

The application follows a multi-stage pipeline to perform the KYC verification:

1. **Video Ingestion**: The user uploads a video file through the Streamlit web interface.

2. **Frame Analysis**: The video is processed frame by frame. On sampled frames, the system runs detection models.

3. **ID & Face Detection**:
   - A custom-trained YOLO model detects the ID card. The system intelligently differentiates between the front (with a face) and the back.
   - A separate YOLO model detects the user's face in the video and the face on the ID card.

4. **Best Crop Selection**: To ensure accuracy, the system analyzes all detected crops and selects the highest-quality image of the user's face, the front of the ID, and the back of the ID based on sharpness and size.

5. **Face Recognition**: The face crop from the ID card is compared against the user's face crop using the DeepFace library, which generates a similarity score and a match/no-match result.

6. **Data Extraction (OCR with LLM)**: The high-quality front and back ID card images are sent to the Google Gemini API, which reads the text and returns a structured JSON object with the user's details.

7. **Results Display & Storage**: The final results—match status, similarity score, extracted data, and captured images—are displayed in the web UI and saved to the local database for future reference.

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Python 3.10
- Git
- Access to Google AI Studio to get a Gemini API Key

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies

The required Python packages are listed in requirements.txt. Install them using pip:

```bash
pip install -r requirements.txt
```

### 3. Download the AI Models

This project requires two pre-trained YOLO models for ID and face detection. Download them and place them in the root directory of the project.

- **ID Detection Model**: `id_detector.pt`
- **Face Detection Model**: `yolov8x-face.pt`

*ask me if wanted*

### 4. Set Up Environment Variables

The application requires an API key for the Google Gemini LLM to extract text from the ID cards.

1. Create a file named `.env` in the root of the project directory.
2. Add your API key to the file as follows:

```env
GEMINI_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"
```
3. **Ask this repository owner for the pretrained model**

## ▶️ Running the Application

Once you have completed the setup, you can run the Streamlit application with a single command:

```bash
streamlit run app.py
```

This will start the web server and open the application in your default web browser.

### How to Use the App

1. **Navigate to the "KYC Processor" Tab**:
   - **Upload a Video**: Click "Choose a video file" and select a video (.mp4, .mov, .avi). The video should clearly show the user's face and both the front and back of their Aadhar card.
   - **Process**: Click the "Process Video" button. The backend will start the analysis.
   - **View Results**: Once processing is complete, the results will be displayed on the page, showing the face match status, similarity score, images, and extracted ID data.

2. **Check History**: Navigate to the "Run History" tab to see a list of all past verifications. You can expand each entry to see the details and download a PDF report.

## 📁 Project Structure

```
.
├── .env                  # Environment variables (needs to be created)
├── app.py                # Main Streamlit application file
├── id_detector.pt        # Model for ID card detection (needs to be downloaded)
├── yolov8x-face.pt       # Model for face detection (needs to be downloaded)
├── kyc_history.db        # SQLite database for storing run history
├── requirements.txt      # Project dependencies
└── src/
    ├── __init__.py
    ├── main.py             # Core VideoKYC processing logic
    ├── config.py           # Configuration settings (model paths, thresholds)
    ├── database.py         # Database initialization and functions
    ├── detector_nd_recognizer.py # Classes for ID/Face detection and recognition
    ├── Id_parser.py        # Class for parsing ID data using LLM
    ├── logger.py           # Logging configuration
    ├── models.py           # Pydantic models for data structure
    ├── prompt_template.py  # LangChain prompt template for the LLM
    └── utils.py            # Utility functions (image processing, etc.)
```
