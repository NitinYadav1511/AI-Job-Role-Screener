# AI Job Role Screener

This project is an AI-powered resume screening tool that predicts the job category based on the contents of a resume. It utilizes natural language processing (NLP) techniques to clean and process resumes and classify them into predefined job roles using a machine learning model.

## Project Overview

In the modern hiring process, screening resumes manually can be time-consuming and prone to human error. This tool automates the resume screening process, allowing recruiters to upload resumes in PDF format and receive predictions on the most relevant job role for each candidate based on their skills, qualifications, and experiences.

## Features

- **Resume Upload**: Upload resumes in PDF format
- **AI-based Prediction**: Predicts the most suitable job category based on resume content
- **Machine Learning Model**: Uses a machine learning model trained on resume data with various job categories
- **Text Cleaning and Processing**: Cleans resume text using regular expressions and NLP techniques before classification
- **Supports Multiple Job Roles**: Classifies resumes into various roles, including Data Science, Web Designing, Java Developer, and more

## Technology Stack

- **Python**: Core language for backend processing
- **Streamlit**: Used to build the web interface for uploading resumes and displaying predictions
- **scikit-learn**: Machine learning library used for building the classification model
- **Natural Language Toolkit (nltk)**: Used for text processing and cleaning
- **TfidfVectorizer**: Converts resume text to a matrix of TF-IDF features
- **K-Nearest Neighbors Classifier**: Used as the classification algorithm
- **Pickle**: Used to save the model and vectorizer for reuse

## Dataset

The project uses a pre-processed dataset of resumes categorized into multiple job roles. The resumes were cleaned and transformed using the TfidfVectorizer to extract relevant features before being fed into the machine learning model.

## How It Works

1. **Resume Upload**: Users upload a resume in PDF format via the web interface
2. **Text Processing**: The uploaded resume is cleaned, and special characters, URLs, and unnecessary content are removed
3. **Feature Extraction**: The cleaned resume text is vectorized using the TfidfVectorizer
4. **Prediction**: The vectorized text is passed through a pre-trained machine learning model to predict the job category
5. **Display**: The predicted job role is displayed to the user

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/NitinYadav1511/AI-Job-Role-Screener.git
cd AI-Job-Role-Screener
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run main.py
```

4. Open the provided URL in your browser to use the web app

## Model Training

The machine learning model is trained using the KNeighborsClassifier with the OneVsRestClassifier strategy for multi-class classification. The resumes are transformed into TF-IDF features before being classified into one of several job categories.

## Usage

1. Launch the app and upload your resume (in PDF format)
2. The application will process the resume, clean it, and predict the job role
3. The predicted job category will be displayed on the screen

## Job Categories

The model can classify resumes into the following job categories:

- Java Developer
- Testing
- DevOps Engineer
- Python Developer
- Web Designing
- HR
- Hadoop
- Blockchain
- ETL Developer
- Operations Manager
- Data Science
- Sales
- Mechanical Engineer
- Arts
- Database
- Electrical Engineering
- Health and Fitness
- PMO
- Business Analyst
- DotNet Developer
- Automation Testing
- Network Security Engineer
- SAP Developer
- Civil Engineer
- Advocate

## Future Enhancements

- Adding support for more job categories
- Improving the model's accuracy by training on larger datasets
- Extending the tool to extract key information like candidate's skills, experience, and education

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss changes or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

Special thanks to the open-source libraries and frameworks that made this project possible.
