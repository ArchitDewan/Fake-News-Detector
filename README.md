# Fake News Detector

## Overview
The **Fake News Detector** is a machine learning project designed to classify news articles as either **REAL** or **FAKE**. By leveraging Natural Language Processing (NLP) techniques and a Linear Support Vector Classifier (LinearSVC), this tool efficiently identifies misleading information in text data.

## Features
✅ Detects fake news articles with high accuracy using advanced text vectorization techniques.  
✅ Utilizes the **TF-IDF Vectorizer** for feature extraction from text.  
✅ Implements a **LinearSVC** model for fast and effective classification.  
✅ Supports new article predictions directly from text files.  
✅ Easy-to-use interface for improved user experience.  

## Installation
To get started, clone this repository and install the necessary dependencies:

```bash
# Clone the repository
git clone https://github.com/yourusername/Fake-News-Detector.git
cd Fake-News-Detector

# Install required packages
pip install -r requirements.txt
```

## Usage
### 1. Train the Model
The model is trained using a dataset of labeled articles. Run the following command to train the model:

```bash
python main.py
```

### 2. Predict Fake or Real News
To test the model on custom text, create a text file (e.g., `mytext.txt`) with your content. Then run:

```bash
python main.py
```

The model will output either `FAKE` or `REAL` based on its prediction.

### Example Code Snippet
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pandas as pd

# Load and preprocess data
df = pd.read_csv('data/fake_or_real_news.csv')
df['label_binary'] = df['label'].map({'REAL': 0, 'FAKE': 1})

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])

# Train the Model
model = LinearSVC()
model.fit(X, df['label_binary'])

# Prediction Example
with open('mytext.txt', 'r') as f:
    text = f.read()
pred = model.predict(vectorizer.transform([text]))
print('FAKE' if pred[0] == 1 else 'REAL')
```

## Dataset
The model is trained using the **fake_or_real_news.csv** dataset, which contains labeled articles categorized as `REAL` or `FAKE`. This dataset is publicly available for educational and research purposes.

## Key Libraries Used
- **Scikit-learn** for model training and evaluation.  
- **Pandas** for data manipulation.  
- **NumPy** for numerical operations.  

## Contributing
Contributions are welcome! If you'd like to improve this project, please follow these steps:
1. Fork the repository.  
2. Create a new branch (`git checkout -b feature/your-feature`).  
3. Commit your changes (`git commit -m "Add feature XYZ"`).  
4. Push to the branch (`git push origin feature/your-feature`).  
5. Create a pull request for review.  

## License
This project is licensed under the **MIT License**. Feel free to modify and distribute it as needed.

## Contact
For questions, suggestions, or contributions, please contact: **architdewan2006@gmail.com**

# Fake-News-Detector
