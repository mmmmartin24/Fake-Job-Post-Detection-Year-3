

<h2> 🕵️ Real/Fake Job Posting Detection </h2>

A machine learning project to detect fraudulent job postings using natural language processing (NLP) and various classification algorithms. This project was developed as part of the IBDA2032 - Kecerdasan Buatan course at Calvin Institute of Technology.


📌 Problem Statement
With the rise of digital job platforms, fake job postings have become a serious issue. These deceptive listings can lead to wasted time, financial loss, and loss of trust in job platforms. This project aims to build a system that can identify and flag potentially fraudulent job listings to protect job seekers and maintain platform integrity.

📁 Dataset
- Source: Kaggle (https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- Description: Contains 17,014 real and 866 fake job listings.
- Features Used: title, location, salary_range, company_profile, description, requirements, benefits, employment_type, required_experience, required_education, industry, function, department.

🔍 Preprocessing
Steps included:
- Dropping irrelevant or low-correlation features (telecommuting, has_questions)
- Handling missing values using imputation
- Feature engineering: extracting country from location, combining multiple text fields into combined_text
- Text cleaning (lowercasing, HTML tag removal, stopwords, etc.)
- Normalization using lemmatization
- Feature extraction with POS tagging and CountVectorizer

📊 Exploratory Data Analysis (EDA)
Visualizations included:
- 🔥 Heatmaps (feature correlation)
- 📊 Bar and Pie Charts (label distribution, experience, education, etc.)
- ☁️ Word Clouds for real vs fake postings
- 🧩 Sparsity pattern visualizations for understanding text matrix structure

🤖 Machine Learning Models
Models Implemented:
- Logistic Regression	Simple, fast, interpretable
- Multinomial Naive Bayes	Good for text classification
- SVM (RBF Kernel)	Handles non-linear separation
- Neural Network (MLP)	Custom layer architecture
- Extra Trees Classifier	Ensemble with fast training
- Random Forest	Robust and generalized

Training Strategy:
- Train-test split (80:20)
- Applied K-Fold and Stratified K-Fold Cross Validation
- Evaluation Metrics: Accuracy, Precision, Recall, F1-Score

📈 Results Summary
The project compares multiple algorithms to determine the most effective model for detecting fake postings. Metrics such as accuracy and F1-score were calculated using both traditional and stratified cross-validation methods.

🛠 Tools & Libraries
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- NLTK & SpaCy for NLP
- Jupyter Notebook

👨‍💻 Authors
- Martin Emmanuel Chang
- Mikha Aldyn Yauw
- Moses Anthony Kwik
- Pixel Ariel Christopher

📚 References
- https://www.kaggle.com/code/seifwael123/real-fake-jobs-eda-modelling-99/notebook
- https://towardsdatascience.com/top-machine-learning-algorithms-for-classification-2197870ff501
- https://www.datacamp.com/blog/classification-machine-learning

📂 How to Run
1. Clone this repo:
git clone https://github.com/yourusername/real-fake-job-posting-detector.git
cd real-fake-job-posting-detector
2. Install dependencies:
pip install -r requirements.txt
3. Run the Jupyter Notebook:
jupyter notebook

