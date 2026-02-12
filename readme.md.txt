# 🛒 Flipkart Review Sentiment Analysis with MLflow & MLOps

**NLP · Machine Learning · MLflow · Streamlit · MLOps**

---

## 🔍 Overview

The **Flipkart Review Sentiment Analysis Project** is an end-to-end **Natural Language Processing (NLP)** and **Machine Learning** solution designed to analyze customer reviews and classify sentiments as **Positive** or **Negative**.

The project adopts **MLOps best practices** by integrating **MLflow** for experiment tracking, model comparison, and reproducibility, and provides an interactive **Streamlit web application** for real-time sentiment prediction.

---

## 🖥 Application Preview

* Interactive Streamlit web app
* Real-time sentiment prediction from user-entered reviews
* Clean and minimal UI for business users

---

## 🎯 Business Objective

* Automatically analyze customer sentiment from product reviews
* Reduce manual review analysis effort
* Identify customer satisfaction trends
* Support data-driven decisions in e-commerce platforms

---

## 📊 Dataset

* **Dataset Name:** Flipkart Product Reviews
* **Total Records:** ~8,500 reviews
* **Features:**

  * Review Text
  * Sentiment Label
  * Product-related metadata
* **Target Variable:** Sentiment (Positive / Negative)
* **Source:** Publicly available e-commerce review dataset

---

## 🛠 Tools & Technologies

* **Programming Language:** Python
* **Data Analysis:** Pandas, NumPy
* **NLP:** TF-IDF Vectorization
* **Machine Learning:** Scikit-learn
* **MLOps:** MLflow (Tracking & Model Registry)
* **Model Persistence:** Pickle
* **Web App:** Streamlit

---

## 🗂 Project Architecture

```
sentiment-analysis-project/
│
├── app/
│   └── app.py                  # Streamlit application
│
├── data/
│   └── flipkart_reviews.csv    # Dataset
│
├── images/
│   └── app_preview.png
│
├── model_building/
│   └── train_with_mlflow.py    # Model training & MLflow logging
│
├── notebooks/
│   └── sentiment_eda.ipynb     # EDA & preprocessing
│
├── artifacts/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧹 Data Preprocessing

To improve model performance, the following NLP steps were applied:

* Handling missing values
* Text normalization (lowercasing)
* Removal of special characters & punctuation
* Tokenization
* Stopword removal
* Lemmatization

These steps significantly enhanced feature quality and sentiment classification accuracy.

---

## 🤖 Model Development & Evaluation

### Feature Engineering

* **TF-IDF Vectorizer**

  * `max_features = 5000`
  * `ngram_range = (1, 2)`

### Models Implemented

* Logistic Regression
* Multinomial Naive Bayes
* Linear Support Vector Machine (SVM)
* Random Forest Classifier

### Evaluation Metric

* **F1 Score** (to handle class imbalance effectively)

### Final Selected Model

* **Linear SVM** (best-performing model)

---

## 🔄 MLflow Integration (MLOps)

MLflow was used to ensure **experiment reproducibility and model governance**.

### MLflow Features Used

* Experiment tracking for multiple algorithms
* Logging hyperparameters & evaluation metrics
* Comparing model performance visually
* Saving trained models as artifacts
* Model versioning using MLflow Model Registry

### Registered Model

* **Model Name:** `FlipkartSentimentModel`
* Version-controlled for lifecycle management

---

## 📈 Key Insights

* Linear SVM achieved the highest F1 score
* TF-IDF with bigrams improved contextual understanding
* NLP preprocessing had a major impact on model performance
* MLflow simplified model comparison and selection

---

## 💼 Business Value

This solution helps organizations to:

* Analyze customer sentiment at scale
* Identify improvement areas from negative feedback
* Enhance customer satisfaction
* Reduce manual review analysis cost
* Maintain reproducible and auditable ML pipelines

---

## ▶ How to Run the Project

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/flipkart-sentiment-analysis.git
cd flipkart-sentiment-analysis
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train Model with MLflow

```bash
python model_building/train_with_mlflow.py
```

### 4️⃣ Launch MLflow UI

```bash
mlflow ui
```

Open browser at:

```
http://127.0.0.1:5000
```

### 5️⃣ Run Streamlit App (Optional)

```bash
streamlit run app/app.py
```

---

## 👤 Author

**Mahesh Bodhankar**
Aspiring Data Analyst / Data Scientist
Skills: Python | SQL | Machine Learning | NLP | MLflow | Power BI

🔗 GitHub: *Add your GitHub link*
🔗 LinkedIn: *Add your LinkedIn link*

---

## 🙏 Acknowledgment

Special thanks to **Innomatics Research Labs** for providing hands-on, industry-focused training and continuous mentorship throughout this project.

---