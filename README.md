# 💓 Smart Health Risk Predictor

A machine learning-powered Streamlit app that predicts the risk of heart disease using user input data. Built using a trained Random Forest Classifier and deployed locally or on Hugging Face Spaces.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kxJ7KRdw2G6yw3kKHPFUipTLFUIIY8HM)

---

## 🚀 Features

- ✔️ Predicts heart disease risk based on user input
- ✔️ Random Forest Classifier trained on UCI dataset
- ✔️ One-hot encoded categorical features
- ✔️ Clean, interactive Streamlit UI
- ✔️ Ready for deployment (Hugging Face/Streamlit Cloud)

---

## 📊 Model Details

- Trained on 303 records (UCI Heart Disease dataset)
- 19 engineered features using one-hot encoding
- Evaluation:
  - Accuracy: ~86%
  - Precision, Recall, F1-score evaluated
- Saved model: `best_random_forest_model.pkl`

---

## 🔗 Notebook Link

- 📓 **[Click here to view the training notebook in Colab](https://colab.research.google.com/drive/1kxJ7KRdw2G6yw3kKHPFUipTLFUIIY8HM)**

---

## 🛠 Tech Stack

| Tool | Usage |
|------|-------|
| `scikit-learn` | ML model training |
| `Streamlit` | Frontend interface |
| `joblib` | Model serialization |
| `pandas`, `numpy` | Data preprocessing |

---

## 🖥️ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/faizan102418/SmartHealthPredictor.git
cd SmartHealthPredictor

# (Optional) Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
