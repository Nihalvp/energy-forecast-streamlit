# ⚡ Energy Production Forecast Web App

An interactive Machine Learning web application that forecasts **next-year energy production** using multiple regression models.

🔗 **Live Demo:** https://energy-forecast-app-azay3yryxedakf449jwdgo.streamlit.app/

---

## 📌 Project Overview

This project predicts:

- Total non-renewable energy (next year)
- Total renewable energy (next year)
- Total energy production (next year)

Users can enter current-year energy values and compare predictions across:

- Linear Regression
- Random Forest (Final Model)
- Tuned Random Forest

---

## 🎯 Key Features

✅ Interactive prediction interface  
✅ Automatic renewable & non-renewable calculation  
✅ Multi-model comparison  
✅ Visual prediction charts  
✅ Actual vs Predicted evaluation plots  
✅ Deployed Streamlit web application  

---

## 🧠 Machine Learning Approach

Separate models were trained for each target:

- Non-renewable energy (next year)
- Renewable energy (next year)
- Total energy value (next year)

Random Forest was selected as the final model based on higher R² performance and its ability to capture nonlinear energy trends.

---

## 📊 Model Evaluation

Although Linear Regression showed smoother visual fit in scatter plots, Random Forest achieved better statistical performance (R²), highlighting the importance of combining visual and quantitative evaluation.

---

## 🛠 Tech Stack

- Python
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Plotly
- Streamlit (Deployment)

---

## 🚀 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py




