# 🚇 NYC Subway AI Predictor

An end-to-end machine learning system that predicts subway delays using real-time data, hybrid dataset engineering, and ensemble learning — with live visualization.

---

## 🔗 Live Demo
https://nyc-subway-predictor-gyb4p46zfl6fnqexgle3df.streamlit.app/

---

## 📌 Overview

This project builds an AI-powered system to:

- Predict subway delays in real-time  
- Use multiple machine learning models for forecasting  
- Visualize predictions interactively  
- Simulate live train movement on a map  

Unlike traditional systems that show **current status**, this system focuses on **predicting future delays**.

---

## 🧠 Machine Learning Approach

### Models Used

- Linear Regression  
- Random Forest Regressor  
- XGBoost Regressor  
- LightGBM Regressor  

### Final Model: Ensemble Learning

The final prediction is obtained by averaging outputs from all models.

**Why Ensemble?**
- Combines strengths of different models  
- Reduces overfitting and variance  
- Produces more stable and reliable predictions  

---

## 📊 Evaluation Metrics

The following metrics were used to evaluate performance:

- **MAE (Mean Absolute Error)** → ~60 seconds  
- **RMSE (Root Mean Squared Error)** → ~69 seconds  
- **R² Score** → Low (due to dataset constraints)  
- **Explained Variance**

> The model performs reasonably well for average delay prediction, but performance is limited by dataset quality and feature relationships.

---

## 📁 Dataset

### Type:
- Hybrid Dataset  
  - Time-series  
  - Tabular  
  - Simulated + real-time  

### Features Used:
- Hour of day  
- Stop sequence  
- Previous delay (temporal dependency)  

---

## ⚠️ Dataset Challenges

- No fully labeled real-world dataset  
- Missing scheduled vs actual times  
- Sparse real-time data  

### Solution

A **hybrid dataset approach** was implemented:

- Simulated realistic delays  
- Incorporated rush-hour patterns  
- Modeled delay propagation across stops  

This allowed the model to learn temporal behavior despite limited real data.

---

## ⚙️ System Architecture



---

## 🌐 Web Application

Built using **Streamlit**, the application includes:

- Interactive user inputs (sliders)  
- Real-time delay prediction  
- Estimated arrival time  
- Delay distribution visualization  
- Live subway map  

---

## 🗺️ Map Visualization

- Uses GTFS subway data (`stops.txt`)  
- Displays subway station locations  
- Simulates live train movement  
- Color-coded delays:

| Delay Level | Color |
|------------|------|
| Low        | Green |
| Medium     | Orange |
| High       | Red |

---

## 🚀 Deployment

The application is deployed using **Streamlit Cloud**.

---

## ⚠️ Challenges & Fixes

During development, several issues were encountered and resolved:

- **File path issues during deployment**  
  → Fixed by restructuring project directories  

- **Missing GTFS map data**  
  → Resolved by correctly placing `stops.txt` inside `gtfs_subway/`  

- **Data type issues (NumPy vs Python types)**  
  → Fixed using explicit type conversions  

- **Weak model performance (low R² score)**  
  → Improved using feature engineering and ensemble learning  

These improvements ensured a stable and functional system.

---

## 📉 Limitations

- Uses partially simulated data  
- Limited real-world validation  
- Weak feature correlation  
- Low R² score  

---

## 🔮 Future Improvements

- Integrate full MTA real-time API  
- Apply deep learning models (LSTM for time-series)  
- Add route-level and station-level predictions  
- Improve dataset quality and feature richness  
- Scale system for production deployment  

---

## 💡 Key Takeaway

This project demonstrates how:

> Machine Learning + Data Engineering + Visualization  
can be combined to build a real-world intelligent prediction system.

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- LightGBM  
- Streamlit  
- PyDeck  

---

## 👤 Author

**Nydia Takhellambam**
