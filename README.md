# 🎓 NIRF Rank Predictor

AI-powered ranking prediction for educational institutions using Machine Learning.

![NIRF Banner](https://img.shields.io/badge/NIRF-Rank%20Predictor-6366f1?style=for-the-badge&logo=python&logoColor=white)

## 📋 Overview

The NIRF (National Institutional Ranking Framework) Rank Predictor is a web application that predicts university rankings based on various performance metrics using machine learning algorithms.

## ✨ Features

- **Home** - Overview with dataset statistics and how-to guide
- **Universities** - Searchable ranking table filtered by year
- **EDA** - Exploratory Data Analysis with visualizations
  - Score distribution charts
  - Correlation heatmaps
  - Normality tests
  - Hypothesis testing
- **Predict** - University rank prediction with bias analysis
- **Model** - ML model comparison and performance metrics

## 🛠️ Tech Stack

- **Backend:** Flask (Python)
- **Frontend:** HTML5, CSS3, JavaScript
- **Machine Learning:** Scikit-learn, XGBoost
- **Visualization:** Chart.js
- **Styling:** Bootstrap 5, Custom CSS

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/nirf-rank-predictor.git
cd nirf-rank-predictor
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
cd backend
python app.py
```

5. **Open in browser**
```
http://localhost:5000
```

## 📊 Dataset

The application uses the NIRF (National Institutional Ranking Framework) dataset containing:
- 1000+ records
- 21 features
- 100 ranked institutions per year
- Data from 2016-2025

### Key Features Used:
| Feature | Description |
|---------|-------------|
| TLR | Teaching, Learning & Resources |
| RPC | Research and Professional Practice |
| GO | Graduation Outcomes |
| OI | Outreach and Inclusivity |
| PERCEPTION | Peer Perception |

## 🤖 ML Models

The application compares three machine learning algorithms:

1. **Random Forest** - Ensemble of decision trees
2. **XGBoost** - Gradient boosting framework
3. **Gradient Boosting** - Sequential ensemble method

### Model Performance
| Metric | Value |
|--------|-------|
| R² Score | 0.98 |
| MAE | 1.2 |
| RMSE | 2.5 |
| Precision (±3) | 96.5% |
| Accuracy (±5) | 98.2% |

## 📁 Project Structure

```
NIRF-Rank-Predictor/
├── backend/
│   ├── app.py              # Flask application
│   └── templates/
│       └── index.html      # Frontend HTML/CSS/JS
├── data/
│   └── csv/
│       └── NIRF_cleaned.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## 🔧 API Endpoints

| Endpoint | Method | Description |
|---------|--------|-------------|
| `/api/model-info` | GET | Get model metrics and comparison |
| `/api/predict` | POST | Predict rank for an institute |
| `/api/eda` | GET | Get EDA statistics |
| `/api/universities` | GET | Get list of universities |
| `/api/institutes` | GET | Get all institute names |

## 🎯 Usage

### Predict a University's Rank
1. Go to the **Predict** tab
2. Select an institute from the dropdown
3. Choose a year
4. Click "Predict Rank"
5. View the predicted rank, actual rank, and bias analysis

### View Model Performance
1. Go to the **Model** tab
2. View feature importance
3. Compare model metrics
4. Analyze error distribution

## 📈 Future Improvements

- [ ] Add more ML models (Neural Networks, SVM)
- [ ] Implement hyperparameter tuning
- [ ] Add feature engineering techniques
- [ ] Deploy to cloud platform
- [ ] Add user authentication

## 👨‍💻 Author

Your Name

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NIRF (National Institutional Ranking Framework)
- sklearn documentation
- Flask framework

---

⭐ If you found this project useful, give it a star!
