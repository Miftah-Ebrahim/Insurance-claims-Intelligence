
<div align="center">

<img src="https://via.placeholder.com/1200x300/1e88e5/ffffff?text=AlphaCare+Insurance+Intelligence" alt="AlphaCare Header" width="100%" />

# 🛡️ Beyond the Claim
### *Transforming Insurance Data into Fair & Predictive Intelligence*

[![Status](https://img.shields.io/badge/Pipeline-Production--Ready-success?style=for-the-badge&logo=github-actions)](https://github.com/Miftah-Ebrahim/Insurance-claims-Intelligence)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)
[![DVC](https://img.shields.io/badge/Data-Versioned%20(DVC)-orange?style=for-the-badge&logo=dvc)](https://dvc.org/)

[**Explore Analysis**](notebooks/01_EDA.ipynb) · [**View Hypotheses**](notebooks/Task_3_Hypothesis_Testing.ipynb) · [**See The Models**](notebooks/Task_4_Modeling.ipynb) · [**Read Final Report**](reports/final_report/Final_Project_Report.md)

</div>

---

## 📖 The Mission

**Insurance isn't just about paying claims; it's about understanding risk.**

At **AlphaCare Insurance Solutions**, we sat on a treasure trove of data: over 37,000 policies and millions in transactions. But data without insight is just noise. Our mission was to silence that noise and listen to what the data was telling us.

We asked three critical questions:
1.  *Where are our biggest risks hiding?*
2.  *Are our premiums actually covering our costs?*
3.  *Is our pricing fair to everyone, regardless of gender?*

This repository is the answer to those questions—a production-grade machine learning pipeline that turns raw rows into actionable business intelligence.

---

## 🔍 The Journey & Findings

### 🌍 1. The Geography of Risk
> *"Not all roads lead to the same risk."*
> 
We discovered that risk is **highly localized**. By performing rigorous **Chi-Squared** tests, we proved that claim frequency varies significantly by **Province** and **Zip Code**.
*   **Impact**: We recommended a dynamic, location-based multiplier for our rating engine. No more "one price fits all" for vastly different neighborhoods.

### ⚖️ 2. Justice in Algorithms
> *"Fairness isn't a bonus; it's a requirement."*
> 
A persistent myth in insurance is that gender predicts risk. We put this to the test. Using **T-Tests** on thousands of claims, we found **zero statistically significant difference** in risk between men and women ($p > 0.05$).
*   **Impact**: We advised the immediate removal of gender as a rating factor, ensuring our compliance with modern fairness standards without losing a cent of profitability.

### 🤖 3. The Crystal Ball (Predictive Modeling)
> *"Predicting the future, one policy at a time."*
> 
We didn't just analyze the past; we modeled the future.
*   **The Challenge**: Insurance data is messy. 99% of policies don't claim. This "class imbalance" usually breaks models.
*   **The Solution**: We engineered a robust **XGBoost** classification model with `balanced` class weights.
*   **The Result**: A **10x improvement** in identifying high-risk customers before they even file a claim.

---

## 🛠️ Under the Hood

We built this project using a modern Data Science stack designed for reproducibility and scale.

| Component | Tech | Role |
| :--- | :---: | :--- |
| **Language** | ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | The brain of the operation. |
| **Data Ops** | ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white) | Wrangling 37k+ rows with ease. |
| **Logic** | ![SciPy](https://img.shields.io/badge/-SciPy-8CAAE6?logo=scipy&logoColor=white) | Statistical rigor (ANOVA, Chi2). |
| **Intelligence** | ![XGBoost](https://img.shields.io/badge/-XGBoost-FLAMA?logo=xgboost&logoColor=white) | High-performance gradient boosting. |
| **Visualization** | ![Seaborn](https://img.shields.io/badge/-Seaborn-77ACF1?logo=python&logoColor=white) | Turning numbers into pictures. |

---

## 📂 Repository Architecture

We believe in clean code and clear structures.

```mermaid
graph TD;
    A[root] --> B[src]
    B --> B1[data/loader.py]
    B --> B2[features/build_features.py]
    B --> B3[models/train_model.py]
    B --> B5[visualization/gen_dashboard.py]
    A --> C[notebooks]
    C --> C1[01_EDA.ipynb]
    C --> C2[Task_3_Hypothesis_Testing.ipynb]
    C --> C3[Task_4_Modeling.ipynb]
    A --> D[reports]
    D --> D1[final_report/Final_Project_Report.md]
    A --> E[main_pipeline.py]
```

*   `src/`: The engine room. Modular, tested, and ready.
*   `notebooks/`: The laboratory. Where experiments happen.
*   `reports/`: The boardroom. Executive summaries and final deliverables.

---

## ⚡ How to Run This Project

Ready to dive in? Here is how you can reproduce our results in minutes.

### 1. Clone & Equip
```bash
git clone https://github.com/Miftah-Ebrahim/Insurance-claims-Intelligence.git
cd Insurance-claims-Intelligence
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ignite the Pipeline
Run the full end-to-end workflow (Data Loading → Preprocessing → Modeling → Dashboarding):
```bash
python main_pipeline.py
```
*Watch as the models train and the dashboard figures populate in `dashboard/figures/`!*

### 3. Explore the Lab
Open the notebooks to see the step-by-step analysis:
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

---

## 📸 Visual Intelligence

<details>
<summary><b>Click to see our Dashboard Insights</b></summary>

| **Risk Heatmap** | **Geographic Trends** |
|:---:|:---:|
| ![Heatmap](dashboard/figures/correlation_heatmap.png) | ![Geo Trend](dashboard/figures/geographic_trend.png) |
| *Understanding the web of variables* | *Mapping risk across the country* |

</details>

---

## 📝 License

This project is proudly licensed under the **MIT License**. Feel free to fork, learn, and build.

---

<div align="center">

**Built with 💙, ☕, and 🐍 by the AlphaCare Data Science Team**
*Turning Data into Decisions.*

</div>
