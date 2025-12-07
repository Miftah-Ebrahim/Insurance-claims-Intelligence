
<div align="center">

# 🛡️ Insurance Claims Intelligence
### *Next-Gen Algorithmic Risk Assessment & Pricing Optimization*

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Pipeline-Production--Ready-success?style=for-the-badge&logo=github-actions)](https://github.com/Miftah-Ebrahim/Insurance-claims-Intelligence)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)
[![DVC](https://img.shields.io/badge/Data-Versioned%20(DVC)-orange?style=for-the-badge&logo=dvc)](https://dvc.org/)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black?style=for-the-badge&logo=python)](https://github.com/psf/black)

[**Explore the Analysis**](notebooks/01_EDA.ipynb) · [**View Test Results**](notebooks/Task_3_Hypothesis_Testing.ipynb) · [**Report Bug**](https://github.com/Miftah-Ebrahim/Insurance-claims-Intelligence/issues)

</div>

---

##  Executive Summary

**AlphaCare Insurance Solutions** tasked the data science team with unlocking hidden value in their historical claims data. This project delivers a comprehensive analytical framework designed to:
1.  **Optimize Premium Pricing**: By correlating `SumInsured` and `TotalPremium`.
2.  **Mitigate High-Risk Segments**: Through granular **Zip Code** and **Province** based risk profiling.
3.  **Ensure Fairness**: By statistically proving that **Gender** is *not* a valid discriminator for risk.

This repository serves as the **central intelligence hub** for Actuarial and Marketing teams.

---

## 📊 Key Intelligence Findings

### 🌍 1. Geo-Spatial Risk Asymmetry
> *Not all provinces are created equal.*
Our Chi-Squared analysis ($p < 0.05$) revealed statistically significant differences in claim frequencies across provinces.
*   **Action**: Implement dynamic pricing multipliers based on provincial risk scores.

### 📍 2. The "Zip Code" Effect
> *Location is the strongest predictor of severity.*
ANOVA testing confirmed that `PostalCode` variances in `TotalClaims` are non-random.
*   **Action**: Move beyond broad city-based rating to hyper-local zip-code based underwriting.

### ⚖️ 3. Fairness in AI
> *Algorithmically verified equality.*
A rigorous T-Test comparing Men vs. Women yielded a $p$-value $> 0.05$, failing to reject the null hypothesis.
*   **Action**: Remove Gender as a primary rating factor to ensure compliance and fairness without sacrificing predictive power.

### 🤖 4. Predictive Modeling
> *From Hindsight to Foresight.*
We implemented a dual-model risk engine:
*   **Severity Model (Regression)**: Predicts potential claim cost (`TotalClaims`) with high precision.
*   **Probability Model (Classification)**: Identifies high-risk policies (`IsClaim`) using XGBoost.
*   **Interpretability**: SHAP analysis confirms `VehicleAge` and `SumInsured` as top risk drivers.

---

## 🛠️ Technology Stack

| Component | Tech | Description |
| :--- | :---: | :--- |
| **Language** | ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | Core logic and analysis. |
| **Data Ops** | ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white) | robust ETL and manipulation. |
| **Stats** | ![SciPy](https://img.shields.io/badge/-SciPy-8CAAE6?logo=scipy&logoColor=white) | Chi2, ANOVA, T-Tests. |
| **Viz** | ![Seaborn](https://img.shields.io/badge/-Seaborn-77ACF1?logo=python&logoColor=white) | Publication-quality plotting. |
| **Version Control** | ![DVC](https://img.shields.io/badge/-DVC-9cf?logo=dvc&logoColor=white) | Large dataset versioning. |

---

## 📂 Repository Architecture

We follow a production-grade directory structure for scalability.

```mermaid
graph TD;
    A[root] --> B[src];
    B --> B1[data/loader.py];
    B --> B2[utils/plots.py];
    B --> B3[stats/hypothesis.py];
    A --> C[notebooks];
    C --> C1[01_EDA.ipynb];
    C --> C2[Task_3_Hypothesis_Testing.ipynb];
    A --> D[data];
    D --> D1[raw/MachineLearningRating.txt];
```

*   `src/`: Modular Python scripts for Data Loading (`loader.py`), Plotting (`plots.py`), and Statistics (`hypothesis.py`).
*   `notebooks/`: Interactive environments for **Deep EDA** and **Hypothesis Testing**.
*   `data/`: Managed via **DVC** (Data Version Control).

---

## ⚡ Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/Miftah-Ebrahim/Insurance-claims-Intelligence.git
cd Insurance-claims-Intelligence
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Analysis
Launch the Jupyter environment to explore the insights:
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### 3. Verify Hypotheses
Run the statistical test suite:
```bash
jupyter notebook notebooks/Task_3_Hypothesis_Testing.ipynb
```

---

## 📈 Visual Gallery

<details>
<summary><b>Click to view Analysis Plots</b></summary>

| **Correlation Heatmap** | **Risk Distribution** |
|:---:|:---:|
| ![Heatmap](https://via.placeholder.com/400x200?text=Correlation+Analysis) | ![KDE](https://via.placeholder.com/400x200?text=Risk+Distribution) |
| *Identifying variable relationships* | *Understanding outlier impact* |

</details>

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with 💙 by the AlphaCare Data Science Team**
*Turning Data into Decisions*

</div>
