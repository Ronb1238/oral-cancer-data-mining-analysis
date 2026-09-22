# Oral Cancer Prediction Analysis & Synthetic Dataset Validation

An end-to-end Data Mining project analyzing an oral cancer risk factor dataset (~85,000 records) sourced from Kaggle[cite: 1, 5, 6]. 

This repository implements a complete Knowledge Discovery in Databases (KDD) pipeline—covering data preprocessing, leakage prevention, supervised classification, association rule mining, unsupervised clustering, and artificial neural networks[cite: 1, 5, 6]. Through empirical testing across multiple algorithm families, this project rigorously evaluates model performance and proves the synthetic, uncorrelated nature of the dataset[cite: 1, 5].

---

## 📌 Key Findings & Methodology

Rather than applying machine learning blindly, this project focuses on **critical data evaluation** and **leakage prevention**:

1. **Preprocessing & Leakage Control:**
   * Removed post-diagnostic features (e.g., tumor size, treatment type, early diagnosis) and late-stage clinical symptoms (e.g., unexplained bleeding, mouth patches) to prevent target leakage[cite: 1, 3, 4, 5].
   * Encoded categorical attributes and normalized features using `MinMaxScaler`.

2. **Supervised Classification (Decision Trees & Random Forest):**
   * Implemented CART Decision Trees and Random Forests to classify oral cancer risk[cite: 5].
   * Models converged at a ~50.2% accuracy rate, offering no predictive lift over random chance[cite: 5].

3. **Association Rule Mining (Apriori):**
   * Extracted 389 rules with strong Support (≥40%) and Confidence (≥60%)[cite: 1, 3, 5].
   * Evaluated rule quality using **Lift**, revealing Lift ≈ 1.00 across all rules[cite: 1, 3, 5]. This proved absolute statistical independence between clinical risk factors (e.g., smoking, HPV, family history) and diagnosis[cite: 1, 3, 5].

4. **Unsupervised Clustering (K-Means):**
   * Applied K-Means clustering (K=2) to evaluate natural grouping in the feature space[cite: 1, 4, 5].
   * Achieved a low **Silhouette Score of 0.1111**[cite: 1, 4, 5]. Cross-tabulation against true labels showed an exact 50/50 split across both clusters, confirming no distinct spatial structure[cite: 1, 4, 5].

5. **Artificial Neural Network (MLP Classifier):**
   * Designed a Multi-Layer Perceptron (16-8 hidden layer structure) trained over 50 epochs using Log Loss evaluation[cite: 2, 5].
   * Tracked epoch-by-epoch loss using `warm_start=True`[cite: 2]. Training loss consistently decreased while test loss increased—a classic demonstration of **overfitting on white noise**[cite: 1, 5].

6. **Core Conclusion:**
   * Machine learning algorithms cannot extract signal from uncorrelated noise[cite: 5]. The failure of all model families to outperform random guessing proves that the dataset is synthetically generated with uniform feature distribution[cite: 1, 5].

## 🚀 Getting Started

### Prerequisites

* **Python 3.9+** installed.

### Installation & Execution

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/oral-cancer-data-mining-analysis.git
cd oral-cancer-data-mining-analysis
```

#### 2. Create and Activate a Virtual Environment

**Linux / macOS:**

```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Download the Dataset

Download `oral_cancer_prediction_dataset.csv` from Kaggle and place it in the root directory.

#### 5. Run an Analysis Module

**Association Rules:**

```bash
python src/02_association_rules.py
```

**K-Means Clustering:**

```bash
python src/03_kmeans_clustering.py
```

**Neural Network:**

```bash
python src/04_neural_network.py
```

---

## 🛠️ Technologies Used

| Category                           | Technologies          |
| ---------------------------------- | --------------------- |
| **Language**                       | Python                |
| **Data Processing**                | Pandas, NumPy         |
| **Machine Learning & Data Mining** | Scikit-Learn, Mlxtend |
| **Visualization**                  | Matplotlib, Seaborn   |

