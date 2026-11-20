# 🚀 Database Optimizer Machine Learning Models

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-blueviolet.svg?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

This repository hosts three production-grade Machine Learning models built for an **Autonomous Database Optimizer AI Agent**. Modern cloud and vector databases handle unpredictable traffic, complex analytical queries, and massive vector embeddings. 

These models enable the AI Agent to:
1. 🛡️ **Detect system & cloud infrastructure anomalies** before system degradation occurs.
2. ⏱️ **Predict SQL query execution times** to optimize workload scheduling and query queues.
3. ⚡ **Predict vector database index benefits (Throughput/QPS)** to select the optimal indexing and quantization strategy.

All models are packaged using **Scikit-Learn End-to-End Pipelines** (including missing value imputation, standard scaling, and one-hot encoding) and exported as `.pkl` artifacts ready for immediate inference.

---

## 📂 Repository Folder Structure

```
Database-optimzer-models/
│
├── 📁 Anomaly_detection/
│   ├── 📓 anomaly_detection.ipynb        # Complete preprocessing, EDA & balanced classification pipeline
│   ├── 📦 anomaly_detection_model.pkl    # Exported Random Forest pipeline (97% Recall)
│   └── 📊 Cloud_Anomaly_Dataset.csv      # Cloud server metrics dataset (277K+ rows)
│
├── 📁 Database query time/
│   ├── 📓 database_time.ipynb            # Modular EDA (5 visual cells) & regression pipeline
│   ├── 📦 database_time_model.pkl        # Exported Random Forest regressor (R² = 0.968)
│   └── 📊 performance_data_time.csv      # Query performance dataset (4.9K queries)
│
├── 📁 Index_benefit/
│   ├── 📓 index_benifit.ipynb            # Vector DB EDA & user-friendly benefit predictor
│   ├── 📦 index_benefit_model.pkl        # Exported QPS throughput regressor (R² = 0.823)
│   └── 📊 enterprise_vector_db_benchmarks_2026.csv # Vector database benchmark dataset (10K runs)
│
├── 📄 requirements.txt                   # Project dependencies
└── 📖 README.md                          # Project documentation
```

---

## 🧠 Models Breakdown

### 1️⃣ Cloud & Database Anomaly Detection 🛡️

* **Problem Type:** Binary Classification (Imbalanced Data)
* **Goal:** Detect server anomalies, resource exhaustion, and irregular performance spikes in cloud database environments.
* **Dataset:** `Cloud_Anomaly_Dataset.csv` (~277,570 records)
* **Target:** `Anomaly status` (`0`: Normal [94%], `1`: Anomaly [6%])

#### ⚙️ Feature Processing & Architecture:
* **Numerical Features:** `cpu_usage`, `memory_usage`, `network_traffic`, `power_consumption`, `num_executed_instructions`, `execution_time`, `energy_efficiency`
* **Categorical Features:** `task_type`, `task_priority`, `task_status`
* **Pipeline:**
  * Numerical: `SimpleImputer(strategy='mean')` ➔ `StandardScaler()`
  * Categorical: `SimpleImputer(strategy='most_frequent')` ➔ `OneHotEncoder(handle_unknown='ignore')`
  * Merge: `ColumnTransformer`
  * Classifier: `RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced')`

#### 📊 Performance:
* **Class 1 (Anomaly) Recall:** **`97.1%`** (Caught 3,235 out of 3,331 anomalies with only 96 misses!)
* **ROC-AUC Score:** **`0.843`**
* **Significance:** Solved the classic *Accuracy Paradox* by switching from dummy 94% majority guessing to a balanced-weight depth-constrained classifier that aggressively flags genuine anomalies.

---

### 2️⃣ Database Query Execution Time Prediction ⏱️

* **Problem Type:** Regression
* **Goal:** Predict the exact execution time (in seconds) of incoming SQL queries before running them.
* **Dataset:** `performance_data_time.csv` (4,900 benchmarked queries)
* **Target:** `Execution Time (s)`

#### ⚙️ Feature Processing & Architecture:
* **Numerical Features:** `Rows Returned`, `Columns Returned`, `CPU Used (%)`, `RAM Used (MB)`, `Disk Read (MB)`
* **Categorical Features:** `Type` (SELECT DISTINCT, GROUP BY, WHERE, Aggregate, etc.), `Table(s)`
* **Pipeline:**
  * Numerical: `SimpleImputer(strategy='mean')` ➔ `StandardScaler()`
  * Categorical: `SimpleImputer(strategy='most_frequent')` ➔ `OneHotEncoder(handle_unknown='ignore')`
  * Regressor: `RandomForestRegressor(n_estimators=100, random_state=42)`

#### 📊 Performance:
* **$R^2$ Score:** **`0.968` (96.8% variance explained)**
* **Mean Absolute Error (MAE):** **`0.071 seconds` (~71 ms)**
* **Root Mean Squared Error (RMSE):** **`0.138 seconds`**
* **Significance:** Highly accurate prediction within 70 milliseconds on queries running up to 3.8 seconds.

---

### 3️⃣ Vector Database Index Benefit & Throughput Predictor ⚡

* **Problem Type:** Regression
* **Goal:** Predict the query throughput (QPS - Queries Per Second) achieved when applying a specific vector index algorithm and quantization configuration.
* **Dataset:** `enterprise_vector_db_benchmarks_2026.csv` (10,000 enterprise benchmark runs)
* **Target:** `throughput_qps` (Queries Per Second)

#### ⚙️ Practical & User-Friendly Features:
Designed specifically for AI agent integration, this model uses **only practical configuration parameters** that a user or agent can easily provide without guessing internal server telemetry:
* **Data Scale:** `vector_dimensions` (e.g., 384, 768, 1536), `dataset_scale_millions` (e.g., 10M, 50M)
* **Workload:** `top_k` (e.g., 10, 100), `concurrent_clients` (e.g., 4, 16, 32, 64)
* **Index Configuration:** `database_engine` (Milvus, Pinecone, Qdrant, Weaviate, ClickHouse), `indexing_algorithm` (HNSW, IVF-PQ, ScaNN, DiskANN, Flat), `quantization_type` (INT8, INT4, FP32, PQ, Binary)
* **Environment:** `hardware_config` (AWS c6i, AWS r6i, GCP g2, etc.), `use_case` (RAG, Code Search, E-Commerce, etc.)

#### 📊 Performance:
* **$R^2$ Score:** **`0.823` (82.3% variance explained)**
* **Mean Absolute Error (MAE):** **`1874.4 QPS`**
* **Root Mean Squared Error (RMSE):** **`4615.1 QPS`**
* **Significance:** Allows automated database agents to simulate throughput across indexing algorithms (e.g. ScaNN vs HNSW vs IVF-PQ) before spending compute hours building indexes.

---

## 📈 Performance Summary Comparison

| Model | Task | Algorithm | Primary Metric | Secondary Metric | Exported File |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Anomaly Detection** | Binary Classification | Balanced Random Forest | **Recall: `97.1%`** | **ROC-AUC: `0.843`** | `anomaly_detection_model.pkl` |
| **Query Execution Time** | Regression | Random Forest Regressor | **$R^2$: `0.968`** | **MAE: `0.071s`** | `database_time_model.pkl` |
| **Vector Index Benefit** | Regression | Random Forest Regressor | **$R^2$: `0.823`** | **Throughput QPS** | `index_benefit_model.pkl` |

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Lalitmohan6376/Database-optimzer-models.git
cd Database-optimzer-models
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 How to Run Inference

Because all models are packaged with **Scikit-Learn Pipelines**, raw input dictionaries or DataFrames can be fed directly without manual preprocessing:

### Example: Predicting Query Execution Time
```python
import joblib
import pandas as pd

# Load model pipeline
model = joblib.load('Database query time/database_time_model.pkl')

# Provide raw query parameters
sample_query = pd.DataFrame([{
    'Rows Returned': 1500,
    'Columns Returned': 8,
    'CPU Used (%)': 45.2,
    'RAM Used (MB)': 512.0,
    'Disk Read (MB)': 128.5,
    'Type': 'SELECT DISTINCT',
    'Table(s)': 'orders'
}])

predicted_time = model.predict(sample_query)
print(f"⏱️ Estimated Execution Time: {predicted_time[0]:.4f} seconds")
```

### Example: Predicting Vector DB Index Benefit
```python
import joblib
import pandas as pd

# Load model pipeline
model = joblib.load('Index_benefit/index_benefit_model.pkl')

# Provide vector database configuration
sample_workload = pd.DataFrame([{
    'vector_dimensions': 768,
    'dataset_scale_millions': 10.0,
    'top_k': 10,
    'concurrent_clients': 16,
    'database_engine': 'Qdrant',
    'indexing_algorithm': 'HNSW',
    'quantization_type': 'INT8 (SQ8)',
    'hardware_config': 'AWS c6i.8xlarge (32 vCPU)',
    'use_case': 'RAG Semantic Retrieval'
}])

predicted_qps = model.predict(sample_workload)
print(f"⚡ Estimated Throughput: {predicted_qps[0]:.1f} QPS")
```

---

## 🎨 Exploratory Data Analysis (EDA) Highlights

Each notebook includes clean, modular exploratory analysis using **only Matplotlib**:
* 📊 **Anomaly Detection:** Distribution of server loads, memory, and energy efficiency vs anomaly classes.
* 📊 **Query Time:** Query type breakdown, CPU vs execution time scatter plots, and Disk I/O impact.
* 📊 **Vector Index Benefit:** Algorithm throughput comparisons, quantization efficiency gains, and client concurrency scaling curves.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or pull request.

---

## 📄 License

This project is licensed under the MIT License.
