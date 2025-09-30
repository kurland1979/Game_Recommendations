# 🎮 Steam Games Recommendation System

## 📌 Project Goal

The goal of this project was to build a reliable recommendation system for Steam, using multiple algorithms — from simple baselines to advanced models — and compare their performance.

On a personal note, the purpose was also to **work with large-scale data and real-world challenges**, strengthening my skills in Machine Learning and building industry-like pipelines.

# 🎮 Steam Game Recommendation System

A comprehensive recommendation system for Steam games, comparing multiple algorithms including collaborative filtering, matrix factorization, and gradient boosting.

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/kurland1979/Game_Recommendations.git
cd Game_Recommendations

## 📂 Dataset

Source: [Kaggle – Steam Game Recommendations](https://www.kaggle.com/datasets)
The dataset contains **~10 million interactions** (reviews + recommendations + users + games metadata).
Due to hardware limitations, the final run was executed on **5 million rows**.
The full dataset is uploaded to Google Drive for those who wish to run it on the complete 10M rows.

### 📥 Download Data Files
The large dataset files are stored on Google Drive. Download them here:

- [**users.csv** (185 MB)](https://drive.google.com/file/d/1loW5XkftMqZNbeoDFlAs9uQODkzYqLui/view?usp=sharing) - User statistics
- [**recommendations.csv** (1.88 GB)](https://drive.google.com/file/d/1AQGA9fiK8XvJaFfyBO-jrynNZ2NpRsoJ/view?usp=sharing) - User-game interactions

**After downloading:** Place both files in the `data/` directory of the project.

### Main files:

* **games.csv** – Game metadata.
* **recommendations.csv** – User-game interactions (recommended or not).
* **users.csv** – User statistics (products owned, reviews written).

## 🔍 EDA – Key Findings

* **Products**: Users with fewer owned games are more likely to recommend, while heavy owners tend to recommend less.
* **Reviews**: Almost no strong correlation with recommendation rate.
* **Hours Played**: Only the lowest bin (0–5 hours) shows noticeably lower recommendation rates.

## 🧪 Baselines

1. **Global Mean** – served as a very naive starting point.
2. **Threshold on Products** – slight improvement (AUC ≈ 0.55).
3. **Item-Item Collaborative Filtering** – performed poorly in this dataset.

## 🤖 Advanced Models

* **LightFM** (with and without features): Balanced precision/recall (~0.78–0.84) but AUC remained low (~0.49–0.54).
* **LightGBM**: The clear winner, significantly outperforming all others with **Val AUC = 0.858, Test AUC = 0.836**, and near-perfect Precision/Recall.

## 📊 Final Results (5M rows run)

| Model                   | Val AUC   | Test AUC  | Precision@K | Runtime (s) |
| ----------------------- | --------- | --------- | ----------- | ----------- |
| Baseline (Mean)         | 0.564     | 0.558     | 0.825       | 13.69       |
| Baseline (Threshold)    | 0.586     | 0.563     | 0.925       | 14.48       |
| Baseline (Item-Item CF) | 0.547     | 0.430     | 0.008       | 0.09        |
| LightFM                 | 0.540     | 0.478     | 0.781       | 57.72       |
| LightFM + Features      | 0.546     | 0.529     | 0.781       | 350.03      |
| **LightGBM**            | **0.858** | **0.836** | **1.0**     | 118.58      |

📌 The comparison chart of AUC scores is saved as `models_auc_comparison.png`.
📌 The full summary table is saved as `summary_results.csv`.

## 🛠 Project Structure

* `get_files.py`, `check_files.py` – File loading and validation.
* `merge_files.py` – Merge all datasets.
* `eda_analysis.py` – Exploratory analysis and visualizations.
* `baseline_mean.py`, `baseline_threshold.py`, `baseline_item_item_cf.py` – Baseline models.
* `algo_lightfm.py`, `algo_lightfm_features.py`, `algo_lightgbm.py` – Advanced models.
* `feature_analysis.py` – Feature exploration.
* `threshold_analysis.py` – Threshold-based evaluation.
* `recommend.py` – Mini recommendation demo with threshold-based suggestions.
* `data_utils.py` – Utility file for train/val/test splitting (random or date-based).
* `config.json`, `config.py` – Centralized configuration management.
* `main.py`, `run.py` – Main execution files.

## Dependencies

All experiments in this project were implemented in Python using common data science and machine learning libraries.
The full list of required packages, along with their exact versions, is provided in the requirements.txt file.
To replicate the environment, simply run:
* pip install -r requirements.txt

## 🚀 How to Run

1. Update file paths in `config.json`.
2. Choose split type (`random` or `date`) in `data_utils.py`.
3. Run:

   ```bash
   python3 main.py
   ```

   Use `verbose=True` to see detailed outputs.

## 💡 Personal Note

I chose this dataset because I wanted a **real challenge**. The scale (10M rows) and technical issues (including switching to Linux/WSL) made it far from easy, but every obstacle taught me something valuable.

This project is more than just about numbers — it’s proof that with persistence, every technical challenge can be overcome.

---

✍️ Author: Marina Kurland
