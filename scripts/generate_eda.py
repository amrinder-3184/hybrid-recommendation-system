import nbformat as nbf
import os

def generate_eda_notebook():
    nb = nbf.v4.new_notebook()
    
    # Title
    nb.cells.append(nbf.v4.new_markdown_cell("# Exploratory Data Analysis (EDA)\n\nThis notebook analyzes the Amazon Reviews 2023 Video Games dataset to understand sparsity, rating distributions, and feature properties."))
    
    # Imports
    nb.cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set aesthetic params
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)"""))
    
    # Load Data
    nb.cells.append(nbf.v4.new_markdown_cell("## 1. Load Data\nLoad the processed reviews and metadata."))
    nb.cells.append(nbf.v4.new_code_cell("""reviews = pd.read_parquet('../data/processed/reviews_processed.parquet')
meta = pd.read_parquet('../data/processed/meta_processed.parquet')
print(f"Reviews shape: {reviews.shape}")
print(f"Meta shape: {meta.shape}")"""))
    
    # Rating Distribution
    nb.cells.append(nbf.v4.new_markdown_cell("## 2. Rating Distribution\nCheck how the ratings are distributed."))
    nb.cells.append(nbf.v4.new_code_cell("""sns.countplot(x='rating', data=reviews, palette='viridis')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()"""))
    
    # Sparsity Analysis
    nb.cells.append(nbf.v4.new_markdown_cell("## 3. Sparsity Analysis\nCheck dataset sparsity using artifacts generated during feature engineering."))
    nb.cells.append(nbf.v4.new_code_cell("""with open('../artifacts/dataset_stats.json', 'r') as f:
    stats = json.load(f)
    
print(f"Total Users: {stats['num_users']}")
print(f"Total Items: {stats['num_items']}")
print(f"Total Interactions: {stats['num_interactions']}")
print(f"Dataset Sparsity: {stats['sparsity'] * 100:.4f}%")"""))

    # User & Item interaction distribution
    nb.cells.append(nbf.v4.new_markdown_cell("## 4. Interaction Distributions\nLet's plot how many reviews users write and how many reviews items receive."))
    nb.cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(15, 5))

user_counts = reviews['user_id'].value_counts()
sns.histplot(user_counts, bins=50, ax=axes[0], log_scale=(False, True))
axes[0].set_title('Number of Reviews per User (Log Scale)')
axes[0].set_xlabel('Number of Reviews')
axes[0].set_ylabel('Count (Log)')

item_counts = reviews['parent_asin'].value_counts()
sns.histplot(item_counts, bins=50, ax=axes[1], log_scale=(False, True))
axes[1].set_title('Number of Reviews per Item (Log Scale)')
axes[1].set_xlabel('Number of Reviews')
axes[1].set_ylabel('Count (Log)')

plt.tight_layout()
plt.show()"""))

    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/01_eda.ipynb', 'w') as f:
        nbf.write(nb, f)
        
    print("EDA notebook generated at notebooks/01_eda.ipynb")

if __name__ == "__main__":
    generate_eda_notebook()
