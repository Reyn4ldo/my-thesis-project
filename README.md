# AMR Analysis Pipeline - Complete Implementation

This repository provides a comprehensive analysis pipeline for Antimicrobial Resistance (AMR) data, implementing data preparation (Phase 1), unsupervised pattern recognition (Phase 2), supervised machine learning (Phase 3), and model deployment (Phase 4).

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Phase 1: Data Preparation](#phase-1-data-preparation)
- [Phase 2: Unsupervised Pattern Recognition](#phase-2-unsupervised-pattern-recognition)
- [Phase 3: Supervised Pattern Recognition](#phase-3-supervised-pattern-recognition)
- [Phase 4: Model Deployment](#phase-4-model-deployment)
- [Quick Start](#quick-start)
- [Testing](#testing)

## Overview

The AMR analysis pipeline consists of four main phases:

**Phase 1: Data Preparation**
1. Data ingestion and inspection
2. Cleaning S/I/R interpretations
3. Binary and ordinal resistance encoding
4. Categorical feature encoding
5. Missing value handling
6. Feature scaling

**Phase 2: Unsupervised Pattern Recognition**
1. Clustering (k-means, hierarchical, DBSCAN)
2. Dimensionality reduction (PCA, t-SNE, UMAP)
3. Association rule mining (Apriori, FP-Growth)

**Phase 3: Supervised Pattern Recognition**
1. Binary classification: Predict high MAR/MDR
2. Multiclass classification: Predict bacterial species
3. Multiclass classification: Predict region/source

**Phase 4: Model Deployment**
1. Model saving with comprehensive metadata
2. Command-line deployment tools
3. Batch prediction from CSV files
4. Single isolate prediction
5. Web application integration (documentation)

## Installation

### Requirements

Install all dependencies using the requirements file:

```bash
pip install -r requirements.txt
```

Or install core packages manually:

```bash
# Core data science libraries
pip install pandas numpy scikit-learn matplotlib seaborn mlxtend umap-learn

# Web frameworks for deployment
pip install fastapi uvicorn streamlit plotly python-multipart
```

**Note:** `umap-learn` is optional but recommended for UMAP analysis.

---

## Quick Start

This section provides a quick guide to get started with the AMR analysis pipeline. For detailed documentation of each phase, see the phase-specific sections below.

### Complete End-to-End Workflow

The simplest way to go from raw data to a deployed model:

```python
from data_preparation import AMRDataPreparation
from supervised_analysis import SupervisedAMRAnalysis
from model_deployment import ModelDeployment

# Phase 1: Prepare data
prep = AMRDataPreparation('rawdata.csv')
df = prep.prepare_data(
    include_binary=True,
    include_ordinal=False,
    include_onehot=True,
    scale=False,
    drop_original_int=True
)

# Get features
groups = prep.get_feature_groups()
feature_cols = groups['binary_resistance']

# Phase 3: Train model
analyzer = SupervisedAMRAnalysis(df)
results = analyzer.task1_high_mar_prediction(
    feature_cols=feature_cols,
    threshold=0.3,
    include_tuning=True,
    save_model_path='high_MAR_model.pkl'
)

print(f"Best Model: {results['best_model']}")
print(f"Test F1: {results['test_metrics']['f1']:.4f}")
print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")

# Phase 4: Deploy model
deployment = ModelDeployment('high_MAR_model.pkl')

# Make predictions on new data
predictions = deployment.predict_from_csv(
    input_csv='new_isolates.csv',
    output_csv='predictions.csv',
    include_proba=True
)

print(f"Predictions saved to predictions.csv")
```

### Quick Start by Use Case

#### 1. Data Preparation Only

```python
from data_preparation import quick_prepare

# One-line preparation with defaults
df = quick_prepare('rawdata.csv', output_path='prepared_data.csv')
```

#### 2. Clustering Analysis

```python
from data_preparation import AMRDataPreparation
from unsupervised_analysis import UnsupervisedAMRAnalysis

# Prepare with ordinal encoding
prep = AMRDataPreparation('rawdata.csv')
df = prep.prepare_data(
    include_binary=False,
    include_ordinal=True,
    include_onehot=False,
    scale=True
)

# Cluster analysis
groups = prep.get_feature_groups()
feature_cols = groups['ordinal_resistance'] + groups['amr_indices']
analyzer = UnsupervisedAMRAnalysis(df, feature_cols)
results = analyzer.kmeans_clustering(k_range=(2, 10))

# Visualize
fig = analyzer.plot_kmeans_evaluation()
```

#### 3. Predict Bacterial Species

```python
from data_preparation import AMRDataPreparation
from supervised_analysis import SupervisedAMRAnalysis

# Prepare data
prep = AMRDataPreparation('rawdata.csv')
df = prep.prepare_data(include_binary=True, include_onehot=False)

# Train species classifier
groups = prep.get_feature_groups()
analyzer = SupervisedAMRAnalysis(df)
results = analyzer.task2_species_classification(
    feature_cols=groups['binary_resistance'],
    include_tuning=True
)

print(f"Species prediction accuracy: {results['test_metrics']['accuracy']:.4f}")
```

#### 4. Deploy Saved Model

```python
from model_deployment import ModelDeployment

# Load and use model
deployment = ModelDeployment('high_MAR_model.pkl')

# Get required features
required_features = deployment.get_required_features()

# Single prediction (provide all required binary resistance features)
features = {
    'ampicillin_binary': 1,
    'cefotaxime_binary': 1,
    'tetracycline_binary': 0,
    # ... include all features from required_features list
}
result = deployment.predict_single(features, return_proba=True)
print(f"Prediction: {result['prediction']}")
```

### Running Complete Examples

The repository includes comprehensive example scripts:

```bash
# Data preparation examples
python examples.py

# Unsupervised analysis examples (10 examples)
python examples_unsupervised.py

# Supervised analysis examples (7 examples)
python examples_supervised.py

# Deployment examples (4 examples)
python examples_deployment.py

# Complete end-to-end workflow
python complete_workflow.py
```

### Next Steps

- **For Data Preparation**: See [Phase 1: Data Preparation](#phase-1-data-preparation)
- **For Clustering/Pattern Discovery**: See [Phase 2: Unsupervised Pattern Recognition](#phase-2-unsupervised-pattern-recognition)
- **For Classification Tasks**: See [Phase 3: Supervised Pattern Recognition](#phase-3-supervised-pattern-recognition)
- **For Model Deployment**: See [Phase 4: Model Deployment](#phase-4-model-deployment)

---

# Phase 1: Data Preparation

Phase 1 implements comprehensive data preparation for AMR analysis.

## Quick Start - Phase 1

### Basic Usage

```python
from data_preparation import quick_prepare

# Quick preparation with default settings
df_prepared = quick_prepare('rawdata.csv', output_path='prepared_data.csv')
```

### Advanced Usage

```python
from data_preparation import AMRDataPreparation

# Initialize
prep = AMRDataPreparation('rawdata.csv')

# Load and inspect
prep.load_data()
inspection_report = prep.inspect_data()

# Full pipeline with custom options
df_prepared = prep.prepare_data(
    include_binary=True,      # Binary resistance encoding (R=1, S/I=0)
    include_ordinal=True,     # Ordinal encoding (S=0, I=1, R=2)
    include_onehot=True,      # One-hot encode categorical variables
    missing_strategy='conservative',  # Treat missing as susceptible
    scale=True,               # Standardize features
    drop_original_int=True    # Remove original _int columns
)

# Get organized feature groups
feature_groups = prep.get_feature_groups()

# Export
prep.export_prepared_data('output.csv')
```

## Features

### 1.1 Data Ingestion

```python
prep = AMRDataPreparation('rawdata.csv')
df = prep.load_data()

# Automatic identification of:
# - Antibiotic interpretation columns (*_int)
# - Antibiotic MIC columns (*_mic)
# - Context columns (species, source, region, etc.)
# - AMR indices (mar_index, scored_resistance, num_antibiotics_tested)
```

### 1.2 Data Inspection

```python
report = prep.inspect_data()

# Returns dictionary with:
# - Shape and column information
# - Missing value statistics
# - Context variable distributions
# - Antibiotic interpretation distributions
# - AMR indices statistics
```

### 1.3 Data Cleaning and Encoding

#### 1.3.1 Clean S/I/R Interpretations

Standardizes resistance interpretations:
- Converts to lowercase
- Maps variants: `*r` → `r`, `*i` → `i`, `*s` → `s`
- Treats blank/unknown as missing

```python
df_clean = prep.clean_sir_interpretations(df)
```

#### 1.3.2 Binary Resistance Encoding

Creates binary resistance features for supervised learning:
- R = 1 (resistant)
- S = 0 (susceptible)
- I = 0 (intermediate, treated as susceptible)
- Missing = 0 (optional, conservative approach)

```python
df_binary = prep.encode_binary_resistance(df_clean, missing_as_susceptible=True)
```

**Result**: Creates columns like `ampicillin_binary`, `cefotaxime_binary`, etc.

#### 1.3.3 Ordinal S/I/R Encoding

Creates ordinal features for unsupervised analysis:
- S = 0 (susceptible)
- I = 1 (intermediate)
- R = 2 (resistant)

```python
df_ordinal = prep.encode_ordinal_sir(df_clean)
```

**Result**: Creates columns like `ampicillin_ordinal`, `cefotaxime_ordinal`, etc.

#### 1.3.4 Categorical Encoding

One-hot encodes context variables:
- `bacterial_species`
- `sample_source`
- `administrative_region`
- `national_site`
- `local_site`
- `esbl`

```python
df_encoded = prep.encode_categorical_onehot(df, drop_first=False)
```

**Result**: Creates dummy variables like `bacterial_species_escherichia_coli`, `sample_source_drinking_water`, etc.

#### 1.3.5 Missing Value Handling

Strategies:
- **Conservative**: Treat missing antibiotic interpretations as susceptible
- **Median imputation**: For numeric AMR indices
- **Drop threshold**: Remove rows with >80% missing values

```python
df_handled = prep.handle_missing_values(df, strategy='conservative', drop_threshold=0.8)
```

#### 1.3.6 Feature Scaling

Standardizes numeric features using z-score normalization:
- Formula: `(X - mean) / std`
- Applied to: AMR indices, MIC values, ordinal columns
- Binary columns excluded by default

```python
df_scaled = prep.scale_features(df, fit=True, exclude_binary=True)
```

## Feature Groups

After preparation, features are organized into groups:

```python
feature_groups = prep.get_feature_groups()

# Returns:
{
    'binary_resistance': [...],  # Binary resistance columns
    'ordinal_resistance': [...],  # Ordinal S/I/R columns
    'amr_indices': [...],  # mar_index, scored_resistance, num_antibiotics_tested
    'context_encoded': [...],  # One-hot encoded categorical variables
    'metadata': [...]  # isolate_code, replicate, colony
}
```

## Data Flow

```
Raw Data (rawdata.csv)
    ↓
Load & Identify Columns
    ↓
Clean S/I/R Interpretations
    ↓
Encode Resistance (Binary & Ordinal)
    ↓
Handle Missing Values
    ↓
One-Hot Encode Categorical Variables
    ↓
Scale Numeric Features
    ↓
Prepared Data (ready for analysis)
```

## Output Structure

After full preparation with default settings:

- **Input**: 583 rows × 58 columns
- **Output**: 583 rows × 91-114 columns (depending on options)

### Column Groups in Output:
1. **Metadata**: isolate_code, replicate, colony
2. **Binary Resistance** (23 columns): *_binary for each antibiotic
3. **Ordinal Resistance** (23 columns, optional): *_ordinal for each antibiotic
4. **AMR Indices** (3 columns): mar_index, scored_resistance, num_antibiotics_tested
5. **Context Encoded** (~39 columns): One-hot encoded categorical variables
6. **MIC Values** (23 columns): Original MIC measurements

## Testing

Run the comprehensive test suite:

```bash
python test_data_preparation.py
```

**Test Coverage:**
- Data loading and inspection
- S/I/R cleaning
- Binary encoding
- Ordinal encoding
- Categorical encoding
- Missing value handling
- Feature scaling
- Complete pipeline
- Feature groups
- Export functionality
- Data quality checks

## Examples

### Example 1: Basic Preparation for Supervised Learning

```python
from data_preparation import AMRDataPreparation

prep = AMRDataPreparation('rawdata.csv')
df = prep.prepare_data(
    include_binary=True,
    include_ordinal=False,
    include_onehot=True,
    scale=True
)

# Get binary resistance features for classification
feature_groups = prep.get_feature_groups()
X_features = feature_groups['binary_resistance'] + feature_groups['context_encoded']
y_target = feature_groups['amr_indices']
```

### Example 2: Preparation for Unsupervised Analysis

```python
prep = AMRDataPreparation('rawdata.csv')
df = prep.prepare_data(
    include_binary=False,
    include_ordinal=True,
    include_onehot=True,
    scale=True
)

# Use ordinal features for clustering
feature_groups = prep.get_feature_groups()
X_clustering = feature_groups['ordinal_resistance']
```

### Example 3: Custom Feature Selection

```python
prep = AMRDataPreparation('rawdata.csv')

# Step-by-step preparation
df = prep.load_data()
df = prep.clean_sir_interpretations(df)
df = prep.encode_binary_resistance(df)

# Select specific antibiotics
beta_lactams = ['ampicillin_binary', 'cefotaxime_binary', 'ceftiofur_binary']
fluoroquinolones = ['nalidixic_acid_binary', 'enrofloxacin_binary']

X = df[beta_lactams + fluoroquinolones]
```

## Key Variables Reference

### Antibiotic Interpretation Columns (*_int)
- ampicillin_int
- amoxicillin/clavulanic_acid_int
- cefotaxime_int
- ceftiofur_int
- nalidixic_acid_int
- enrofloxacin_int
- tetracycline_int
- chloramphenicol_int
- trimethoprim/sulfamethazole_int
- (and 14 more...)

### Context Columns
- bacterial_species (13 unique species)
- sample_source (9 sources: drinking_water, river_water, fish_tilapia, etc.)
- administrative_region (3 regions)
- national_site (3 sites)
- local_site (9 sites)
- esbl (pos/neg/missing)

### AMR Indices
- mar_index: Multiple Antibiotic Resistance index
- scored_resistance: Total resistance score
- num_antibiotics_tested: Number of antibiotics tested

## Best Practices

1. **For Classification Tasks**: Use binary encoding with one-hot categorical encoding
2. **For Clustering**: Use ordinal encoding with standardized features
3. **Missing Values**: Conservative approach (treat as susceptible) is recommended
4. **Scaling**: Always scale for distance-based algorithms (k-means, SVM, kNN)
5. **Data Leakage**: Drop original _int columns after encoding to prevent leakage

## Troubleshooting

### Issue: FutureWarning about downcasting
This is a pandas warning and does not affect functionality. The code will continue to work correctly.

### Issue: RuntimeWarning in scaling
This occurs when scaling columns with all NaN values. These columns are handled appropriately.

### Issue: Too many missing values
Adjust `drop_threshold` parameter in `handle_missing_values()` to be more or less aggressive.

---

# Phase 2: Unsupervised Pattern Recognition

Phase 2 implements unsupervised learning methods to discover patterns in AMR data.

## Quick Start - Phase 2

### Clustering Analysis

```python
from data_preparation import AMRDataPreparation
from unsupervised_analysis import UnsupervisedAMRAnalysis

# Prepare data for clustering
prep = AMRDataPreparation('rawdata.csv')
df = prep.prepare_data(
    include_binary=False,
    include_ordinal=True,  # Use ordinal for clustering
    include_onehot=False,
    scale=True
)

# Get features
groups = prep.get_feature_groups()
feature_cols = groups['ordinal_resistance'] + groups['amr_indices']

# Initialize analyzer
analyzer = UnsupervisedAMRAnalysis(df, feature_cols)

# Run k-means clustering
results = analyzer.kmeans_clustering(k_range=(2, 10))

# Plot evaluation
fig = analyzer.plot_kmeans_evaluation()
```

### Dimensionality Reduction

```python
# Run PCA
pca_results = analyzer.pca_analysis(n_components=3)

# Run t-SNE
tsne_results = analyzer.tsne_analysis(n_components=2)

# Visualize
fig = analyzer.plot_dimred_scatter(method='pca', color_by='bacterial_species')
```

### Association Rule Mining

```python
# Prepare transactions
transactions = analyzer.prepare_transactions(
    resistance_threshold=0.5,
    include_metadata=True,
    mar_threshold=0.3
)

# Mine rules with Apriori
rules = analyzer.apriori_mining(
    transactions,
    min_support=0.05,
    min_confidence=0.6,
    min_lift=1.0
)

# Interpret top rules
formatted_rules = analyzer.interpret_rules(rules, top_n=10)
```

## 2.1 Clustering Methods

### 2.1.1 K-means Clustering

**Purpose:** Partition isolates into k clusters based on AMR features.

**Methods:**
- Runs for k = 2–10
- Uses elbow method and silhouette scores to determine optimal k
- Provides cluster labels for each isolate

**Example:**

```python
analyzer = UnsupervisedAMRAnalysis(df, feature_cols)

# Run k-means
results = analyzer.kmeans_clustering(k_range=(2, 10))

# Plot evaluation metrics
fig = analyzer.plot_kmeans_evaluation()

# Analyze clusters for k=5
cluster_labels = results['labels'][5]
analysis = analyzer.analyze_clusters(cluster_labels, method_name='K-means')
```

**Output:**
- `inertias`: Within-cluster sum of squares for each k
- `silhouette_scores`: Silhouette coefficient for each k
- `labels`: Cluster assignments for each k value
- Evaluation plots showing elbow curve and silhouette scores

### 2.1.2 Hierarchical Clustering

**Purpose:** Build a hierarchy of clusters using Ward linkage.

**Methods:**
- Computes distance matrix (Euclidean on standardized features)
- Uses Ward linkage to minimize variance
- Cuts dendrogram at specified height to extract clusters

**Example:**

```python
# Run hierarchical clustering
results = analyzer.hierarchical_clustering(n_clusters=5, linkage_method='ward')

# Plot dendrogram
fig = analyzer.plot_dendrogram(max_display_levels=10)

# Analyze clusters
analysis = analyzer.analyze_clusters(results['labels'], method_name='Hierarchical')
```

**Output:**
- `linkage_matrix`: Hierarchical clustering linkage matrix
- `labels`: Cluster assignments
- `silhouette_score`: Clustering quality metric
- Dendrogram visualization

### 2.1.3 DBSCAN

**Purpose:** Density-based clustering that identifies clusters and outliers.

**Methods:**
- Uses standardized features
- Requires two parameters:
  - `eps`: Maximum distance between neighbors
  - `min_samples`: Minimum samples in a neighborhood (e.g., 5–10)
- Uses k-distance plot for parameter selection
- Labels outliers as -1

**Example:**

```python
# Plot k-distance to choose eps
fig = analyzer.plot_k_distance(k=5)

# Run DBSCAN with selected parameters
results = analyzer.dbscan_clustering(eps=2.0, min_samples=5)

print(f"Clusters: {results['n_clusters']}")
print(f"Noise points: {results['n_noise']}")

# Analyze clusters
analysis = analyzer.analyze_clusters(results['labels'], method_name='DBSCAN')
```

**Output:**
- `n_clusters`: Number of clusters found
- `n_noise`: Number of outlier points
- `labels`: Cluster assignments (-1 for outliers)
- k-distance plot for parameter selection

### 2.1.4 Cluster Analysis

For each cluster, the analysis computes:

- **Resistance proportions**: Percentage resistant for each antibiotic
- **MAR index statistics**: Mean, median, std of MAR index
- **Scored resistance**: Average resistance score
- **Metadata distributions**: Most common species, sources, regions, ESBL status

**Example:**

```python
# Analyze clusters
analysis_df = analyzer.analyze_clusters(cluster_labels)

print(analysis_df[['cluster', 'size', 'mar_index_mean', 'mar_index_median']])
```

### 2.1.5 Cluster Visualizations

**Resistance Heatmap:**
```python
fig = analyzer.plot_cluster_heatmap(cluster_labels, figsize=(12, 8))
```
Shows resistance patterns for all isolates sorted by cluster.

**Cluster Composition:**
```python
fig = analyzer.plot_cluster_composition(cluster_labels, 
                                       metadata_col='bacterial_species')
```
Shows distribution of metadata (e.g., species) within each cluster.

## 2.2 Dimensionality Reduction

### 2.2.1 PCA (Principal Component Analysis)

**Purpose:** Linear dimensionality reduction to identify main sources of variation.

**Example:**

```python
# Run PCA
pca_results = analyzer.pca_analysis(n_components=3)

print(f"Explained variance: {pca_results['explained_variance_ratio']}")
print(f"Cumulative variance: {pca_results['cumulative_variance']}")

# Visualize
fig = analyzer.plot_dimred_scatter(method='pca', color_by='bacterial_species')
```

**Output:**
- PC1, PC2, PC3 components
- Explained variance ratios
- Cumulative variance explained

### 2.2.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Purpose:** Non-linear dimensionality reduction for visualization.

**Example:**

```python
# Run t-SNE
tsne_results = analyzer.tsne_analysis(n_components=2, perplexity=30.0)

# Visualize
fig = analyzer.plot_dimred_scatter(method='tsne', color_by='sample_source')
```

**Output:**
- TSNE1, TSNE2 2D embedding
- Preserves local structure better than PCA

### 2.2.3 UMAP (Uniform Manifold Approximation and Projection)

**Purpose:** Non-linear dimensionality reduction that preserves both local and global structure.

**Example:**

```python
# Run UMAP (requires umap-learn package)
umap_results = analyzer.umap_analysis(n_components=2, n_neighbors=15, min_dist=0.1)

# Visualize
fig = analyzer.plot_dimred_scatter(method='umap', color_by='administrative_region')
```

**Output:**
- UMAP1, UMAP2 2D embedding
- Better preserves global structure than t-SNE

### 2.2.4 Visualization Options

Scatter plots can be colored by:
- **Species**: `color_by='bacterial_species'`
- **Sample source**: `color_by='sample_source'`
- **Region**: `color_by='administrative_region'`
- **MAR index**: `color_by='mar_index'` (continuous)
- **Clusters**: `color_by='cluster'` (after clustering)
- **ESBL status**: `color_by='esbl'`

### 2.2.5 Interpretation

**Key questions to explore:**
- Do high-MAR isolates group together?
- Are species/regions separated in AMR-space?
- Do clusters from k-means correspond to natural groupings?
- Which antibiotics drive the main components in PCA?

## 2.3 Association Rule Mining

### 2.3.1 Transaction Preparation

Transform each isolate into a transaction with items:
- Resistance items: `R_ampicillin`, `R_cefotaxime`, etc.
- Metadata items: `species=escherichia_coli`, `source=drinking_water`
- MAR category: `MAR_high` if MAR index ≥ threshold

**Example:**

```python
# Prepare transactions
transactions = analyzer.prepare_transactions(
    resistance_threshold=0.5,
    include_metadata=True,
    mar_threshold=0.3  # High MAR if ≥ 0.3
)

print(f"Transactions: {transactions.shape}")
print(f"Items: {transactions.columns.tolist()}")
```

### 2.3.2 Apriori Algorithm

**Purpose:** Mine frequent itemsets and generate association rules.

**Parameters:**
- `min_support`: Minimum support threshold (e.g., ≥ 5% = 0.05)
- `min_confidence`: Minimum confidence threshold (e.g., ≥ 60% = 0.6)
- `min_lift`: Minimum lift threshold (e.g., > 1.0)

**Example:**

```python
# Mine rules with Apriori
rules = analyzer.apriori_mining(
    transactions,
    min_support=0.05,
    min_confidence=0.6,
    min_lift=1.0
)

# Display top rules
formatted_rules = analyzer.interpret_rules(rules, top_n=10)
```

### 2.3.3 FP-Growth Algorithm

**Purpose:** Efficient algorithm for mining frequent itemsets (faster than Apriori).

**Example:**

```python
# Mine rules with FP-Growth
rules = analyzer.fpgrowth_mining(
    transactions,
    min_support=0.05,
    min_confidence=0.6,
    min_lift=1.0
)

# Save rules
rules.to_csv('association_rules.csv', index=False)
```

### 2.3.4 Rule Interpretation

**Rule format:** `{antecedent} => {consequent}`

**Metrics:**
- **Support**: Frequency of itemset in dataset
- **Confidence**: P(consequent | antecedent)
- **Lift**: How much more likely consequent occurs with antecedent vs. independently

**Example rules:**
```
{R_ampicillin, R_cefotaxime} => {R_ceftiofur}
  Support: 0.15, Confidence: 0.75, Lift: 2.3

{species=klebsiella, source=effluent} => {MAR_high}
  Support: 0.08, Confidence: 0.82, Lift: 3.1
```

**Interpretation:**
- High lift (> 2): Strong association between items
- High confidence (> 0.7): Reliable rule for prediction
- Rules reveal co-resistance patterns and contextual relationships

### 2.3.5 Common Patterns to Look For

1. **Co-resistance patterns**: Which antibiotics tend to be resistant together?
2. **Species-specific patterns**: Do certain species show specific resistance profiles?
3. **Source-specific patterns**: Are certain sources associated with high MAR?
4. **Regional patterns**: Do specific regions show distinct resistance patterns?

## Examples

See comprehensive examples in:
- `examples_unsupervised.py`: 10 detailed examples covering all Phase 2 functionality

Run examples:
```bash
python examples_unsupervised.py
```

**Example outputs:**
- Example 1: K-means with elbow method and silhouette analysis
- Example 2: Hierarchical clustering with dendrogram
- Example 3: DBSCAN with parameter selection
- Example 4-6: PCA, t-SNE, UMAP analysis
- Example 7-8: Association rule mining with Apriori and FP-Growth
- Example 9: Combined clustering and dimensionality reduction
- Example 10: Comparative analysis of clustering methods

## Testing - Phase 2

Run comprehensive test suite:

```bash
python test_unsupervised_analysis.py
```

**Test Coverage:**
- 20 tests covering all unsupervised methods
- Clustering: k-means, hierarchical, DBSCAN
- Dimensionality reduction: PCA, t-SNE, UMAP
- Association rule mining: Apriori, FP-Growth
- Visualization functions
- Integration workflows

## Best Practices - Phase 2

1. **For Clustering**: Use ordinal encoding with standardized features
2. **Optimal k Selection**: Use both elbow method and silhouette scores
3. **DBSCAN Parameters**: Use k-distance plot to select eps
4. **Dimensionality Reduction**: Use PCA first for linear relationships, then t-SNE/UMAP for non-linear
5. **Association Rules**: Start with higher support (0.1) and adjust down if needed
6. **Interpretation**: Always analyze clusters by multiple metadata dimensions

## Workflow Recommendations

### Complete Unsupervised Analysis Workflow

1. **Prepare data with ordinal encoding**
2. **Run multiple clustering methods** (k-means, hierarchical, DBSCAN)
3. **Compare clustering results** using silhouette scores
4. **Analyze cluster characteristics** (resistance patterns, MAR index, metadata)
5. **Perform dimensionality reduction** (PCA, t-SNE)
6. **Visualize clusters** in reduced dimensional space
7. **Mine association rules** to find co-resistance patterns
8. **Interpret findings** in context of species, sources, regions

## API Reference

### UnsupervisedAMRAnalysis Class

Main class for unsupervised analysis.

**Initialization:**
```python
analyzer = UnsupervisedAMRAnalysis(
    df,                    # Prepared dataframe
    feature_cols,          # Feature columns for analysis
    metadata_cols=None     # Metadata columns for labeling
)
```

**Clustering Methods:**
- `kmeans_clustering(k_range, standardize=True, random_state=42)`
- `hierarchical_clustering(n_clusters, linkage_method='ward', standardize=True)`
- `dbscan_clustering(eps, min_samples, standardize=True)`
- `analyze_clusters(cluster_labels, method_name)`
- `plot_kmeans_evaluation(figsize)`
- `plot_dendrogram(max_display_levels, figsize)`
- `plot_k_distance(k, figsize)`
- `plot_cluster_heatmap(cluster_labels, resistance_cols, figsize)`
- `plot_cluster_composition(cluster_labels, metadata_col, figsize)`

**Dimensionality Reduction Methods:**
- `pca_analysis(n_components, standardize=True)`
- `tsne_analysis(n_components, perplexity, standardize=True, random_state=42)`
- `umap_analysis(n_components, n_neighbors, min_dist, standardize=True, random_state=42)`
- `plot_dimred_scatter(method, color_by, figsize, alpha)`

**Association Rule Mining Methods:**
- `prepare_transactions(resistance_threshold, include_metadata, mar_threshold)`
- `apriori_mining(transactions, min_support, min_confidence, min_lift, max_len)`
- `fpgrowth_mining(transactions, min_support, min_confidence, min_lift, max_len)`
- `interpret_rules(rules, top_n)`

**Convenience Functions:**
- `quick_clustering_analysis(df, feature_cols, methods, k)`
- `quick_dimred_analysis(df, feature_cols, methods)`

---

# Phase 3: Supervised Pattern Recognition

Phase 3 implements supervised machine learning for three classification tasks.

## Quick Start - Phase 3

### Task 1: High MAR Prediction

```python
from data_preparation import AMRDataPreparation
from supervised_analysis import SupervisedAMRAnalysis

# Prepare data
prep = AMRDataPreparation('rawdata.csv')
df = prep.prepare_data(
    include_binary=True,
    include_ordinal=False,
    include_onehot=False,
    scale=False
)

# Get features
groups = prep.get_feature_groups()
feature_cols = groups['binary_resistance']

# Run Task 1: Predict high MAR/MDR
analyzer = SupervisedAMRAnalysis(df)
results = analyzer.task1_high_mar_prediction(
    feature_cols=feature_cols,
    threshold=0.3,
    include_tuning=True,
    tune_top_n=3
)

print(f"Best Model: {results['best_model']}")
print(f"Test F1: {results['test_metrics']['f1']:.4f}")
print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
```

### Task 2: Species Classification

```python
# Run Task 2: Predict species from resistance profile
results = analyzer.task2_species_classification(
    feature_cols=feature_cols,
    min_samples=10,
    include_tuning=True
)

print(f"Best Model: {results['best_model']}")
print(f"Test F1 (macro): {results['test_metrics']['f1']:.4f}")
print(f"Test F1 (weighted): {results['test_metrics']['f1_weighted']:.4f}")
```

### Task 3: Region/Source Classification

```python
# Add species features for Task 3
species_cols = [col for col in df.columns if 'bacterial_species_' in col]
feature_cols_task3 = feature_cols + species_cols

# Run Task 3: Predict region
results = analyzer.task3_region_source_classification(
    feature_cols=feature_cols_task3,
    target_col='administrative_region',
    include_tuning=True
)

print(f"Best Model: {results['best_model']}")
print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
```

### Quick All Tasks

```python
from supervised_analysis import quick_supervised_analysis

# Run all tasks at once
results = quick_supervised_analysis(
    df,
    task='all',
    include_tuning=True
)

for task_name, task_results in results.items():
    print(f"\n{task_name.upper()}:")
    print(f"  Best Model: {task_results['best_model']}")
    print(f"  Test F1: {task_results['test_metrics']['f1']:.4f}")
```

## 3.0 Common Pipeline Elements

### 3.0.1 Data Splitting (70/15/15)

All three tasks use stratified splitting:

```python
X_train, X_val, X_test, y_train, y_val, y_test = analyzer.stratified_split(
    X, y,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_state=42
)
```

**Features:**
- Stratified by target class to maintain class proportions
- Removes rows with missing target values
- Returns train (70%), validation (15%), and test (15%) sets

### 3.0.2 Evaluation Metrics

**Binary Classification (Task 1):**
- Accuracy
- Precision
- Recall
- F1-score (positive class = high MAR = 1)
- Confusion matrix

**Multiclass Classification (Tasks 2 & 3):**
- Accuracy
- Macro-averaged precision, recall, F1
- Weighted F1
- Confusion matrix

```python
metrics = analyzer.evaluate_model(y_true, y_pred, average='binary')
# Returns: {'accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'}
```

### 3.0.3 Algorithms

All tasks use 6 classification algorithms:

1. **Logistic Regression**
   - Hyperparameters: C (regularization strength)
   
2. **Random Forest**
   - Hyperparameters: n_estimators, max_depth, min_samples_leaf
   
3. **Gradient Boosting Machine (GBM)**
   - Hyperparameters: n_estimators, learning_rate, max_depth
   
4. **Naive Bayes**
   - Hyperparameters: var_smoothing (alpha)
   
5. **Support Vector Machine (SVM)**
   - Hyperparameters: C, kernel (linear/RBF), gamma
   
6. **k-Nearest Neighbors (kNN)**
   - Hyperparameters: n_neighbors, weights

**Pipeline includes:**
- Imputation (median for numeric features)
- One-hot encoding (for categorical features)
- Scaling (for SVM, kNN, Logistic Regression)
- Classifier

```python
# Example: Get all algorithm configs
configs = analyzer.get_algorithm_configs()

# Train all models
results = analyzer.train_and_evaluate_all_models(
    X_train, X_val, y_train, y_val,
    task_type='binary'  # or 'multiclass'
)
```

## 3.1 Task 1: Predict High MAR/MDR

### 3.1.1 Goal

Binary classification: Predict whether an isolate is multidrug-resistant (high MAR) based on its resistance pattern.

### 3.1.2 Target Variable

```python
# Define high_MAR threshold
high_MAR = 1 if mar_index >= 0.3 else 0
```

The threshold can be customized:

```python
y = analyzer.create_high_mar_target(threshold=0.3)  # Default
y = analyzer.create_high_mar_target(threshold=0.4)  # Stricter
```

### 3.1.3 Features

**Primary features:**
- Binary resistance features: `ampicillin_binary`, `cefotaxime_binary`, etc.

**Optional context features:**
- One-hot encoded `bacterial_species`
- One-hot encoded `sample_source`
- One-hot encoded `administrative_region`

**Note:** Do NOT include `mar_index` itself as a feature (it's the target).

```python
# Option A: AMR features only
feature_cols = groups['binary_resistance']

# Option B: AMR + context
feature_cols = groups['binary_resistance'] + groups['context_encoded']
```

### 3.1.4 Model Training and Selection

**Workflow:**

1. **Baseline training:** Train all 6 models on train set
2. **Validation:** Evaluate on validation set
3. **Hyperparameter tuning:** Grid search for top N models
4. **Model selection:** Choose best based on validation F1-score

```python
results = analyzer.task1_high_mar_prediction(
    feature_cols=feature_cols,
    threshold=0.3,
    include_tuning=True,  # Enable hyperparameter tuning
    tune_top_n=3          # Tune top 3 models
)

# Access all model results
for name, result in results['all_models'].items():
    print(f"{name}: Val F1 = {result['val_metrics']['f1']:.4f}")

# Best model
print(f"Best: {results['best_model']}")
print(f"Best params: {results['best_params']}")
```

### 3.1.5 Final Training and Test Evaluation

**Workflow:**

1. Combine train + validation → `train_full`
2. Re-fit best model on `train_full`
3. Evaluate once on test set
4. Report final metrics

```python
# Final test metrics
print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
print(f"Test Precision: {results['test_metrics']['precision']:.4f}")
print(f"Test Recall: {results['test_metrics']['recall']:.4f}")
print(f"Test F1: {results['test_metrics']['f1']:.4f}")
print(f"Confusion Matrix:\n{results['test_metrics']['confusion_matrix']}")
```

## 3.2 Task 2: Species Classification from Resistance Profile

### 3.2.1 Goal

Multiclass classification: Predict `bacterial_species` using only AMR resistance patterns.

### 3.2.2 Target Variable

```python
y = df['bacterial_species']
```

**Rare species handling:**
- Species with < `min_samples` examples are grouped as "Other"
- Default: `min_samples=10`

```python
results = analyzer.task2_species_classification(
    feature_cols=feature_cols,
    min_samples=10  # Group species with <10 samples
)
```

### 3.2.3 Features

**Primary features:**
- Binary resistance features: All `*_binary` columns

**Strategy:**
- Use ONLY AMR features (do not include species, region, or source)
- This tests whether AMR patterns alone can distinguish species

```python
# Prepare data WITHOUT one-hot encoding species
prep = AMRDataPreparation('rawdata.csv')
df = prep.prepare_data(
    include_binary=True,
    include_onehot=False,  # Don't encode species (it's the target!)
    scale=False
)

feature_cols = groups['binary_resistance']
```

### 3.2.4 Pipeline and Evaluation

**Workflow:**

1. 70/15/15 stratified split by `bacterial_species`
2. Run all 6 algorithms
3. Compute multiclass metrics:
   - Accuracy
   - Macro-averaged F1 (equal weight per class)
   - Weighted F1 (weighted by class frequency)
   - Confusion matrix
4. Tune hyperparameters for top models
5. Select best model

```python
results = analyzer.task2_species_classification(
    feature_cols=feature_cols,
    min_samples=10,
    include_tuning=True,
    tune_top_n=3
)
```

### 3.2.5 Final Evaluation

```python
print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
print(f"Test F1 (macro): {results['test_metrics']['f1']:.4f}")
print(f"Test F1 (weighted): {results['test_metrics']['f1_weighted']:.4f}")

# Confusion matrix shows which species are confused
cm = results['test_metrics']['confusion_matrix']
print("Confusion Matrix:")
print(cm)
```

**Interpretation:**
- High accuracy → AMR patterns distinguish species well
- Confusion matrix → Identifies which species have similar resistance profiles

## 3.3 Task 3: Region/Source Classification

### 3.3.1 Goal

Multiclass classification: Predict where an isolate came from based on AMR pattern + species.

### 3.3.2 Target Options

**Option A: Administrative Region**

```python
results = analyzer.task3_region_source_classification(
    feature_cols=feature_cols,
    target_col='administrative_region'
)
```

**Option B: Sample Source**

```python
results = analyzer.task3_region_source_classification(
    feature_cols=feature_cols,
    target_col='sample_source'
)
```

**Rare class handling:**
- Classes with < `min_samples` are grouped as "Other"

### 3.3.3 Features

**Required features:**
- Binary resistance features: All `*_binary` columns
- One-hot encoded `bacterial_species`

**Rationale:**
- Investigates whether certain species + AMR signatures are associated with regions/sources

```python
# Prepare features
feature_cols = groups['binary_resistance']

# Add species one-hot columns
species_cols = [col for col in df.columns if 'bacterial_species_' in col]
feature_cols += species_cols
```

### 3.3.4 Pipeline and Evaluation

Same workflow as Task 2:

1. 70/15/15 stratified split by target
2. Train all 6 algorithms
3. Evaluate with multiclass metrics
4. Tune top candidates
5. Select best model

```python
results = analyzer.task3_region_source_classification(
    feature_cols=feature_cols,
    target_col='sample_source',  # or 'administrative_region'
    min_samples=10,
    include_tuning=True,
    tune_top_n=3
)
```

### 3.3.5 Final Evaluation

```python
print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
print(f"Test F1 (macro): {results['test_metrics']['f1']:.4f}")
print(f"Test F1 (weighted): {results['test_metrics']['f1_weighted']:.4f}")

# Confusion matrix
cm = results['test_metrics']['confusion_matrix']
print("Confusion Matrix:")
print(cm)
```

## Examples - Phase 3

See comprehensive examples in `examples_supervised.py`:

```bash
python examples_supervised.py
```

**Included examples:**

1. **Example 1:** Task 1 - High MAR prediction with tuning
2. **Example 2:** Task 2 - Species classification
3. **Example 3:** Task 3 - Region classification
4. **Example 4:** Task 3 - Source classification
5. **Example 5:** Quick all tasks
6. **Example 6:** Compare different feature sets
7. **Example 7:** Test custom MAR thresholds

## Testing - Phase 3

Run comprehensive test suite:

```bash
python test_supervised_analysis.py
```

**Test Coverage:**
- 20 tests covering all supervised methods
- Data splitting and stratification
- All 6 classification algorithms
- Binary and multiclass tasks
- Hyperparameter tuning
- Evaluation metrics
- Rare class grouping
- Pipeline persistence

## Best Practices - Phase 3

1. **Data Preparation:**
   - Use binary encoding for supervised tasks
   - Keep original context columns for Tasks 2 & 3
   - Do not include target variable as a feature

2. **Feature Selection:**
   - Task 1: Use AMR features ± context
   - Task 2: Use ONLY AMR features (not species/region)
   - Task 3: Use AMR + species features (not target context)

3. **Model Selection:**
   - Use validation F1 as primary metric
   - Consider precision/recall trade-offs
   - Examine confusion matrix for insights

4. **Hyperparameter Tuning:**
   - Tune top 3 models to save time
   - Use GridSearchCV with 5-fold CV
   - Focus on models with high baseline performance

5. **Interpretation:**
   - Report both macro and weighted F1 for multiclass
   - Analyze confusion matrix to understand errors
   - Consider class imbalance in interpretation

## API Reference - Phase 3

### SupervisedAMRAnalysis Class

Main class for supervised analysis.

**Initialization:**
```python
analyzer = SupervisedAMRAnalysis(df)  # df from data_preparation
```

**Task Methods:**
- `task1_high_mar_prediction(feature_cols, threshold=0.3, include_tuning=True, tune_top_n=3)`
- `task2_species_classification(feature_cols, min_samples=10, include_tuning=True, tune_top_n=3)`
- `task3_region_source_classification(feature_cols, target_col, min_samples=10, include_tuning=True, tune_top_n=3)`

**Core Methods:**
- `create_high_mar_target(threshold=0.3)` - Create binary target
- `stratified_split(X, y, train_size=0.7, val_size=0.15, test_size=0.15)` - Split data
- `get_algorithm_configs()` - Get all algorithm configurations
- `create_pipeline(model, numeric_features, scale=True)` - Create sklearn pipeline
- `evaluate_model(y_true, y_pred, average='binary')` - Calculate metrics
- `train_and_evaluate_all_models(X_train, X_val, y_train, y_val, task_type)` - Train all 6 models
- `tune_hyperparameters(X_train, y_train, model_name, param_grid, pipeline)` - Grid search
- `select_best_model(results, metric='f1')` - Select best by metric
- `final_evaluation(pipeline, X_train, X_val, X_test, y_train, y_val, y_test)` - Final test

**Convenience Function:**
- `quick_supervised_analysis(df, task='all', feature_cols=None, **kwargs)` - Run tasks quickly

---

# Phase 4: Model Deployment

Phase 4 implements comprehensive model deployment functionality for trained AMR models, enabling their use in production environments.

## Overview

Phase 4 provides:
1. **Model Saving**: Save trained pipelines with complete metadata
2. **Model Loading**: Load saved models for deployment
3. **Batch Prediction**: Predict from CSV files
4. **Single Prediction**: Predict for individual isolates
5. **Command-line Tools**: Ready-to-use deployment scripts

## Quick Start - Phase 4

### 4.1 Training and Saving a Model

```python
from data_preparation import AMRDataPreparation
from supervised_analysis import SupervisedAMRAnalysis

# Prepare data
prep = AMRDataPreparation('rawdata.csv')
df = prep.prepare_data(
    include_binary=True,
    include_ordinal=False,
    include_onehot=True,
    scale=False,
    drop_original_int=True
)

# Get features
groups = prep.get_feature_groups()
feature_cols = groups['binary_resistance']

# Train and save model
analyzer = SupervisedAMRAnalysis(df)
results = analyzer.task1_high_mar_prediction(
    feature_cols=feature_cols,
    threshold=0.3,
    include_tuning=True,
    tune_top_n=3,
    save_model_path='high_MAR_model.pkl'  # Auto-save after training
)
```

This creates two files:
- `high_MAR_model.pkl` - The trained pipeline
- `high_MAR_model_metadata.json` - Model metadata and documentation

### 4.2 Using a Saved Model for Deployment

#### 4.2.1 Batch Prediction from CSV

```python
from model_deployment import ModelDeployment

# Load model
deployment = ModelDeployment('high_MAR_model.pkl')

# Make predictions on new data
results = deployment.predict_from_csv(
    input_csv='new_isolates.csv',
    output_csv='predictions.csv',
    include_proba=True,      # Include prediction probabilities
    include_original=True    # Include original columns in output
)
```

#### 4.2.2 Single Isolate Prediction

```python
# For real-time prediction (e.g., web applications)
features = {
    'AMP_binary': 1,
    'GEN_binary': 0,
    'CIP_binary': 0,
    # ... all required features
}

result = deployment.predict_single(features, return_proba=True)
print(f"Prediction: {result['prediction']}")
print(f"Probability High MAR: {result['probability_class_1']:.2%}")
```

#### 4.2.3 Command-line Deployment

```bash
# View model information
python deploy_model.py --model high_MAR_model.pkl --info

# Make predictions
python deploy_model.py \
    --model high_MAR_model.pkl \
    --input new_isolates.csv \
    --output predictions.csv

# Without probabilities
python deploy_model.py \
    --model high_MAR_model.pkl \
    --input new_isolates.csv \
    --output predictions.csv \
    --no-proba
```

#### 4.2.4 Web Deployment with FastAPI

```bash
# Start FastAPI server
uvicorn api:app --reload --port 8000

# API will be available at:
# - http://localhost:8000 (API root)
# - http://localhost:8000/docs (Swagger UI)
# - http://localhost:8000/redoc (ReDoc documentation)
```

**Available Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `GET /models` - List available models
- `POST /models/info` - Get model information
- `POST /predict` - Single isolate prediction
- `POST /predict/batch` - Batch prediction from CSV

**Example API Usage:**

```python
import requests

# Single prediction
response = requests.post(
    'http://localhost:8000/predict',
    json={
        'features': {
            'ampicillin_binary': 1.0,
            'gentamicin_binary': 0.0,
            'ciprofloxacin_binary': 0.0,
            # ... all required features
        },
        'model_path': 'high_MAR_model.pkl',
        'return_proba': True
    }
)

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability_class_1']:.2%}")
```

#### 4.2.5 Web UI with Streamlit

```bash
# Start Streamlit app
streamlit run app.py

# App will open in your browser at http://localhost:8501
```

**Features:**
- 📊 **Model Information**: View model details, metrics, and hyperparameters
- 🔬 **Single Prediction**: Interactive form for predicting individual isolates
- 📁 **Batch Prediction**: Upload CSV files for bulk predictions with visualization
- 📈 **Results Visualization**: Charts and graphs for prediction distribution
- ⬇️ **Download Results**: Export predictions as CSV

The Streamlit app provides a user-friendly interface for:
1. Loading and viewing model information
2. Making single isolate predictions with visual confidence indicators
3. Uploading CSV files for batch predictions
4. Visualizing prediction distributions with interactive charts
5. Downloading prediction results

## Model Metadata

Each saved model includes comprehensive metadata:

```json
{
  "model_info": {
    "task_name": "high_mar_prediction",
    "model_type": "RandomForest",
    "hyperparameters": {...},
    "created_at": "2025-12-09T..."
  },
  "features": {
    "feature_columns": [...],
    "num_features": 45,
    "feature_format": "Binary resistance encoding (R=1, S/I=0)"
  },
  "metrics": {
    "training": {...},
    "validation": {...},
    "test": {...}
  },
  "data_splits": {
    "train_size": 380,
    "val_size": 81,
    "test_size": 82
  }
}
```

## Deployment Scenarios

### Scenario 1: Command-line Script

**Use case**: Process batch files periodically

```python
# deploy_batch.py
from model_deployment import predict_from_csv

predict_from_csv(
    model_path='high_MAR_model.pkl',
    input_csv='new_lab_results.csv',
    output_csv='predictions_' + date + '.csv'
)
```

### Scenario 2: Web Application

**Use case**: Interactive decision-support tool

```python
from flask import Flask, request, jsonify
from model_deployment import ModelDeployment

app = Flask(__name__)
deployment = ModelDeployment('high_MAR_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.json
    result = deployment.predict_single(features, return_proba=True)
    return jsonify({
        'prediction': 'High MAR' if result['prediction'] == 1 else 'Low MAR',
        'confidence': result['probability_class_1']
    })
```

### Scenario 3: Surveillance Dashboard

**Use case**: Monitor trends over time

```python
import pandas as pd
from model_deployment import ModelDeployment

# Load model
deployment = ModelDeployment('high_MAR_model.pkl')

# Process monthly data
for month in monthly_files:
    # Load new data
    df = pd.read_csv(month)
    
    # Make predictions
    predictions = deployment.predict(df, include_proba=True)
    
    # Aggregate by region/site/time
    df['prediction'] = predictions
    summary = df.groupby(['region', 'site'])['prediction'].agg(['sum', 'count', 'mean'])
    
    # Visualize trends
    plot_mdr_trends(summary)
```

## API Reference

### SupervisedAMRAnalysis

**Model Saving Methods:**

```python
# Save model with metadata
save_pipeline_with_metadata(
    pipeline, filepath, task_name, feature_cols, model_type,
    hyperparameters, train_metrics, val_metrics, test_metrics,
    splits, additional_info=None
)

# Simple save
save_model(pipeline, filepath, metadata=None)

# Load model
load_model(filepath)
```

**Task Method with Auto-save:**

```python
task1_high_mar_prediction(
    feature_cols,
    threshold=0.3,
    include_tuning=True,
    tune_top_n=3,
    save_model_path=None  # NEW: Optional path to save model
)
```

### ModelDeployment Class

```python
# Initialize
deployment = ModelDeployment(model_path)

# Get information
deployment.get_required_features()
deployment.get_model_info()
deployment.get_performance_metrics()

# Make predictions
predictions = deployment.predict(X, include_proba=True)

# Batch prediction
results = deployment.predict_from_csv(
    input_csv, output_csv,
    include_proba=True,
    include_original=True
)

# Single prediction
result = deployment.predict_single(features, return_proba=True)
```

### Convenience Functions

```python
from model_deployment import predict_from_csv, predict_single_isolate

# Batch prediction
predict_from_csv(model_path, input_csv, output_csv)

# Single prediction
predict_single_isolate(model_path, features, return_proba=True)
```

## Input Data Requirements

New data for prediction must include all required features:

1. **Binary resistance features**: All antibiotic columns with `_binary` suffix
2. **Feature names**: Must match training data exactly
3. **Missing values**: Will be imputed using pipeline's imputation strategy
4. **Format**: CSV file with column headers

Example input CSV structure:
```csv
AMP_binary,GEN_binary,CIP_binary,CTX_binary,...
1,0,0,1,...
0,1,0,0,...
```

## Output Format

Predictions include:

- **Prediction**: Class label (0 = Low MAR, 1 = High MAR)
- **Probabilities**: Probability for each class (optional)
- **Original data**: All input columns (optional)

Example output:
```csv
AMP_binary,GEN_binary,...,high_mar_prediction_prediction,probability_class_0,probability_class_1
1,0,...,1,0.12,0.88
0,1,...,0,0.91,0.09
```

## Best Practices

1. **Model Versioning**: Include version/date in model filename
   - `high_MAR_model_v1.0_20250609.pkl`

2. **Metadata Review**: Always check model metadata before deployment
   - Review features, metrics, and hyperparameters

3. **Feature Validation**: Ensure new data matches training data format
   - Use `get_required_features()` to verify

4. **Error Handling**: Wrap predictions in try-except blocks
   - Handle missing features gracefully

5. **Performance Monitoring**: Track prediction distribution over time
   - Alert on significant changes

6. **Model Retraining**: Retrain periodically with new data
   - Monitor for concept drift

## Examples

Complete examples are provided in `examples_deployment.py`:

```python
# Run all examples
python examples_deployment.py

# Or import specific examples
from examples_deployment import (
    example_1_train_and_save_model,
    example_2_load_and_predict,
    example_3_single_isolate_prediction,
    example_4_command_line_deployment
)
```

## Testing

Test the deployment functionality:

```bash
# Run deployment tests
python -m unittest test_model_deployment -v

# Run specific test
python -m unittest test_model_deployment.TestModelDeployment.test_07_predict_from_csv -v
```

---

## License

This module is part of the thesis project for AMR analysis.

## Author

Thesis Project - December 2025
