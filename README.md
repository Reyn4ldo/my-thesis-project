# AMR Analysis Pipeline - Complete Implementation

This repository provides a comprehensive analysis pipeline for Antimicrobial Resistance (AMR) data, implementing both data preparation (Phase 1) and unsupervised pattern recognition (Phase 2).

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Phase 1: Data Preparation](#phase-1-data-preparation)
- [Phase 2: Unsupervised Pattern Recognition](#phase-2-unsupervised-pattern-recognition)
- [Quick Start](#quick-start)
- [Testing](#testing)

## Overview

The AMR analysis pipeline consists of two main phases:

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

## Installation

### Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn mlxtend umap-learn
```

**Note:** `umap-learn` is optional but recommended for UMAP analysis.

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

## License

This module is part of the thesis project for AMR analysis.

## Author

Thesis Project - December 2025
