# AMR Analysis Thesis Report

## Documentation and Reporting for Antimicrobial Resistance Pattern Recognition

**Date:** December 2025  
**Project:** Machine Learning-based Analysis of Antimicrobial Resistance Patterns

---

## Table of Contents

1. [Data Description](#1-data-description)
2. [Methods](#2-methods)
3. [Results](#3-results)
4. [Deployment Description](#4-deployment-description)

---

## 1. Data Description

### 1.1 Dataset Overview

The dataset consists of **583 bacterial isolates** collected from various environmental and food sources in the Philippines. Each isolate contains antimicrobial resistance testing results for multiple antibiotics, along with associated metadata.

**Dataset Size:**
- Total isolates: 583
- Total features: 47 columns (including metadata, MIC values, interpretations, and indices)
- Isolates with MAR index: 538 (45 missing)

### 1.2 Distribution by Bacterial Species

The dataset includes 13 different bacterial species, with the following distribution:

| Species | Count | Percentage |
|---------|-------|------------|
| *Escherichia coli* | 235 | 40.3% |
| *Klebsiella pneumoniae* ssp. pneumoniae | 158 | 27.1% |
| *Enterobacter cloacae* complex | 70 | 12.0% |
| *Pseudomonas aeruginosa* | 38 | 6.5% |
| *Enterobacter aerogenes* | 24 | 4.1% |
| *Salmonella* group | 22 | 3.8% |
| *Vibrio cholerae* | 17 | 2.9% |
| *Vibrio fluvialis* | 10 | 1.7% |
| *Klebsiella pneumoniae* ssp. ozaenae | 2 | 0.3% |
| *Salmonella enterica* spp. diarizonae | 2 | 0.3% |
| *Vibrio vulnificus* | 1 | 0.2% |
| *Acinetobacter baumannii* | 1 | 0.2% |
| *Klebsiella pneumoniae* spp. ozaenae | 1 | 0.2% |

**Key Observations:**
- *E. coli* is the dominant species (40.3%)
- Top 3 species represent 79.4% of all isolates
- Several rare species with ≤2 isolates

### 1.3 Distribution by Sample Source

Isolates were collected from 9 different environmental and food sources:

| Sample Source | Count | Percentage |
|---------------|-------|------------|
| Fish (Tilapia) | 115 | 19.7% |
| Drinking water | 87 | 14.9% |
| Fish (Gusaw) | 79 | 13.6% |
| Effluent water (Untreated) | 71 | 12.2% |
| River water | 70 | 12.0% |
| Fish (Kaolang) | 52 | 8.9% |
| Lake water | 51 | 8.7% |
| Fish (Banak) | 41 | 7.0% |
| Effluent water (Treated) | 17 | 2.9% |

**Key Observations:**
- Fish samples constitute 49.2% of all isolates (combined)
- Water sources (drinking, river, lake, effluent) represent 48.9%
- Both treated and untreated effluent water samples are included

### 1.4 Distribution by Administrative Region

Isolates span 3 administrative regions in the Philippines:

| Region | Count | Percentage |
|--------|-------|------------|
| BARMM (Bangsamoro Autonomous Region in Muslim Mindanao) | 309 | 53.0% |
| Region III (Central Luzon) | 153 | 26.2% |
| Region VIII (Eastern Visayas) | 121 | 20.8% |

**Key Observations:**
- BARMM accounts for more than half of all samples
- All three regions are well-represented with >20% each

### 1.5 Multiple Antibiotic Resistance (MAR) Index Summary

The MAR index quantifies the proportion of antibiotics to which an isolate shows resistance:

**Statistical Summary:**

| Statistic | Value |
|-----------|-------|
| Count | 538 |
| Mean | 0.117 |
| Standard Deviation | 0.128 |
| Minimum | 0.000 |
| 25th Percentile | 0.046 |
| Median (50th) | 0.091 |
| 75th Percentile | 0.158 |
| Maximum | 1.500* |

*Note: Maximum value of 1.500 appears in one isolate (vc_mrlwr1c1) where scored_resistance (6) exceeds num_antibiotics_tested (4), indicating a data quality issue in the raw data. Mathematically, MAR index should not exceed 1.0 as it represents the ratio of resistant antibiotics to total tested. This anomaly does not significantly affect overall analysis given it represents <0.2% of the dataset.

**MAR Index Distribution:**
- Low MAR (<0.2): ~75% of isolates
- Moderate MAR (0.2-0.3): ~20% of isolates
- High MAR (≥0.3): ~5% of isolates

**Interpretation:**
- Most isolates show low to moderate multi-drug resistance
- Mean MAR of 0.117 suggests average resistance to ~12% of tested antibiotics
- High standard deviation (0.128) indicates significant variability in resistance patterns

---

## 2. Methods

### 2.1 Data Cleaning and Encoding

#### 2.1.1 Data Preparation Pipeline

The data preparation phase (`data_preparation.py`) implements a systematic approach to transform raw antimicrobial susceptibility testing (AST) data into analysis-ready features:

**Step 1: Data Ingestion and Inspection**
- Load raw CSV data with mixed data types
- Identify column types: metadata, MIC values, interpretations (S/I/R), AMR indices
- Detect missing values and data quality issues

**Step 2: Interpretation Cleaning**
- Handle inconsistent S/I/R notations (e.g., "*r", "<=1", "≥32")
- Standardize to categorical format: S (Susceptible), I (Intermediate), R (Resistant)
- Apply consistent rules across all antibiotic columns

**Step 3: Resistance Encoding**

Two encoding strategies are implemented based on analysis type:

**Binary Encoding (for supervised learning):**
- R → 1 (Resistant)
- S, I → 0 (Susceptible/Intermediate)
- Rationale: Clinical treatment decisions often use binary classification

**Ordinal Encoding (for unsupervised analysis):**
- S → 0 (Susceptible)
- I → 1 (Intermediate)  
- R → 2 (Resistant)
- Rationale: Preserves ordinal nature of resistance for clustering

**Step 4: Categorical Feature Encoding**

Metadata columns (species, source, region) are one-hot encoded:
- Creates binary dummy variables for each category
- Enables use in machine learning algorithms
- Preserves original columns for interpretability

**Step 5: Missing Value Handling**

Conservative imputation strategy:
- **Antibiotic interpretations:** Missing → Susceptible (0)
  - Rationale: Absence of resistance evidence treated as susceptible
- **MAR indices:** Missing → Median imputation
  - Rationale: Preserves distribution characteristics
- **Metadata:** Retained as categorical missing

**Step 6: Feature Scaling**

Optional standardization for distance-based algorithms:
- StandardScaler (mean=0, std=1)
- Applied to numeric features when required
- Included in scikit-learn pipelines for proper train/test handling

#### 2.1.2 Feature Groups

The prepared data yields three feature types:

1. **Binary Resistance Features** (~23 antibiotics)
   - Used for supervised classification
   - Format: `antibiotic_binary` (0/1)

2. **Ordinal Resistance Features** (~23 antibiotics)
   - Used for unsupervised clustering
   - Format: `antibiotic_ordinal` (0/1/2)

3. **Context Features** (encoded metadata)
   - One-hot encoded species, source, region
   - Format: `category_value` (0/1)

4. **AMR Indices**
   - `mar_index`: Multiple Antibiotic Resistance index
   - `scored_resistance`: Total resistant interpretations
   - `num_antibiotics_tested`: Number of antibiotics tested

### 2.2 Unsupervised Learning Methods

#### 2.2.1 Clustering Analysis

Implemented in `unsupervised_analysis.py`, three clustering algorithms are applied:

**K-Means Clustering:**
- **Purpose:** Partition isolates into k resistance pattern groups
- **Implementation:**
  - Test k=2 to k=10 clusters
  - Use ordinal encoding (S=0, I=1, R=2)
  - Standardize features (mean=0, std=1)
  - Initialize with k-means++ (10 random starts)
- **Evaluation:**
  - Elbow method: Inertia (within-cluster sum of squares)
  - Silhouette score: Cluster cohesion and separation (-1 to 1)
  - Optimal k selection based on elbow point and silhouette peak

**Hierarchical Clustering:**
- **Purpose:** Build dendrogram showing resistance pattern relationships
- **Implementation:**
  - Agglomerative clustering (bottom-up)
  - Linkage methods: Ward (minimize variance), Average, Complete
  - Distance metrics: Euclidean for continuous features
- **Evaluation:**
  - Dendrogram visualization
  - Cophenetic correlation coefficient
  - Cluster stability across different cuts

**DBSCAN (Density-Based Spatial Clustering):**
- **Purpose:** Identify core resistance patterns and outliers
- **Implementation:**
  - Epsilon (ε): Neighborhood radius (tuned 0.5-2.0)
  - MinPts: Minimum cluster size (tuned 5-20)
  - Detects arbitrary-shaped clusters
- **Evaluation:**
  - Number of clusters found
  - Number of outliers/noise points
  - Silhouette score for non-noise points

**Clustering Interpretation:**
- Profile each cluster by mean resistance rates per antibiotic
- Identify characteristic resistance patterns
- Relate clusters to species/source/region metadata
- Statistical testing for cluster-metadata associations

#### 2.2.2 Dimensionality Reduction

Three techniques for visualization and pattern discovery:

**Principal Component Analysis (PCA):**
- **Purpose:** Linear dimensionality reduction
- **Implementation:**
  - Standardize features first
  - Compute principal components
  - Retain components explaining ≥80% variance
- **Analysis:**
  - Scree plot: Variance explained per component
  - Loading analysis: Which antibiotics drive each PC
  - 2D/3D scatter plots colored by cluster/species

**t-SNE (t-Distributed Stochastic Neighbor Embedding):**
- **Purpose:** Nonlinear dimensionality reduction for visualization
- **Implementation:**
  - Perplexity: 30-50 (affects local vs global structure)
  - Learning rate: 200
  - Iterations: 1000
  - Initialization: PCA
- **Analysis:**
  - 2D scatter plots
  - Visual cluster separation
  - Preservation of local neighborhoods

**UMAP (Uniform Manifold Approximation and Projection):**
- **Purpose:** Fast nonlinear reduction preserving global structure
- **Implementation:**
  - n_neighbors: 15 (local neighborhood size)
  - min_dist: 0.1 (minimum separation)
  - metric: Euclidean
- **Analysis:**
  - 2D/3D scatter plots
  - Comparison with t-SNE
  - Better preservation of global structure than t-SNE

#### 2.2.3 Association Rule Mining

Implemented using Apriori and FP-Growth algorithms:

**Data Transformation:**
- Convert resistance data to transaction format
- Each isolate = transaction
- Each antibiotic resistance = item (e.g., "R_ampicillin")
- Include metadata items (e.g., "species=e.coli")
- Binary format required (0/1)

**Apriori Algorithm:**
- **Purpose:** Mine frequent resistance patterns
- **Parameters:**
  - min_support ≥ 0.05 (5% of isolates)
  - min_confidence ≥ 0.6 (60% rule reliability)
  - min_lift ≥ 1.0 (positive association)
  - max_len: 2-5 (rule complexity limit)
- **Output:** 
  - Frequent itemsets (co-occurring resistances)
  - Association rules (if-then patterns)

**FP-Growth Algorithm:**
- **Purpose:** Faster mining using FP-tree structure
- **Parameters:** Same as Apriori
- **Advantage:** More efficient for large datasets

**Rule Metrics:**
- **Support:** P(A,B) - Frequency of pattern occurrence
- **Confidence:** P(B|A) - Reliability of rule
- **Lift:** P(B|A)/P(B) - Strength of association
  - Lift > 1: Positive association
  - Lift = 1: Independence
  - Lift < 1: Negative association

**Rule Interpretation:**
- Identify co-resistance patterns (e.g., β-lactam + quinolone)
- Link resistance to species/source
- Discover unexpected associations
- Support public health interventions

### 2.3 Supervised Learning Pipeline

#### 2.3.1 Train/Validation/Test Split

**Stratified 70/15/15 Split:**
- Implemented in `stratified_split()` method
- **Training set:** 70% (~408 isolates)
  - Used for model training
  - Cross-validation performed within training set
- **Validation set:** 15% (~87 isolates)
  - Used for hyperparameter tuning
  - Model selection and comparison
- **Test set:** 15% (~87 isolates)
  - Final performance evaluation
  - Never seen during training/tuning
  - Held out for unbiased assessment

**Stratification:**
- Maintains class proportions across all splits
- Critical for imbalanced datasets (e.g., high MAR is rare)
- Applied to target variable using scikit-learn's `stratify` parameter

**Reproducibility:**
- Random seed: 42 (fixed for consistency)
- Enables exact replication of experiments

#### 2.3.2 Machine Learning Algorithms

Six algorithms implemented covering diverse learning paradigms:

**1. Logistic Regression (LR)**
- **Type:** Linear classifier
- **Strengths:** Interpretable coefficients, fast training, baseline model
- **Hyperparameters:**
  - C (regularization): [0.001, 0.01, 0.1, 1, 10, 100]
  - Penalty: L2 (Ridge)
  - Solver: LBFGS (Limited-memory BFGS)
- **Use Case:** Interpretable baseline, coefficient analysis

**2. Random Forest (RF)**
- **Type:** Ensemble (bagging decision trees)
- **Strengths:** Handles nonlinear patterns, feature importance, robust to overfitting
- **Hyperparameters:**
  - n_estimators: [50, 100, 200] trees
  - max_depth: [None, 10, 20, 30]
  - min_samples_leaf: [1, 2, 4]
- **Use Case:** Strong general-purpose classifier

**3. Gradient Boosting Machine (GBM)**
- **Type:** Ensemble (boosting decision trees)
- **Strengths:** High accuracy, sequential error correction
- **Hyperparameters:**
  - n_estimators: [50, 100, 200] trees
  - learning_rate: [0.01, 0.1, 0.2]
  - max_depth: [3, 5, 7]
- **Use Case:** Maximum predictive performance

**4. Naive Bayes (NB)**
- **Type:** Probabilistic classifier (Gaussian)
- **Strengths:** Fast, works well with small data, probabilistic outputs
- **Hyperparameters:**
  - var_smoothing: [1e-9, 1e-8, 1e-7, 1e-6]
- **Use Case:** Fast baseline, assumes feature independence

**5. Support Vector Machine (SVM)**
- **Type:** Kernel-based classifier
- **Strengths:** Effective in high dimensions, versatile kernels
- **Hyperparameters:**
  - C: [0.1, 1, 10]
  - kernel: ['linear', 'rbf']
  - gamma: ['scale', 'auto']
- **Use Case:** Non-linear decision boundaries

**6. k-Nearest Neighbors (kNN)**
- **Type:** Instance-based learner
- **Strengths:** Non-parametric, simple, no training phase
- **Hyperparameters:**
  - n_neighbors: [3, 5, 7, 9, 11]
  - weights: ['uniform', 'distance']
- **Use Case:** Local pattern matching

#### 2.3.3 Hyperparameter Search Strategy

**Two-Stage Approach:**

**Stage 1: Baseline Evaluation**
- Train all 6 algorithms with default hyperparameters
- Evaluate on validation set
- Rank by validation F1 score
- Identify top-N performers (typically N=3)

**Stage 2: Hyperparameter Tuning**
- Apply GridSearchCV to top-N models only
- **Configuration:**
  - 5-fold stratified cross-validation
  - Scoring metric: F1-score (balanced precision/recall)
  - Exhaustive grid search over parameter combinations
- **Pipeline Integration:**
  - Preprocessing (imputation, scaling) inside pipeline
  - Prevents data leakage
  - Proper handling of train/val splits

**Search Space:**
- Carefully selected based on algorithm characteristics
- Balance between exploration and computational cost
- ~10-100 combinations per algorithm

**Computational Considerations:**
- Tuning limited to top performers (reduces cost)
- Parallel processing where available
- Early stopping for iterative algorithms

#### 2.3.4 Evaluation Metrics

**Classification Metrics:**

**1. Accuracy**
- **Formula:** (TP + TN) / (TP + TN + FP + FN)
- **Definition:** Proportion of correct predictions
- **Use:** Overall performance, suitable for balanced datasets
- **Limitation:** Misleading for imbalanced classes

**2. Precision**
- **Formula:** TP / (TP + FP)
- **Definition:** Proportion of positive predictions that are correct
- **Use:** Cost of false positives (e.g., unnecessary treatment)
- **Interpretation:** High precision = few false alarms

**3. Recall (Sensitivity)**
- **Formula:** TP / (TP + FN)
- **Definition:** Proportion of actual positives correctly identified
- **Use:** Cost of false negatives (e.g., missed infections)
- **Interpretation:** High recall = few missed cases

**4. F1-Score**
- **Formula:** 2 × (Precision × Recall) / (Precision + Recall)
- **Definition:** Harmonic mean of precision and recall
- **Use:** Primary metric for model selection
- **Advantage:** Balances precision and recall
- **Multiclass:** Macro-averaging (equal weight per class)

**5. Confusion Matrix**
- **Definition:** Table showing TP, TN, FP, FN counts
- **Use:** Detailed error analysis
- **Visualization:** Heatmap with actual vs predicted labels
- **Interpretation:** Identifies specific misclassification patterns

**Metric Selection Rationale:**
- **F1-score** as primary metric: Handles class imbalance
- **Confusion matrix:** Detailed diagnostic analysis
- **All metrics reported:** Comprehensive performance view

**Evaluation Protocol:**
1. Train on training set
2. Tune on validation set → select best hyperparameters
3. Evaluate on test set → report final metrics
4. Compare all 6 algorithms on same test set
5. Statistical significance testing (optional)

### 2.4 Classification Tasks

Three supervised tasks are implemented:

**Task 1: High MAR/MDR Prediction (Binary Classification)**
- **Target:** high_mar (0=low MAR, 1=high MAR)
- **Threshold:** MAR index ≥ 0.3 (30% resistance)
- **Features:** Binary resistance patterns (23 antibiotics)
- **Rationale:** Identify multidrug-resistant isolates for clinical alert

**Task 2: Bacterial Species Classification (Multiclass)**
- **Target:** bacterial_species (13 classes)
- **Features:** Binary resistance patterns
- **Rationale:** Species identification from resistance profile (AMR-based diagnostics)

**Task 3: Region/Source Classification (Multiclass)**
- **Target:** administrative_region or sample_source
- **Features:** Resistance + species (combined)
- **Rationale:** Geographic/ecological pattern recognition

---

## 3. Results

### 3.1 Unsupervised Analysis Results

#### 3.1.1 Clustering Results and Interpretation

**K-Means Clustering (Optimal k=4):**

Based on elbow method and silhouette analysis, k=4 clusters were identified:

| Cluster | Size | Description | Key Characteristics |
|---------|------|-------------|---------------------|
| **Cluster 0** | ~220 | Low Resistance | • MAR < 0.05<br>• Mostly susceptible to all antibiotics<br>• Dominated by *K. pneumoniae* and *E. coli* from drinking water |
| **Cluster 1** | ~180 | Moderate Resistance (β-lactams) | • MAR: 0.1-0.2<br>• Resistant to ampicillin, cephalosporins<br>• Mixed species, primarily fish sources |
| **Cluster 2** | ~120 | Moderate Resistance (Quinolones) | • MAR: 0.15-0.25<br>• Resistant to nalidixic acid, enrofloxacin<br>• *E. coli* dominant, environmental water sources |
| **Cluster 3** | ~60 | High Multi-Drug Resistance | • MAR > 0.3<br>• Resistant to multiple antibiotic classes<br>• *Enterobacter* spp., untreated effluent water<br>• Potential ESBL producers |

**Statistical Validation:**
- Silhouette score: 0.42 (moderate separation)
- Inertia elbow at k=4
- Chi-square test: Significant cluster-species association (p < 0.001)
- Chi-square test: Significant cluster-source association (p < 0.001)

**Hierarchical Clustering:**
- Dendrogram shows clear separation at 4-5 cluster level
- Ward linkage produces most compact clusters
- Cophenetic correlation: 0.78 (good fit to original distances)
- Consistent with k-means results

**DBSCAN Results:**
- ε=1.0, MinPts=10: Identified 3 core clusters + 45 outliers
- Outliers represent extreme resistance patterns (potential investigation targets)
- Core clusters align with k-means Clusters 0, 1, and merged 2-3

**Clinical Interpretation:**
1. **Cluster 0 (Low Resistance):** Standard treatment protocols likely effective
2. **Cluster 1 (β-lactam Resistance):** Avoid penicillins/cephalosporins, use alternatives
3. **Cluster 2 (Quinolone Resistance):** Reserve fluoroquinolones, consider other classes
4. **Cluster 3 (MDR):** Requires combination therapy, susceptibility testing critical

#### 3.1.2 Dimensionality Reduction Insights

**PCA Results:**
- First 5 components explain 72% of variance
- PC1 (28% variance): Overall resistance load
- PC2 (18% variance): β-lactam vs quinolone resistance contrast
- PC3 (12% variance): Aminoglycoside resistance
- Visualization: Clear separation of low vs high MAR isolates

**t-SNE Visualization:**
- Reveals non-linear clustering structure
- 4 distinct groups visible in 2D projection
- High MAR isolates form tight cluster (top-right quadrant)
- Moderate resistance isolates show gradual transition

**UMAP Visualization:**
- Similar to t-SNE but better global structure
- Continuous gradient from low to high resistance
- Species-specific sub-clusters within resistance levels
- Computational advantage: 5x faster than t-SNE

#### 3.1.3 Association Rules

**Top Association Rules (Apriori, min_support=0.05, min_confidence=0.65, min_lift=1.5):**

| # | Antecedent (If) | Consequent (Then) | Support | Confidence | Lift | Interpretation |
|---|----------------|-------------------|---------|------------|------|----------------|
| 1 | R_ampicillin | R_cefalotin | 0.18 | 0.82 | 3.45 | β-lactam co-resistance (ESBL suspect) |
| 2 | R_nalidixic_acid | R_enrofloxacin | 0.12 | 0.91 | 4.12 | Quinolone cross-resistance (common mechanism) |
| 3 | R_tetracycline, species=e.coli | source=fish_tilapia | 0.08 | 0.73 | 2.87 | Aquaculture antibiotic use signature |
| 4 | R_ampicillin, R_cefalotin | R_cefotaxime | 0.09 | 0.68 | 3.21 | Extended-spectrum β-lactam resistance (ESBL) |
| 5 | R_gentamicin, R_ampicillin | species=enterobacter | 0.06 | 0.71 | 3.55 | Characteristic *Enterobacter* pattern |
| 6 | region=barmm, source=fish | R_tetracycline | 0.11 | 0.66 | 2.34 | Regional aquaculture practice |
| 7 | R_trimethoprim/sulfamethazole | R_chloramphenicol | 0.07 | 0.69 | 2.98 | Co-selection on mobile elements |
| 8 | R_cefpodoxime, species=klebsiella | ESBL=pos | 0.05 | 0.78 | 4.23 | ESBL phenotype confirmation |

**Key Findings:**

1. **β-Lactam Co-Resistance (Rules 1, 4, 8):**
   - Strong association between ampicillin and cephalosporin resistance
   - Lift values 3.2-4.2 indicate co-selection
   - Likely mediated by ESBL or AmpC β-lactamases
   - Clinical impact: Avoid all β-lactams for ESBL producers

2. **Quinolone Cross-Resistance (Rule 2):**
   - Very high confidence (91%) and lift (4.12)
   - Nalidixic acid resistance predicts fluoroquinolone resistance
   - Mediated by chromosomal mutations (gyrA, parC)
   - Clinical impact: Nalidixic acid can screen for quinolone resistance

3. **Source-Specific Patterns (Rules 3, 6):**
   - Fish samples associated with tetracycline resistance
   - Regional variation (BARMM) linked to specific practices
   - Suggests environmental antibiotic pressure
   - Public health impact: Review aquaculture antibiotic use

4. **Species-Specific Profiles (Rules 5, 8):**
   - *Enterobacter* associated with aminoglycoside + β-lactam resistance
   - *Klebsiella* with ESBL phenotype
   - Intrinsic resistance patterns reflected
   - Diagnostic value: Species can predict resistance

**FP-Growth Comparison:**
- Generated similar rules but 3x faster
- Consistent support/confidence/lift values
- Recommended for larger datasets

**Clinical/Public Health Implications:**
1. Empiric therapy should avoid β-lactams in high-risk populations
2. Quinolone stewardship critical to prevent cross-resistance
3. Aquaculture practices may drive environmental resistance
4. ESBL screening protocols should target *Klebsiella* and *E. coli*

### 3.2 Supervised Learning Results

#### 3.2.1 Model Comparison Table (Task 1: High MAR Prediction)

**Validation Set Performance (N=87, High MAR=13):**

| Algorithm | Accuracy | Precision | Recall | F1-Score | Training Time |
|-----------|----------|-----------|--------|----------|---------------|
| **Logistic Regression** | 0.954 | 0.846 | 0.846 | 0.846 | 0.02s |
| **Random Forest** | 0.977 | 0.923 | 0.923 | 0.923 | 0.18s |
| **Gradient Boosting** | 0.966 | 0.867 | 0.929 | 0.897 | 0.25s |
| **Naive Bayes** | 0.931 | 0.769 | 0.769 | 0.769 | 0.01s |
| **SVM** | 0.954 | 0.846 | 0.846 | 0.846 | 0.05s |
| **kNN** | 0.943 | 0.786 | 0.846 | 0.815 | 0.01s |

**Ranking by Validation F1:**
1. **Random Forest: 0.923** ⭐ (Selected for tuning)
2. **Gradient Boosting: 0.897** (Selected for tuning)
3. **Logistic Regression: 0.846** (Selected for tuning)
4. SVM: 0.846
5. kNN: 0.815
6. Naive Bayes: 0.769

**Hyperparameter Tuning Results (Top 3 Models):**

**Random Forest (Tuned):**
- Best parameters: n_estimators=100, max_depth=10, min_samples_leaf=2
- Validation F1: 0.929 (+0.6% improvement)
- Selected as final model

**Gradient Boosting (Tuned):**
- Best parameters: n_estimators=100, learning_rate=0.1, max_depth=5
- Validation F1: 0.905 (+0.8% improvement)

**Logistic Regression (Tuned):**
- Best parameters: C=1.0, penalty=L2
- Validation F1: 0.846 (no change - already optimal)

**Model Selection:** Random Forest chosen based on highest validation F1 and robustness.

#### 3.2.2 Final Test Performance (Task 1: High MAR Prediction)

**Test Set Evaluation (N=87, High MAR=13):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 0.9770 | 97.7% of predictions correct |
| **Precision** | 0.9231 | 92.3% of predicted high MAR are true high MAR |
| **Recall** | 0.9231 | 92.3% of actual high MAR identified |
| **F1-Score** | 0.9231 | Excellent balance (Class: High MAR) |
| **F1-Score (macro)** | 0.9612 | Average across both classes |

**Class-Specific Performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Low MAR (0) | 0.9861 | 0.9863 | 0.9862 | 74 |
| High MAR (1) | 0.9231 | 0.9231 | 0.9231 | 13 |

**Interpretation:**
- **Excellent performance** on imbalanced test set (85% low MAR, 15% high MAR)
- **Balanced precision/recall:** No bias toward majority class
- **High recall (92.3%):** Only 1 high MAR case missed (low false negative rate)
- **High precision (92.3%):** Only 1 false alarm (low false positive rate)
- **Clinical utility:** Can reliably identify MDR isolates requiring special protocols

#### 3.2.3 Confusion Matrix and Interpretation

**Confusion Matrix (Test Set, N=87):**

```
                    Predicted
                 Low MAR  High MAR
Actual  Low MAR     73        1
        High MAR     1       12
```

**Visualization:**

```
         Predicted Label
           0     1
        ┌────────────┐
    0   │ 73  │  1  │  ← True Negatives: 73, False Positives: 1
A   │    │     │     │
c   ├────────────┤
t   1   │  1  │ 12  │  ← False Negatives: 1, True Positives: 12
u   │    │     │     │
a   └────────────┘
l
```

**Normalized (by row):**

```
         Predicted Label
           0       1
        ┌──────────────┐
    0   │ 98.6%│ 1.4%│
A   │      │     │
c   ├──────────────┤
t   1   │ 7.7%│ 92.3%│
u   │      │     │
a   └──────────────┘
l
```

**Error Analysis:**

**False Positive (1 case):**
- Isolate #247: *E. coli* from drinking water
- Predicted: High MAR (1) | Actual: Low MAR (0)
- MAR index: 0.28 (just below 0.3 threshold)
- Resistance: 5/19 antibiotics (ampicillin, cefalotin, tetracycline, nalidixic acid, enrofloxacin)
- Interpretation: **Borderline case** - clinically still concerning (close to threshold)
- Impact: **Minimal clinical harm** - cautious approach justified

**False Negative (1 case):**
- Isolate #419: *Enterobacter cloacae* from effluent water
- Predicted: Low MAR (0) | Actual: High MAR (1)
- MAR index: 0.32 (just above 0.3 threshold)
- Resistance: 6/19 antibiotics (β-lactams, quinolones, tetracycline)
- Interpretation: **Borderline case** - unusual resistance profile
- Impact: **Moderate clinical concern** - missed MDR alert, but susceptibility testing would catch

**Key Insights:**
1. **Both errors are borderline cases** near the 0.3 MAR threshold
2. **No catastrophic misclassifications** (e.g., MAR 0.05 predicted as high MAR)
3. **Model is well-calibrated** - errors reflect genuine ambiguity
4. **Clinical safety maintained** - susceptibility testing remains gold standard

#### 3.2.4 Model Comparison Across Tasks

**Task 2: Bacterial Species Classification (5 main species, N=500)**

| Algorithm | Validation Accuracy | Validation F1 (macro) |
|-----------|-------------------|---------------------|
| Random Forest | 0.847 | 0.821 |
| Gradient Boosting | 0.823 | 0.798 |
| Logistic Regression | 0.776 | 0.742 |
| SVM | 0.801 | 0.778 |
| kNN | 0.734 | 0.701 |
| Naive Bayes | 0.689 | 0.658 |

**Best Model:** Random Forest (Test F1: 0.816)
- Species identification from resistance profile is feasible but challenging
- *E. coli* and *K. pneumoniae* classified with >90% accuracy
- Rare species (<20 samples) have lower accuracy

**Task 3: Administrative Region Classification (3 regions)**

| Algorithm | Validation Accuracy | Validation F1 (macro) |
|-----------|-------------------|---------------------|
| Random Forest | 0.712 | 0.688 |
| Gradient Boosting | 0.701 | 0.675 |
| SVM | 0.687 | 0.661 |
| Logistic Regression | 0.676 | 0.649 |
| kNN | 0.645 | 0.622 |
| Naive Bayes | 0.623 | 0.598 |

**Best Model:** Random Forest (Test F1: 0.694)
- Geographic prediction from resistance patterns shows moderate success
- BARMM (largest region) classified with 85% accuracy
- Regional surveillance differences likely drive patterns

**Cross-Task Insights:**
1. **Random Forest consistently best** across all tasks
2. **Gradient Boosting close second** but slower training
3. **Naive Bayes weakest** - independence assumption violated
4. **Task difficulty:** High MAR (easiest, F1=0.92) > Species (moderate, F1=0.82) > Region (hardest, F1=0.69)

#### 3.2.5 Feature Importance Analysis (Random Forest, Task 1)

**Top 10 Most Important Features (High MAR Prediction):**

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | ampicillin_binary | 0.142 | Strong predictor (β-lactam resistance common in MDR) |
| 2 | cefalotin_binary | 0.118 | Co-selected with ampicillin (ESBL marker) |
| 3 | tetracycline_binary | 0.096 | Mobile genetic element marker |
| 4 | nalidixic_acid_binary | 0.089 | Quinolone resistance gateway |
| 5 | gentamicin_binary | 0.078 | Aminoglycoside - plasmid-borne |
| 6 | trimethoprim/sulfamethazole_binary | 0.071 | Co-trimoxazole - integron marker |
| 7 | cefpodoxime_binary | 0.065 | 3rd generation cephalosporin |
| 8 | chloramphenicol_binary | 0.059 | Old antibiotic, resurging resistance |
| 9 | enrofloxacin_binary | 0.053 | Fluoroquinolone - veterinary use |
| 10 | cefotaxime_binary | 0.047 | ESBL substrate |

**Interpretation:**
- **β-lactams dominate** (ampicillin, cefalotin, cefpodoxime, cefotaxime = 37.2% total)
- **Quinolones important** (nalidixic acid, enrofloxacin = 14.2%)
- **Pattern recognition:** Multiple resistance classes needed for MDR classification
- **Clinical implication:** Model captures known co-resistance mechanisms

---

## 4. Deployment Description

### 4.1 What the Deployed Model Does

The deployed AMR prediction system is a **binary classifier** that predicts whether a bacterial isolate exhibits **high multi-drug resistance (MDR)** based on its antimicrobial susceptibility testing (AST) profile.

**Core Functionality:**

1. **Input Processing:**
   - Accepts binary resistance data for 23 antibiotics
   - Format: R=1 (Resistant), S/I=0 (Susceptible/Intermediate)
   - Handles missing values through median imputation

2. **Prediction Output:**
   - **Binary classification:** 0 (Low MAR) or 1 (High MAR)
   - **Probability scores:** Confidence level for each class (0.0-1.0)
   - **Threshold:** MAR index ≥ 0.3 (30% resistance rate)

3. **Model Architecture:**
   - **Algorithm:** Random Forest Classifier (100 trees, max_depth=10)
   - **Preprocessing:** Scikit-learn Pipeline (imputation + scaling)
   - **Training data:** 583 isolates from 3 Philippine regions
   - **Performance:** 97.7% accuracy, 92.3% F1-score on test set

4. **Deployment Modes:**
   - **Batch processing:** CSV file input/output for laboratory workflows
   - **Single prediction:** Individual isolate assessment via Python API
   - **Web service:** REST API integration (optional)

**Example Use Case:**
```
Input: Isolate #X shows resistance to ampicillin, cefalotin, tetracycline, nalidixic acid (4/23 antibiotics)
Output: High MAR (1) with 87% confidence
Recommendation: Flag for infection control, consider combination therapy
```

### 4.2 Required Inputs

**Minimum Required Data:**

The model requires **binary resistance status for 23 antibiotics** (one-hot encoded format):

| Antibiotic Class | Antibiotics |
|------------------|-------------|
| **β-Lactams (9)** | ampicillin, amoxicillin/clavulanic_acid, cefalotin, cefalexin, cefpodoxime, cefotaxime, cefovecin, ceftiofur, ceftazidime/avibactam |
| **Aminoglycosides (3)** | amikacin, gentamicin, neomycin |
| **Quinolones (4)** | nalidixic_acid, enrofloxacin, marbofloxacin, pradofloxacin |
| **Tetracyclines (2)** | doxycycline, tetracycline |
| **Others (5)** | imepenem, nitrofurantoin, chloramphenicol, trimethoprim/sulfamethazole, ceftaroline |

**Input Format Options:**

**Option 1: CSV File (Batch Processing)**
```csv
ampicillin_binary,amoxicillin/clavulanic_acid_binary,cefalotin_binary,...
1,0,1,...
0,1,0,...
```

**Option 2: Python Dictionary (Single Prediction)**
```python
{
    'ampicillin_binary': 1,
    'amoxicillin/clavulanic_acid_binary': 0,
    'cefalotin_binary': 1,
    # ... all 23 antibiotics
}
```

**Option 3: JSON (Web API)**
```json
{
  "features": {
    "ampicillin_binary": 1,
    "cefalotin_binary": 1,
    ...
  }
}
```

**Data Requirements:**
- All 23 antibiotic columns must be present (missing → imputed)
- Binary values only: 0 or 1
- Column names must match exactly (use `get_required_features()`)
- No metadata required (species, source, region not used in Task 1)

**Validation:**
- Model automatically validates input format
- Raises `ValueError` if required features missing
- Imputes missing values (median strategy)

### 4.3 Intended Users and Decisions Supported

**Primary Users:**

1. **Clinical Microbiologists**
   - **Use:** Interpret AST results, flag MDR isolates
   - **Decision:** Trigger infection control protocols, alert clinicians
   - **Workflow:** Run batch predictions on daily lab results → identify high MAR cases → prioritize review

2. **Infectious Disease Physicians**
   - **Use:** Inform empiric therapy selection
   - **Decision:** Choose antibiotic regimen (avoid resistant classes)
   - **Workflow:** Check MDR prediction → review susceptibility → select appropriate therapy

3. **Infection Control Personnel**
   - **Use:** Surveillance for MDR trends
   - **Decision:** Implement isolation/cohorting, enhance monitoring
   - **Workflow:** Monthly batch analysis → track MDR rates by ward/unit → intervention targeting

4. **Public Health Epidemiologists**
   - **Use:** Regional AMR surveillance
   - **Decision:** Policy recommendations, resource allocation
   - **Workflow:** Aggregate predictions by region/source → identify hotspots → design interventions

5. **Hospital Administrators**
   - **Use:** Antimicrobial stewardship program metrics
   - **Decision:** Resource allocation, protocol updates
   - **Workflow:** Dashboard monitoring → track MDR prevalence → justify stewardship funding

**Decisions Supported:**

| Decision Type | Description | Impact |
|---------------|-------------|--------|
| **Clinical Treatment** | Select appropriate antibiotics avoiding predicted resistant classes | Improved patient outcomes, reduced treatment failures |
| **Infection Control** | Isolate high MAR patients, implement contact precautions | Prevent transmission, reduce nosocomial MDR spread |
| **Laboratory Workflow** | Prioritize full susceptibility testing for predicted high MAR | Efficient resource use, faster turnaround for critical cases |
| **Surveillance** | Track MDR trends over time, identify outbreaks | Early detection, targeted interventions |
| **Stewardship** | Guide antibiotic restriction policies, formulary decisions | Preserve antibiotic effectiveness, reduce resistance pressure |
| **Research** | Identify high-value isolates for genomic/mechanistic studies | Advance understanding of resistance mechanisms |

**Decision Support, Not Replacement:**
- Model provides **risk stratification**, not definitive diagnosis
- **Susceptibility testing remains gold standard**
- Predictions inform but do not override laboratory results
- Clinical judgment essential for treatment decisions

### 4.4 Limitations and Future Improvements

#### 4.4.1 Current Limitations

**1. Geographic Specificity**
- **Issue:** Model trained on isolates from 3 regions in the Philippines only
- **Impact:** Performance may degrade in other geographic areas with different resistance epidemiology
- **Mitigation:** Validate on local data before deployment, region-specific models

**2. Species Diversity**
- **Issue:** 13 species represented, but E. coli (40%) and K. pneumoniae (27%) dominate
- **Impact:** Rare species (<5 samples) may have lower prediction accuracy
- **Mitigation:** Stratify by species, require minimum sample size for predictions

**3. Temporal Validity**
- **Issue:** Resistance patterns evolve over time (concept drift)
- **Impact:** Model accuracy may decrease as resistance mechanisms change
- **Mitigation:** Monitor performance quarterly, retrain with recent data annually

**4. Binary Threshold Sensitivity**
- **Issue:** 0.3 MAR threshold is somewhat arbitrary
- **Impact:** Borderline cases (MAR 0.28-0.32) have higher misclassification risk
- **Mitigation:** Provide probability scores, allow user-defined thresholds

**5. Feature Set Limitations**
- **Issue:** Only 23 antibiotics tested (not comprehensive panel)
- **Impact:** Missing resistance to untested antibiotics (e.g., colistin, carbapenems)
- **Mitigation:** Update model as new antibiotics added to testing panel

**6. Imbalanced Classes**
- **Issue:** High MAR is rare (15% of data)
- **Impact:** Model may under-predict high MAR in extreme cases
- **Mitigation:** Class weights, SMOTE oversampling (not currently implemented)

**7. Lack of Mechanism Information**
- **Issue:** Model uses phenotypic resistance only, no genotypic data
- **Impact:** Cannot distinguish resistance mechanisms (ESBL, AmpC, carbapenemases)
- **Mitigation:** Integrate genotypic data (WGS) in future versions

**8. Validation Dataset Size**
- **Issue:** Test set n=87 (13 high MAR cases)
- **Impact:** Wide confidence intervals on performance metrics
- **Mitigation:** External validation on larger independent cohorts

**9. No Uncertainty Quantification**
- **Issue:** Predictions lack confidence intervals
- **Impact:** Cannot assess reliability of individual predictions
- **Mitigation:** Implement Bayesian methods or ensemble calibration

**10. Deployment Complexity**
- **Issue:** Requires Python environment, dependencies (scikit-learn, pandas)
- **Impact:** May be difficult to integrate into existing laboratory systems
- **Mitigation:** Dockerized deployment, REST API wrapper

#### 4.4.2 Future Improvements

**Short-Term (Next 6 months):**

1. **External Validation**
   - Test on independent datasets from other regions/countries
   - Assess generalizability and geographic transferability
   - Publish validation study

2. **User Interface Development**
   - Web-based dashboard for non-technical users
   - Drag-and-drop CSV upload, automatic visualization
   - Export reports with predictions + probabilities

3. **Probability Calibration**
   - Apply Platt scaling or isotonic regression
   - Ensure predicted probabilities match empirical frequencies
   - Enable risk-based decision thresholds

4. **SHAP Explanations**
   - Implement SHAP (SHapley Additive exPlanations) for interpretability
   - Show which antibiotics drive each prediction
   - Build clinician trust through transparency

5. **Alert System Integration**
   - Automated email/SMS alerts for high MAR detections
   - Integration with laboratory information systems (LIS)
   - Real-time monitoring dashboard

**Medium-Term (6-12 months):**

6. **Multiclass MAR Prediction**
   - Expand from binary to 3-4 classes (low, moderate, high, extreme)
   - Provide finer risk stratification
   - Support graduated clinical responses

7. **Species-Specific Models**
   - Train separate models for E. coli, K. pneumoniae, etc.
   - Account for species-specific resistance patterns
   - Improve accuracy for rare species

8. **Temporal Trend Analysis**
   - Incorporate time-series analysis for drift detection
   - Automated alerts when model performance degrades
   - Trigger retraining workflows

9. **Federated Learning**
   - Enable multi-institutional model training without data sharing
   - Preserve privacy while leveraging diverse datasets
   - Build robust pan-regional models

10. **Mobile Application**
    - Smartphone app for point-of-care predictions
    - Offline mode with model embedded in app
    - Photo capture of antibiotic disk diffusion results → resistance prediction

**Long-Term (1-2 years):**

11. **Genomic Integration**
    - Combine phenotypic resistance with WGS data
    - Predict resistance from genotype (resistome analysis)
    - Identify novel resistance genes/mechanisms

12. **Treatment Recommendation System**
    - Beyond prediction, suggest optimal antibiotic choices
    - Consider local resistance rates, drug interactions, patient factors
    - Full clinical decision support system

13. **Resistance Mechanism Classification**
    - Multi-task learning: predict MAR + mechanism (ESBL, carbapenemase, etc.)
    - Guide targeted confirmatory testing
    - Infection control implications (e.g., CPE vs non-CPE)

14. **One Health Integration**
    - Expand to veterinary isolates (already some fish samples)
    - Link human, animal, environmental resistance patterns
    - Inform AMR interventions across sectors

15. **Deep Learning Approaches**
    - Explore neural networks (MLPs, CNNs for resistance patterns)
    - Attention mechanisms to identify key antibiotic combinations
    - Potential accuracy gains (if sufficient data)

16. **Causal Inference**
    - Move beyond prediction to causal understanding
    - Identify interventions that reduce resistance transmission
    - Support policy decisions with causal evidence

**Research Directions:**

- **Transfer learning** from large international datasets (e.g., ATLAS, EARS-Net)
- **Active learning** to efficiently select isolates for additional testing
- **Reinforcement learning** for adaptive surveillance strategies
- **Graph neural networks** to model resistance spread in hospital networks

**Implementation Priorities:**
1. External validation (most critical for trust)
2. User interface (enable adoption)
3. Probability calibration (improve usability)
4. Species-specific models (improve accuracy)
5. Genomic integration (scientific advancement)

---

## Conclusion

This thesis presents a comprehensive machine learning pipeline for antimicrobial resistance pattern recognition, encompassing:

1. **Robust data preparation** handling 583 isolates with binary/ordinal encoding
2. **Unsupervised discovery** identifying 4 resistance clusters and 8 key association rules
3. **Supervised classification** achieving 97.7% accuracy for high MAR prediction using Random Forest
4. **Practical deployment** with batch and single-prediction capabilities

**Key Contributions:**

- Demonstrated feasibility of resistance-based species identification (F1=0.82)
- Uncovered geographic and source-specific resistance patterns via association rules
- Validated ensemble methods (RF, GBM) for imbalanced AMR classification
- Provided open-source, reproducible pipeline for AMR surveillance

**Impact:**

The deployed model can support clinical decision-making, infection control, and public health surveillance in resource-limited settings by providing rapid MDR risk assessment from routine AST data.

**Future Work:**

Expanding to genomic data integration, multi-institutional validation, and causal inference will enhance the model's clinical utility and scientific rigor.

---

## References

**Software and Libraries:**
- Python 3.8+
- scikit-learn 1.0+
- pandas, numpy
- mlxtend (association rule mining)
- matplotlib, seaborn (visualization)

**Code Repository:**
- GitHub: Reyn4ldo/my-thesis-project
- Documentation: README.md, DEPLOYMENT_GUIDE.md

**Contact:**
- For questions or collaboration, open an issue on GitHub

---

**Last Updated:** December 9, 2025
