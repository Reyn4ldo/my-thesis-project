# Phase 1: Data Preparation for AMR Analysis

This module provides comprehensive data preparation functionality for Antimicrobial Resistance (AMR) analysis, implementing all requirements from Phase 1 of the analysis pipeline.

## Overview

The data preparation module handles:
1. Data ingestion and inspection
2. Cleaning S/I/R interpretations
3. Binary and ordinal resistance encoding
4. Categorical feature encoding
5. Missing value handling
6. Feature scaling

## Installation

### Requirements

```bash
pip install pandas numpy scikit-learn
```

## Quick Start

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

## License

This module is part of the thesis project for AMR analysis.

## Author

Thesis Project - December 2025
