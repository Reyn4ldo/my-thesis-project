"""
Phase 2: Unsupervised Pattern Recognition Module for AMR Analysis

This module implements unsupervised learning methods for antimicrobial resistance analysis:
- Clustering (k-means, hierarchical, DBSCAN)
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Association rule mining (Apriori, FP-Growth)

Author: Thesis Project
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Import optional dependencies with error handling
try:
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    warnings.warn("mlxtend not installed. Association rule mining will not be available. "
                 "Install with: pip install mlxtend")


class UnsupervisedAMRAnalysis:
    """
    Comprehensive unsupervised analysis class for AMR data.
    
    This class handles:
    - Clustering analysis (k-means, hierarchical, DBSCAN)
    - Dimensionality reduction (PCA, t-SNE, UMAP)
    - Association rule mining (Apriori, FP-Growth)
    """
    
    # Class constants for column patterns
    BINARY_PATTERN = '_binary'
    ORDINAL_PATTERN = '_ordinal'
    MIC_PATTERN = '_mic'
    
    # Default visualization limits
    DEFAULT_MAX_RESISTANCE_COLS = 20
    
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], metadata_cols: Optional[List[str]] = None):
        """
        Initialize unsupervised analysis.
        
        Args:
            df (pd.DataFrame): Prepared dataframe with features
            feature_cols (List[str]): Column names to use as features for analysis
            metadata_cols (List[str], optional): Metadata columns to keep for labeling
        """
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.metadata_cols = metadata_cols or []
        
        # Extract feature matrix
        self.X = df[feature_cols].copy()
        
        # Store results
        self.clustering_results = {}
        self.dimensionality_results = {}
        self.rule_mining_results = {}
        
    def _prepare_features(self, standardize: bool = True) -> np.ndarray:
        """
        Prepare feature matrix for analysis.
        
        Args:
            standardize (bool): Whether to standardize features
            
        Returns:
            np.ndarray: Prepared feature matrix
        """
        X_prepared = self.X.fillna(0).values
        
        if standardize:
            scaler = StandardScaler()
            X_prepared = scaler.fit_transform(X_prepared)
        
        return X_prepared
    
    # ==================== 2.1 CLUSTERING METHODS ====================
    
    def kmeans_clustering(self, k_range: Tuple[int, int] = (2, 10), 
                         standardize: bool = True,
                         random_state: int = 42) -> Dict:
        """
        Perform k-means clustering with multiple k values.
        
        Uses elbow method and silhouette scores to determine optimal k.
        
        Args:
            k_range (Tuple[int, int]): Range of k values to try (min, max)
            standardize (bool): Whether to standardize features
            random_state (int): Random seed for reproducibility
            
        Returns:
            Dict: Results containing inertias, silhouette scores, and cluster labels
        """
        X_prepared = self._prepare_features(standardize=standardize)
        
        results = {
            'k_values': list(range(k_range[0], k_range[1] + 1)),
            'inertias': [],
            'silhouette_scores': [],
            'models': {},
            'labels': {}
        }
        
        print(f"Running k-means for k={k_range[0]} to k={k_range[1]}...")
        
        for k in results['k_values']:
            # Fit k-means
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(X_prepared)
            
            # Store results
            results['inertias'].append(kmeans.inertia_)
            results['models'][k] = kmeans
            results['labels'][k] = labels
            
            # Calculate silhouette score
            if k > 1:
                sil_score = silhouette_score(X_prepared, labels)
                results['silhouette_scores'].append(sil_score)
                print(f"  k={k}: inertia={kmeans.inertia_:.2f}, silhouette={sil_score:.3f}")
            else:
                results['silhouette_scores'].append(None)
        
        # Store in object
        self.clustering_results['kmeans'] = results
        
        return results
    
    def plot_kmeans_evaluation(self, figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
        """
        Plot elbow curve and silhouette scores for k-means.
        
        Args:
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if 'kmeans' not in self.clustering_results:
            raise ValueError("Run kmeans_clustering() first")
        
        results = self.clustering_results['kmeans']
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Elbow plot
        axes[0].plot(results['k_values'], results['inertias'], 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of clusters (k)', fontsize=12)
        axes[0].set_ylabel('Inertia', fontsize=12)
        axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Silhouette plot
        k_vals_for_sil = [k for k in results['k_values'] if k > 1]
        sil_vals = [s for s in results['silhouette_scores'] if s is not None]
        axes[1].plot(k_vals_for_sil, sil_vals, 'go-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of clusters (k)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def hierarchical_clustering(self, n_clusters: int = 5, 
                               linkage_method: str = 'ward',
                               standardize: bool = True) -> Dict:
        """
        Perform hierarchical clustering with Ward linkage.
        
        Args:
            n_clusters (int): Number of clusters to extract
            linkage_method (str): Linkage method ('ward', 'complete', 'average')
            standardize (bool): Whether to standardize features
            
        Returns:
            Dict: Results containing linkage matrix and cluster labels
        """
        X_prepared = self._prepare_features(standardize=standardize)
        
        print(f"Running hierarchical clustering with {linkage_method} linkage...")
        
        # Compute linkage matrix
        linkage_matrix = linkage(X_prepared, method=linkage_method)
        
        # Get cluster labels
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = model.fit_predict(X_prepared)
        
        # Calculate silhouette score
        sil_score = silhouette_score(X_prepared, labels)
        
        results = {
            'linkage_matrix': linkage_matrix,
            'n_clusters': n_clusters,
            'labels': labels,
            'silhouette_score': sil_score,
            'model': model
        }
        
        print(f"  Hierarchical clustering: {n_clusters} clusters, silhouette={sil_score:.3f}")
        
        self.clustering_results['hierarchical'] = results
        
        return results
    
    def plot_dendrogram(self, max_display_levels: Optional[int] = None,
                       figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
        """
        Plot dendrogram for hierarchical clustering.
        
        Args:
            max_display_levels (int, optional): Maximum number of levels to display
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if 'hierarchical' not in self.clustering_results:
            raise ValueError("Run hierarchical_clustering() first")
        
        linkage_matrix = self.clustering_results['hierarchical']['linkage_matrix']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        dendrogram(
            linkage_matrix,
            ax=ax,
            truncate_mode='level' if max_display_levels else None,
            p=max_display_levels,
            leaf_font_size=10,
            color_threshold=0
        )
        
        ax.set_xlabel('Sample Index (or Cluster Size)', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        ax.set_title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def dbscan_clustering(self, eps: float = 0.5, min_samples: int = 5,
                         standardize: bool = True) -> Dict:
        """
        Perform DBSCAN clustering.
        
        Args:
            eps (float): Maximum distance between samples
            min_samples (int): Minimum samples in a neighborhood
            standardize (bool): Whether to standardize features
            
        Returns:
            Dict: Results containing cluster labels and statistics
        """
        X_prepared = self._prepare_features(standardize=standardize)
        
        print(f"Running DBSCAN with eps={eps}, min_samples={min_samples}...")
        
        # Fit DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_prepared)
        
        # Calculate statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        results = {
            'eps': eps,
            'min_samples': min_samples,
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'model': dbscan
        }
        
        # Calculate silhouette score (excluding noise points)
        if n_clusters > 1:
            mask = labels != -1
            if mask.sum() > 0:
                sil_score = silhouette_score(X_prepared[mask], labels[mask])
                results['silhouette_score'] = sil_score
                print(f"  DBSCAN: {n_clusters} clusters, {n_noise} noise points, silhouette={sil_score:.3f}")
            else:
                results['silhouette_score'] = None
                print(f"  DBSCAN: {n_clusters} clusters, {n_noise} noise points")
        else:
            results['silhouette_score'] = None
            print(f"  DBSCAN: {n_clusters} clusters, {n_noise} noise points")
        
        self.clustering_results['dbscan'] = results
        
        return results
    
    def plot_k_distance(self, k: int = 5, figsize: Tuple[int, int] = (10, 5)) -> plt.Figure:
        """
        Plot k-distance graph for DBSCAN parameter selection.
        
        Args:
            k (int): Number of nearest neighbors
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        X_prepared = self._prepare_features(standardize=True)
        
        # Calculate k-nearest neighbor distances
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_prepared)
        distances, indices = nbrs.kneighbors(X_prepared)
        
        # Sort distances
        k_distances = np.sort(distances[:, k-1], axis=0)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(k_distances, linewidth=2)
        ax.set_xlabel('Data Points sorted by distance', fontsize=12)
        ax.set_ylabel(f'{k}-NN Distance', fontsize=12)
        ax.set_title(f'k-Distance Graph (k={k}) for DBSCAN eps Selection', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def analyze_clusters(self, cluster_labels: np.ndarray,
                        method_name: str = 'clustering') -> pd.DataFrame:
        """
        Analyze cluster characteristics.
        
        For each cluster, compute:
        - Proportion resistant for each antibiotic
        - Mean/median MAR index
        - Distribution of species, sources, regions, ESBL
        
        Args:
            cluster_labels (np.ndarray): Cluster assignments
            method_name (str): Name of clustering method
            
        Returns:
            pd.DataFrame: Cluster analysis results
        """
        # Add cluster labels to dataframe
        df_clustered = self.df.copy()
        df_clustered['cluster'] = cluster_labels
        
        # Identify antibiotic resistance columns using class constants
        resistance_cols = [col for col in df_clustered.columns 
                          if self.BINARY_PATTERN in col or self.ORDINAL_PATTERN in col]
        
        # Identify metadata columns
        metadata_cols = ['bacterial_species', 'sample_source', 
                        'administrative_region', 'esbl']
        metadata_cols = [col for col in metadata_cols if col in df_clustered.columns]
        
        # Get unique clusters (exclude noise if present)
        clusters = sorted([c for c in df_clustered['cluster'].unique() if c != -1])
        
        analysis_results = []
        
        for cluster_id in clusters:
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            
            result = {
                'cluster': cluster_id,
                'size': len(cluster_data)
            }
            
            # Resistance proportions
            for col in resistance_cols[:10]:  # Limit to first 10 for summary
                if col in cluster_data.columns:
                    result[f'{col}_mean'] = cluster_data[col].mean()
            
            # MAR index statistics
            if 'mar_index' in cluster_data.columns:
                result['mar_index_mean'] = cluster_data['mar_index'].mean()
                result['mar_index_median'] = cluster_data['mar_index'].median()
                result['mar_index_std'] = cluster_data['mar_index'].std()
            
            # Scored resistance
            if 'scored_resistance' in cluster_data.columns:
                result['scored_resistance_mean'] = cluster_data['scored_resistance'].mean()
            
            # Most common metadata values
            for col in metadata_cols:
                if col in cluster_data.columns:
                    most_common = cluster_data[col].mode()
                    if len(most_common) > 0:
                        result[f'{col}_most_common'] = most_common.iloc[0]
            
            analysis_results.append(result)
        
        analysis_df = pd.DataFrame(analysis_results)
        
        print(f"\n{method_name} Cluster Analysis:")
        print(f"  Total clusters: {len(clusters)}")
        print(f"  Cluster sizes: {analysis_df['size'].tolist()}")
        
        return analysis_df
    
    def plot_cluster_heatmap(self, cluster_labels: np.ndarray,
                            resistance_cols: Optional[List[str]] = None,
                            max_cols: Optional[int] = None,
                            figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot heatmap of resistance patterns sorted by cluster.
        
        Args:
            cluster_labels (np.ndarray): Cluster assignments
            resistance_cols (List[str], optional): Resistance columns to plot
            max_cols (int, optional): Maximum columns to display (default: DEFAULT_MAX_RESISTANCE_COLS)
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        df_clustered = self.df.copy()
        df_clustered['cluster'] = cluster_labels
        
        # Get resistance columns
        if resistance_cols is None:
            resistance_cols = [col for col in df_clustered.columns 
                             if self.BINARY_PATTERN in col or self.ORDINAL_PATTERN in col]
            # Limit to reasonable number for visualization
            if max_cols is None:
                max_cols = self.DEFAULT_MAX_RESISTANCE_COLS
            resistance_cols = resistance_cols[:max_cols]
        
        # Sort by cluster
        df_sorted = df_clustered.sort_values('cluster')
        
        # Create heatmap data
        heatmap_data = df_sorted[resistance_cols].fillna(0)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(heatmap_data, cmap='RdYlGn_r', cbar_kws={'label': 'Resistance Level'},
                   yticklabels=False, ax=ax)
        
        ax.set_xlabel('Antibiotics', fontsize=12)
        ax.set_ylabel('Isolates (sorted by cluster)', fontsize=12)
        ax.set_title('Resistance Heatmap by Cluster', fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def plot_cluster_composition(self, cluster_labels: np.ndarray,
                                metadata_col: str = 'bacterial_species',
                                figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot bar chart of cluster composition by metadata.
        
        Args:
            cluster_labels (np.ndarray): Cluster assignments
            metadata_col (str): Metadata column to analyze
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        df_clustered = self.df.copy()
        df_clustered['cluster'] = cluster_labels
        
        if metadata_col not in df_clustered.columns:
            raise ValueError(f"Column {metadata_col} not found in dataframe")
        
        # Create cross-tabulation
        ct = pd.crosstab(df_clustered['cluster'], df_clustered[metadata_col], 
                        normalize='index')
        
        fig, ax = plt.subplots(figsize=figsize)
        ct.plot(kind='bar', stacked=True, ax=ax, colormap='Set3')
        
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Proportion', fontsize=12)
        ax.set_title(f'Cluster Composition by {metadata_col}', 
                    fontsize=14, fontweight='bold')
        ax.legend(title=metadata_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.xticks(rotation=0)
        plt.tight_layout()
        return fig
    
    # ==================== 2.2 DIMENSIONALITY REDUCTION ====================
    
    def pca_analysis(self, n_components: int = 3, standardize: bool = True) -> Dict:
        """
        Perform PCA dimensionality reduction.
        
        Args:
            n_components (int): Number of principal components
            standardize (bool): Whether to standardize features
            
        Returns:
            Dict: PCA results including components and explained variance
        """
        X_prepared = self._prepare_features(standardize=standardize)
        
        print(f"Running PCA with {n_components} components...")
        
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X_prepared)
        
        results = {
            'model': pca,
            'components': components,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
        }
        
        print(f"  Explained variance: {pca.explained_variance_ratio_}")
        print(f"  Cumulative variance: {results['cumulative_variance']}")
        
        # Add to dataframe
        for i in range(n_components):
            self.df[f'PC{i+1}'] = components[:, i]
        
        self.dimensionality_results['pca'] = results
        
        return results
    
    def tsne_analysis(self, n_components: int = 2, perplexity: float = 30.0,
                     standardize: bool = True, random_state: int = 42) -> Dict:
        """
        Perform t-SNE dimensionality reduction.
        
        Args:
            n_components (int): Number of dimensions (usually 2)
            perplexity (float): Perplexity parameter
            standardize (bool): Whether to standardize features
            random_state (int): Random seed
            
        Returns:
            Dict: t-SNE results
        """
        X_prepared = self._prepare_features(standardize=standardize)
        
        print(f"Running t-SNE with perplexity={perplexity}...")
        
        tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                   random_state=random_state)
        components = tsne.fit_transform(X_prepared)
        
        results = {
            'model': tsne,
            'components': components
        }
        
        # Add to dataframe
        for i in range(n_components):
            self.df[f'TSNE{i+1}'] = components[:, i]
        
        self.dimensionality_results['tsne'] = results
        
        print(f"  t-SNE embedding complete")
        
        return results
    
    def umap_analysis(self, n_components: int = 2, n_neighbors: int = 15,
                     min_dist: float = 0.1, standardize: bool = True,
                     random_state: int = 42) -> Dict:
        """
        Perform UMAP dimensionality reduction.
        
        Args:
            n_components (int): Number of dimensions (usually 2)
            n_neighbors (int): Number of neighbors
            min_dist (float): Minimum distance
            standardize (bool): Whether to standardize features
            random_state (int): Random seed
            
        Returns:
            Dict: UMAP results
        """
        try:
            import umap
        except ImportError:
            raise ImportError("UMAP requires 'umap-learn' package. Install with: pip install umap-learn")
        
        X_prepared = self._prepare_features(standardize=standardize)
        
        print(f"Running UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}...")
        
        umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                              min_dist=min_dist, random_state=random_state)
        components = umap_model.fit_transform(X_prepared)
        
        results = {
            'model': umap_model,
            'components': components
        }
        
        # Add to dataframe
        for i in range(n_components):
            self.df[f'UMAP{i+1}'] = components[:, i]
        
        self.dimensionality_results['umap'] = results
        
        print(f"  UMAP embedding complete")
        
        return results
    
    def plot_dimred_scatter(self, method: str = 'pca', color_by: str = 'bacterial_species',
                           figsize: Tuple[int, int] = (10, 8),
                           alpha: float = 0.6) -> plt.Figure:
        """
        Plot scatter plot of dimensionality reduction results.
        
        Args:
            method (str): Method to plot ('pca', 'tsne', 'umap')
            color_by (str): Column to color points by
            figsize (Tuple[int, int]): Figure size
            alpha (float): Point transparency
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        method = method.lower()
        
        # Determine component columns
        if method == 'pca':
            x_col, y_col = 'PC1', 'PC2'
        elif method == 'tsne':
            x_col, y_col = 'TSNE1', 'TSNE2'
        elif method == 'umap':
            x_col, y_col = 'UMAP1', 'UMAP2'
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if x_col not in self.df.columns or y_col not in self.df.columns:
            raise ValueError(f"Run {method}_analysis() first")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Handle different color types
        if color_by in self.df.columns:
            if self.df[color_by].dtype in ['object', 'category']:
                # Categorical coloring
                categories = self.df[color_by].unique()
                colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
                
                for i, cat in enumerate(categories):
                    mask = self.df[color_by] == cat
                    ax.scatter(self.df.loc[mask, x_col], self.df.loc[mask, y_col],
                             label=str(cat)[:30], alpha=alpha, s=50, c=[colors[i]])
                
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            else:
                # Numeric coloring
                scatter = ax.scatter(self.df[x_col], self.df[y_col], 
                                   c=self.df[color_by], cmap='viridis',
                                   alpha=alpha, s=50)
                plt.colorbar(scatter, ax=ax, label=color_by)
        else:
            ax.scatter(self.df[x_col], self.df[y_col], alpha=alpha, s=50)
        
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.set_title(f'{method.upper()} - Colored by {color_by}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    # ==================== 2.3 ASSOCIATION RULE MINING ====================
    
    def prepare_transactions(self, resistance_threshold: float = 0.5,
                           include_metadata: bool = True,
                           mar_threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Transform data into transaction format for association rule mining.
        
        Each isolate becomes a transaction with items like:
        - R_ampicillin
        - species=escherichia_coli
        - source=drinking_water
        - MAR_high
        
        Args:
            resistance_threshold (float): Threshold for considering resistance (for ordinal)
            include_metadata (bool): Include metadata as items
            mar_threshold (float, optional): Threshold for MAR_high label
            
        Returns:
            pd.DataFrame: Transaction dataframe (binary format)
        """
        transactions = pd.DataFrame(index=self.df.index)
        
        # Add resistance items using class constants
        resistance_cols = [col for col in self.df.columns 
                          if self.BINARY_PATTERN in col or self.ORDINAL_PATTERN in col]
        
        for col in resistance_cols:
            # Create item name
            antibiotic = col.replace(self.BINARY_PATTERN, '').replace(self.ORDINAL_PATTERN, '')
            item_name = f'R_{antibiotic}'
            
            # Binary: already 0/1
            if self.BINARY_PATTERN in col:
                transactions[item_name] = self.df[col].fillna(0).astype(int)
            # Ordinal: threshold to binary
            elif self.ORDINAL_PATTERN in col:
                transactions[item_name] = (self.df[col].fillna(0) >= 
                                          resistance_threshold * 2).astype(int)
        
        # Add metadata items
        if include_metadata:
            metadata_cols = ['bacterial_species', 'sample_source', 
                           'administrative_region', 'esbl']
            
            for col in metadata_cols:
                if col in self.df.columns:
                    # One-hot encode
                    for value in self.df[col].dropna().unique():
                        if pd.notna(value) and str(value).lower() not in ['nan', 'none', '']:
                            item_name = f'{col}={str(value).lower()}'
                            transactions[item_name] = (self.df[col] == value).astype(int)
        
        # Add MAR category
        if mar_threshold is not None and 'mar_index' in self.df.columns:
            transactions['MAR_high'] = (self.df['mar_index'] >= mar_threshold).astype(int)
        
        return transactions
    
    def apriori_mining(self, transactions: pd.DataFrame,
                      min_support: float = 0.05,
                      min_confidence: float = 0.6,
                      min_lift: float = 1.0,
                      max_len: Optional[int] = None) -> pd.DataFrame:
        """
        Mine association rules using Apriori algorithm.
        
        Args:
            transactions (pd.DataFrame): Transaction dataframe
            min_support (float): Minimum support threshold
            min_confidence (float): Minimum confidence threshold
            min_lift (float): Minimum lift threshold
            max_len (int, optional): Maximum rule length
            
        Returns:
            pd.DataFrame: Association rules
        """
        if not MLXTEND_AVAILABLE:
            raise ImportError("mlxtend is required for association rule mining. "
                            "Install with: pip install mlxtend")
        
        print(f"Running Apriori with min_support={min_support}...")
        
        # Find frequent itemsets
        frequent_itemsets = apriori(transactions, min_support=min_support, 
                                    use_colnames=True, max_len=max_len)
        
        if len(frequent_itemsets) == 0:
            print("  No frequent itemsets found. Try lowering min_support.")
            return pd.DataFrame()
        
        print(f"  Found {len(frequent_itemsets)} frequent itemsets")
        
        # Generate rules
        rules = association_rules(frequent_itemsets, metric="confidence", 
                                 min_threshold=min_confidence)
        
        if len(rules) == 0:
            print("  No rules found. Try lowering min_confidence.")
            return pd.DataFrame()
        
        # Filter by lift
        rules = rules[rules['lift'] >= min_lift]
        
        print(f"  Generated {len(rules)} rules")
        
        # Sort by lift
        rules = rules.sort_values('lift', ascending=False)
        
        self.rule_mining_results['apriori'] = {
            'frequent_itemsets': frequent_itemsets,
            'rules': rules
        }
        
        return rules
    
    def fpgrowth_mining(self, transactions: pd.DataFrame,
                       min_support: float = 0.05,
                       min_confidence: float = 0.6,
                       min_lift: float = 1.0,
                       max_len: Optional[int] = None) -> pd.DataFrame:
        """
        Mine association rules using FP-Growth algorithm.
        
        Args:
            transactions (pd.DataFrame): Transaction dataframe
            min_support (float): Minimum support threshold
            min_confidence (float): Minimum confidence threshold
            min_lift (float): Minimum lift threshold
            max_len (int, optional): Maximum rule length
            
        Returns:
            pd.DataFrame: Association rules
        """
        if not MLXTEND_AVAILABLE:
            raise ImportError("mlxtend is required for association rule mining. "
                            "Install with: pip install mlxtend")
        
        print(f"Running FP-Growth with min_support={min_support}...")
        
        # Find frequent itemsets
        frequent_itemsets = fpgrowth(transactions, min_support=min_support,
                                    use_colnames=True, max_len=max_len)
        
        if len(frequent_itemsets) == 0:
            print("  No frequent itemsets found. Try lowering min_support.")
            return pd.DataFrame()
        
        print(f"  Found {len(frequent_itemsets)} frequent itemsets")
        
        # Generate rules
        rules = association_rules(frequent_itemsets, metric="confidence",
                                 min_threshold=min_confidence)
        
        if len(rules) == 0:
            print("  No rules found. Try lowering min_confidence.")
            return pd.DataFrame()
        
        # Filter by lift
        rules = rules[rules['lift'] >= min_lift]
        
        print(f"  Generated {len(rules)} rules")
        
        # Sort by lift
        rules = rules.sort_values('lift', ascending=False)
        
        self.rule_mining_results['fpgrowth'] = {
            'frequent_itemsets': frequent_itemsets,
            'rules': rules
        }
        
        return rules
    
    def interpret_rules(self, rules: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Interpret and format association rules for reporting.
        
        Args:
            rules (pd.DataFrame): Association rules dataframe
            top_n (int): Number of top rules to display
            
        Returns:
            pd.DataFrame: Formatted rules
        """
        if len(rules) == 0:
            print("No rules to interpret")
            return pd.DataFrame()
        
        # Select top rules
        top_rules = rules.head(top_n).copy()
        
        # Format rules as readable strings
        top_rules['rule'] = top_rules.apply(
            lambda row: f"{set(row['antecedents'])} => {set(row['consequents'])}", 
            axis=1
        )
        
        # Select key columns
        formatted = top_rules[['rule', 'support', 'confidence', 'lift']].copy()
        formatted = formatted.round(3)
        
        print(f"\nTop {min(top_n, len(rules))} Association Rules (by lift):")
        print("=" * 80)
        for idx, row in formatted.iterrows():
            print(f"\n{row['rule']}")
            print(f"  Support: {row['support']:.3f}, Confidence: {row['confidence']:.3f}, Lift: {row['lift']:.3f}")
        
        return formatted


# ==================== CONVENIENCE FUNCTIONS ====================

def quick_clustering_analysis(df: pd.DataFrame, feature_cols: List[str],
                              methods: List[str] = ['kmeans', 'hierarchical', 'dbscan'],
                              k: int = 5) -> Dict:
    """
    Quick clustering analysis with default parameters.
    
    Args:
        df (pd.DataFrame): Prepared dataframe
        feature_cols (List[str]): Feature columns
        methods (List[str]): Clustering methods to run
        k (int): Number of clusters for k-means and hierarchical
        
    Returns:
        Dict: Results from all methods
    """
    analyzer = UnsupervisedAMRAnalysis(df, feature_cols)
    results = {}
    
    if 'kmeans' in methods:
        results['kmeans'] = analyzer.kmeans_clustering(k_range=(2, 10))
    
    if 'hierarchical' in methods:
        results['hierarchical'] = analyzer.hierarchical_clustering(n_clusters=k)
    
    if 'dbscan' in methods:
        results['dbscan'] = analyzer.dbscan_clustering(eps=0.5, min_samples=5)
    
    return results


def quick_dimred_analysis(df: pd.DataFrame, feature_cols: List[str],
                          methods: List[str] = ['pca', 'tsne']) -> Dict:
    """
    Quick dimensionality reduction analysis.
    
    Args:
        df (pd.DataFrame): Prepared dataframe
        feature_cols (List[str]): Feature columns
        methods (List[str]): Methods to run ('pca', 'tsne', 'umap')
        
    Returns:
        Dict: Results from all methods
    """
    analyzer = UnsupervisedAMRAnalysis(df, feature_cols)
    results = {}
    
    if 'pca' in methods:
        results['pca'] = analyzer.pca_analysis(n_components=3)
    
    if 'tsne' in methods:
        results['tsne'] = analyzer.tsne_analysis(n_components=2)
    
    if 'umap' in methods:
        try:
            results['umap'] = analyzer.umap_analysis(n_components=2)
        except ImportError:
            print("UMAP requires 'umap-learn' package. Skipping UMAP analysis.")
    
    return results


if __name__ == "__main__":
    print("="*80)
    print("Unsupervised Pattern Recognition Module")
    print("="*80)
    print("\nThis module provides unsupervised learning methods for AMR analysis:")
    print("  1. Clustering (k-means, hierarchical, DBSCAN)")
    print("  2. Dimensionality reduction (PCA, t-SNE, UMAP)")
    print("  3. Association rule mining (Apriori, FP-Growth)")
    print("\nSee examples_unsupervised.py for usage examples.")
