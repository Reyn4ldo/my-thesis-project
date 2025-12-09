"""
Example Usage Scripts for Unsupervised Pattern Recognition

This script demonstrates various use cases for the unsupervised analysis module,
including clustering, dimensionality reduction, and association rule mining.

Author: Thesis Project
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_preparation import AMRDataPreparation
from unsupervised_analysis import UnsupervisedAMRAnalysis, quick_clustering_analysis, quick_dimred_analysis


def example_1_kmeans_clustering():
    """
    Example 1: K-means clustering with elbow method and silhouette analysis
    """
    print("="*80)
    print("EXAMPLE 1: K-means Clustering")
    print("="*80)
    
    # Prepare data
    prep = AMRDataPreparation('rawdata.csv')
    df = prep.prepare_data(
        include_binary=False,
        include_ordinal=True,  # Use ordinal for clustering
        include_onehot=False,
        scale=True
    )
    
    # Get feature groups
    groups = prep.get_feature_groups()
    feature_cols = groups['ordinal_resistance'] + groups['amr_indices']
    
    print(f"\n✓ Using {len(feature_cols)} features for clustering")
    
    # Initialize analyzer
    analyzer = UnsupervisedAMRAnalysis(df, feature_cols, 
                                      metadata_cols=['bacterial_species', 'sample_source'])
    
    # Run k-means for k=2 to 10
    results = analyzer.kmeans_clustering(k_range=(2, 10))
    
    # Plot evaluation metrics
    fig = analyzer.plot_kmeans_evaluation(figsize=(12, 4))
    plt.savefig('kmeans_evaluation.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: kmeans_evaluation.png")
    plt.close()
    
    # Analyze clusters for optimal k (e.g., k=5)
    optimal_k = 5
    cluster_labels = results['labels'][optimal_k]
    analysis_df = analyzer.analyze_clusters(cluster_labels, method_name='K-means')
    
    print(f"\n✓ Cluster analysis for k={optimal_k}:")
    print(analysis_df[['cluster', 'size', 'mar_index_mean']].to_string())
    
    return analyzer, results


def example_2_hierarchical_clustering():
    """
    Example 2: Hierarchical clustering with dendrogram
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Hierarchical Clustering")
    print("="*80)
    
    # Prepare data
    prep = AMRDataPreparation('rawdata.csv')
    df = prep.prepare_data(
        include_binary=False,
        include_ordinal=True,
        include_onehot=False,
        scale=True
    )
    
    groups = prep.get_feature_groups()
    feature_cols = groups['ordinal_resistance'] + groups['amr_indices']
    
    # Initialize analyzer
    analyzer = UnsupervisedAMRAnalysis(df, feature_cols,
                                      metadata_cols=['bacterial_species', 'sample_source', 
                                                    'administrative_region'])
    
    # Run hierarchical clustering
    results = analyzer.hierarchical_clustering(n_clusters=5, linkage_method='ward')
    
    # Plot dendrogram
    fig = analyzer.plot_dendrogram(max_display_levels=10, figsize=(14, 6))
    plt.savefig('hierarchical_dendrogram.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: hierarchical_dendrogram.png")
    plt.close()
    
    # Analyze clusters
    cluster_labels = results['labels']
    analysis_df = analyzer.analyze_clusters(cluster_labels, method_name='Hierarchical')
    
    # Plot cluster heatmap
    fig = analyzer.plot_cluster_heatmap(cluster_labels, figsize=(12, 8))
    plt.savefig('hierarchical_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: hierarchical_heatmap.png")
    plt.close()
    
    return analyzer, results


def example_3_dbscan_clustering():
    """
    Example 3: DBSCAN clustering with k-distance plot
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: DBSCAN Clustering")
    print("="*80)
    
    # Prepare data
    prep = AMRDataPreparation('rawdata.csv')
    df = prep.prepare_data(
        include_binary=False,
        include_ordinal=True,
        include_onehot=False,
        scale=True
    )
    
    groups = prep.get_feature_groups()
    feature_cols = groups['ordinal_resistance'] + groups['amr_indices']
    
    # Initialize analyzer
    analyzer = UnsupervisedAMRAnalysis(df, feature_cols,
                                      metadata_cols=['bacterial_species', 'sample_source'])
    
    # Plot k-distance for eps selection
    fig = analyzer.plot_k_distance(k=5, figsize=(10, 5))
    plt.savefig('dbscan_k_distance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: dbscan_k_distance.png")
    plt.close()
    
    # Run DBSCAN with selected parameters
    results = analyzer.dbscan_clustering(eps=2.0, min_samples=5)
    
    print(f"\n✓ DBSCAN Results:")
    print(f"  Number of clusters: {results['n_clusters']}")
    print(f"  Number of noise points: {results['n_noise']}")
    
    # Analyze clusters (excluding noise)
    if results['n_clusters'] > 0:
        cluster_labels = results['labels']
        analysis_df = analyzer.analyze_clusters(cluster_labels, method_name='DBSCAN')
        
        # Plot composition by species
        fig = analyzer.plot_cluster_composition(cluster_labels, 
                                               metadata_col='bacterial_species',
                                               figsize=(12, 6))
        plt.savefig('dbscan_composition.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: dbscan_composition.png")
        plt.close()
    
    return analyzer, results


def example_4_pca_analysis():
    """
    Example 4: PCA dimensionality reduction
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: PCA Analysis")
    print("="*80)
    
    # Prepare data
    prep = AMRDataPreparation('rawdata.csv')
    df = prep.prepare_data(
        include_binary=False,
        include_ordinal=True,
        include_onehot=False,
        scale=True
    )
    
    groups = prep.get_feature_groups()
    feature_cols = groups['ordinal_resistance'] + groups['amr_indices']
    
    # Initialize analyzer
    analyzer = UnsupervisedAMRAnalysis(df, feature_cols,
                                      metadata_cols=['bacterial_species', 'sample_source',
                                                    'administrative_region', 'esbl'])
    
    # Run PCA
    pca_results = analyzer.pca_analysis(n_components=3)
    
    print(f"\n✓ PCA Results:")
    print(f"  Explained variance: {pca_results['explained_variance_ratio']}")
    print(f"  Cumulative variance: {pca_results['cumulative_variance']}")
    
    # Plot PCA colored by different attributes
    attributes = ['bacterial_species', 'sample_source', 'administrative_region']
    
    for attr in attributes:
        if attr in analyzer.df.columns:
            fig = analyzer.plot_dimred_scatter(method='pca', color_by=attr, 
                                              figsize=(10, 8))
            filename = f'pca_{attr}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
            plt.close()
    
    # Plot colored by MAR index
    if 'mar_index' in analyzer.df.columns:
        fig = analyzer.plot_dimred_scatter(method='pca', color_by='mar_index',
                                          figsize=(10, 8))
        plt.savefig('pca_mar_index.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: pca_mar_index.png")
        plt.close()
    
    return analyzer, pca_results


def example_5_tsne_analysis():
    """
    Example 5: t-SNE dimensionality reduction
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: t-SNE Analysis")
    print("="*80)
    
    # Prepare data
    prep = AMRDataPreparation('rawdata.csv')
    df = prep.prepare_data(
        include_binary=False,
        include_ordinal=True,
        include_onehot=False,
        scale=True
    )
    
    groups = prep.get_feature_groups()
    feature_cols = groups['ordinal_resistance'] + groups['amr_indices']
    
    # Initialize analyzer
    analyzer = UnsupervisedAMRAnalysis(df, feature_cols,
                                      metadata_cols=['bacterial_species', 'sample_source'])
    
    # Run t-SNE
    tsne_results = analyzer.tsne_analysis(n_components=2, perplexity=30.0)
    
    print("\n✓ t-SNE embedding complete")
    
    # Plot t-SNE colored by species
    fig = analyzer.plot_dimred_scatter(method='tsne', color_by='bacterial_species',
                                      figsize=(12, 9))
    plt.savefig('tsne_species.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: tsne_species.png")
    plt.close()
    
    # Plot colored by sample source
    fig = analyzer.plot_dimred_scatter(method='tsne', color_by='sample_source',
                                      figsize=(10, 8))
    plt.savefig('tsne_source.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: tsne_source.png")
    plt.close()
    
    return analyzer, tsne_results


def example_6_umap_analysis():
    """
    Example 6: UMAP dimensionality reduction (optional)
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: UMAP Analysis")
    print("="*80)
    
    try:
        import umap
    except ImportError:
        print("\n⚠ UMAP requires 'umap-learn' package. Skipping this example.")
        print("  Install with: pip install umap-learn")
        return None, None
    
    # Prepare data
    prep = AMRDataPreparation('rawdata.csv')
    df = prep.prepare_data(
        include_binary=False,
        include_ordinal=True,
        include_onehot=False,
        scale=True
    )
    
    groups = prep.get_feature_groups()
    feature_cols = groups['ordinal_resistance'] + groups['amr_indices']
    
    # Initialize analyzer
    analyzer = UnsupervisedAMRAnalysis(df, feature_cols,
                                      metadata_cols=['bacterial_species', 'sample_source'])
    
    # Run UMAP
    umap_results = analyzer.umap_analysis(n_components=2, n_neighbors=15, min_dist=0.1)
    
    print("\n✓ UMAP embedding complete")
    
    # Plot UMAP
    fig = analyzer.plot_dimred_scatter(method='umap', color_by='bacterial_species',
                                      figsize=(12, 9))
    plt.savefig('umap_species.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: umap_species.png")
    plt.close()
    
    return analyzer, umap_results


def example_7_association_rules_apriori():
    """
    Example 7: Association rule mining with Apriori
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: Association Rule Mining - Apriori")
    print("="*80)
    
    # Prepare data with binary encoding
    prep = AMRDataPreparation('rawdata.csv')
    df = prep.prepare_data(
        include_binary=True,
        include_ordinal=False,
        include_onehot=False,  # We'll handle this in transaction preparation
        scale=False  # Don't scale for rule mining
    )
    
    # Add back metadata columns for transaction preparation
    df['bacterial_species'] = prep.df_raw['bacterial_species']
    df['sample_source'] = prep.df_raw['sample_source']
    df['administrative_region'] = prep.df_raw['administrative_region']
    df['esbl'] = prep.df_raw['esbl']
    
    # Get feature columns
    groups = prep.get_feature_groups()
    feature_cols = groups['binary_resistance']
    
    # Initialize analyzer
    analyzer = UnsupervisedAMRAnalysis(df, feature_cols,
                                      metadata_cols=['bacterial_species', 'sample_source'])
    
    # Prepare transactions
    print("\n✓ Preparing transactions...")
    transactions = analyzer.prepare_transactions(
        resistance_threshold=0.5,
        include_metadata=True,
        mar_threshold=0.3  # High MAR if > 0.3
    )
    
    print(f"  Transaction shape: {transactions.shape}")
    print(f"  Items: {transactions.shape[1]}")
    
    # Mine rules with Apriori
    rules = analyzer.apriori_mining(
        transactions,
        min_support=0.05,
        min_confidence=0.6,
        min_lift=1.0
    )
    
    if len(rules) > 0:
        # Interpret top rules
        formatted_rules = analyzer.interpret_rules(rules, top_n=10)
        
        # Save rules to CSV
        rules.to_csv('association_rules_apriori.csv', index=False)
        print("\n✓ Saved: association_rules_apriori.csv")
    
    return analyzer, rules


def example_8_association_rules_fpgrowth():
    """
    Example 8: Association rule mining with FP-Growth
    """
    print("\n" + "="*80)
    print("EXAMPLE 8: Association Rule Mining - FP-Growth")
    print("="*80)
    
    # Prepare data
    prep = AMRDataPreparation('rawdata.csv')
    df = prep.prepare_data(
        include_binary=True,
        include_ordinal=False,
        include_onehot=False,
        scale=False
    )
    
    # Add back metadata
    df['bacterial_species'] = prep.df_raw['bacterial_species']
    df['sample_source'] = prep.df_raw['sample_source']
    
    groups = prep.get_feature_groups()
    feature_cols = groups['binary_resistance']
    
    # Initialize analyzer
    analyzer = UnsupervisedAMRAnalysis(df, feature_cols)
    
    # Prepare transactions
    transactions = analyzer.prepare_transactions(
        resistance_threshold=0.5,
        include_metadata=True,
        mar_threshold=0.3
    )
    
    # Mine rules with FP-Growth
    rules = analyzer.fpgrowth_mining(
        transactions,
        min_support=0.05,
        min_confidence=0.6,
        min_lift=1.0
    )
    
    if len(rules) > 0:
        # Interpret rules
        formatted_rules = analyzer.interpret_rules(rules, top_n=10)
        
        # Save rules
        rules.to_csv('association_rules_fpgrowth.csv', index=False)
        print("\n✓ Saved: association_rules_fpgrowth.csv")
    
    return analyzer, rules


def example_9_combined_analysis():
    """
    Example 9: Combined clustering and dimensionality reduction
    
    Visualize clusters in reduced dimensional space
    """
    print("\n" + "="*80)
    print("EXAMPLE 9: Combined Clustering + Dimensionality Reduction")
    print("="*80)
    
    # Prepare data
    prep = AMRDataPreparation('rawdata.csv')
    df = prep.prepare_data(
        include_binary=False,
        include_ordinal=True,
        include_onehot=False,
        scale=True
    )
    
    groups = prep.get_feature_groups()
    feature_cols = groups['ordinal_resistance'] + groups['amr_indices']
    
    # Initialize analyzer
    analyzer = UnsupervisedAMRAnalysis(df, feature_cols,
                                      metadata_cols=['bacterial_species', 'sample_source'])
    
    # Run k-means
    print("\n✓ Running k-means clustering...")
    kmeans_results = analyzer.kmeans_clustering(k_range=(2, 10))
    optimal_k = 5
    cluster_labels = kmeans_results['labels'][optimal_k]
    
    # Add cluster labels to dataframe
    analyzer.df['kmeans_cluster'] = cluster_labels
    
    # Run PCA
    print("\n✓ Running PCA...")
    pca_results = analyzer.pca_analysis(n_components=3)
    
    # Run t-SNE
    print("\n✓ Running t-SNE...")
    tsne_results = analyzer.tsne_analysis(n_components=2)
    
    # Plot PCA colored by clusters
    fig = analyzer.plot_dimred_scatter(method='pca', color_by='kmeans_cluster',
                                      figsize=(10, 8))
    plt.savefig('pca_kmeans_clusters.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: pca_kmeans_clusters.png")
    plt.close()
    
    # Plot t-SNE colored by clusters
    fig = analyzer.plot_dimred_scatter(method='tsne', color_by='kmeans_cluster',
                                      figsize=(10, 8))
    plt.savefig('tsne_kmeans_clusters.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: tsne_kmeans_clusters.png")
    plt.close()
    
    return analyzer


def example_10_comparative_clustering():
    """
    Example 10: Compare different clustering methods
    """
    print("\n" + "="*80)
    print("EXAMPLE 10: Comparative Clustering Analysis")
    print("="*80)
    
    # Prepare data
    prep = AMRDataPreparation('rawdata.csv')
    df = prep.prepare_data(
        include_binary=False,
        include_ordinal=True,
        include_onehot=False,
        scale=True
    )
    
    groups = prep.get_feature_groups()
    feature_cols = groups['ordinal_resistance'] + groups['amr_indices']
    
    # Initialize analyzer
    analyzer = UnsupervisedAMRAnalysis(df, feature_cols,
                                      metadata_cols=['bacterial_species', 'sample_source',
                                                    'administrative_region'])
    
    # Run all clustering methods
    print("\n✓ Running k-means...")
    kmeans_results = analyzer.kmeans_clustering(k_range=(2, 10))
    kmeans_labels = kmeans_results['labels'][5]  # k=5
    
    print("\n✓ Running hierarchical clustering...")
    hier_results = analyzer.hierarchical_clustering(n_clusters=5)
    hier_labels = hier_results['labels']
    
    print("\n✓ Running DBSCAN...")
    dbscan_results = analyzer.dbscan_clustering(eps=2.0, min_samples=5)
    dbscan_labels = dbscan_results['labels']
    
    # Analyze each method
    print("\n" + "="*80)
    print("COMPARATIVE RESULTS")
    print("="*80)
    
    print("\nK-means (k=5):")
    kmeans_analysis = analyzer.analyze_clusters(kmeans_labels, method_name='K-means')
    
    print("\nHierarchical (n=5):")
    hier_analysis = analyzer.analyze_clusters(hier_labels, method_name='Hierarchical')
    
    print("\nDBSCAN:")
    if dbscan_results['n_clusters'] > 0:
        dbscan_analysis = analyzer.analyze_clusters(dbscan_labels, method_name='DBSCAN')
    
    # Create comparison visualization with PCA
    analyzer.pca_analysis(n_components=2)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Add cluster labels
    analyzer.df['kmeans'] = kmeans_labels
    analyzer.df['hierarchical'] = hier_labels
    analyzer.df['dbscan'] = dbscan_labels
    
    for ax, method in zip(axes, ['kmeans', 'hierarchical', 'dbscan']):
        scatter = ax.scatter(analyzer.df['PC1'], analyzer.df['PC2'],
                           c=analyzer.df[method], cmap='Set3', alpha=0.6, s=30)
        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('PC2', fontsize=11)
        ax.set_title(f'{method.capitalize()} Clustering', fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clustering_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: clustering_comparison.png")
    plt.close()
    
    return analyzer


def main():
    """
    Run all examples
    """
    print("\n" + "="*80)
    print("UNSUPERVISED PATTERN RECOGNITION - USAGE EXAMPLES")
    print("="*80)
    
    try:
        # Clustering examples
        analyzer1, results1 = example_1_kmeans_clustering()
        analyzer2, results2 = example_2_hierarchical_clustering()
        analyzer3, results3 = example_3_dbscan_clustering()
        
        # Dimensionality reduction examples
        analyzer4, results4 = example_4_pca_analysis()
        analyzer5, results5 = example_5_tsne_analysis()
        analyzer6, results6 = example_6_umap_analysis()
        
        # Association rule mining examples
        analyzer7, rules7 = example_7_association_rules_apriori()
        analyzer8, rules8 = example_8_association_rules_fpgrowth()
        
        # Combined analyses
        analyzer9 = example_9_combined_analysis()
        analyzer10 = example_10_comparative_clustering()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nGenerated files:")
        print("  Clustering:")
        print("    - kmeans_evaluation.png")
        print("    - hierarchical_dendrogram.png")
        print("    - hierarchical_heatmap.png")
        print("    - dbscan_k_distance.png")
        print("    - dbscan_composition.png")
        print("  Dimensionality Reduction:")
        print("    - pca_*.png (multiple files)")
        print("    - tsne_*.png (multiple files)")
        print("    - umap_species.png (if UMAP installed)")
        print("  Association Rules:")
        print("    - association_rules_apriori.csv")
        print("    - association_rules_fpgrowth.csv")
        print("  Combined:")
        print("    - pca_kmeans_clusters.png")
        print("    - tsne_kmeans_clusters.png")
        print("    - clustering_comparison.png")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
