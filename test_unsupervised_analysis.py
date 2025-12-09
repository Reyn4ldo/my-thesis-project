"""
Unit tests for Unsupervised Pattern Recognition Module

This test suite validates all unsupervised analysis functions including:
- Clustering methods (k-means, hierarchical, DBSCAN)
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Association rule mining (Apriori, FP-Growth)
"""

import unittest
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from data_preparation import AMRDataPreparation
from unsupervised_analysis import (
    UnsupervisedAMRAnalysis,
    quick_clustering_analysis,
    quick_dimred_analysis
)


class TestUnsupervisedAMRAnalysis(unittest.TestCase):
    """Test suite for Unsupervised AMR Analysis class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - run once before all tests"""
        cls.test_csv_path = 'rawdata.csv'
        if not os.path.exists(cls.test_csv_path):
            raise FileNotFoundError(f"Test data file not found: {cls.test_csv_path}")
        
        # Prepare data once for all tests
        print("\nPreparing data for tests...")
        prep = AMRDataPreparation(cls.test_csv_path)
        cls.df = prep.prepare_data(
            include_binary=True,
            include_ordinal=True,
            include_onehot=False,
            scale=True,
            drop_original_int=True
        )
        
        # Keep raw metadata
        cls.df['bacterial_species'] = prep.df_raw['bacterial_species']
        cls.df['sample_source'] = prep.df_raw['sample_source']
        cls.df['administrative_region'] = prep.df_raw['administrative_region']
        cls.df['esbl'] = prep.df_raw['esbl']
        
        # Get feature groups
        groups = prep.get_feature_groups()
        cls.ordinal_features = groups['ordinal_resistance']
        cls.binary_features = groups['binary_resistance']
        cls.amr_features = groups['amr_indices']
        
        print(f"✓ Data prepared: {cls.df.shape}")
        print(f"✓ Ordinal features: {len(cls.ordinal_features)}")
        print(f"✓ Binary features: {len(cls.binary_features)}")
    
    def setUp(self):
        """Set up for each test"""
        # Use small subset for faster testing
        self.test_df = self.df.head(100).copy()
        self.feature_cols = self.ordinal_features[:10] + self.amr_features
        self.metadata_cols = ['bacterial_species', 'sample_source']
    
    # ==================== CLUSTERING TESTS ====================
    
    def test_01_kmeans_clustering(self):
        """Test k-means clustering"""
        print("\n" + "="*60)
        print("TEST 01: K-means Clustering")
        print("="*60)
        
        analyzer = UnsupervisedAMRAnalysis(self.test_df, self.feature_cols, 
                                          self.metadata_cols)
        
        results = analyzer.kmeans_clustering(k_range=(2, 5), random_state=42)
        
        # Check results structure
        self.assertIn('k_values', results)
        self.assertIn('inertias', results)
        self.assertIn('silhouette_scores', results)
        self.assertIn('labels', results)
        
        # Check k values
        self.assertEqual(results['k_values'], [2, 3, 4, 5])
        
        # Check that we have results for each k
        self.assertEqual(len(results['inertias']), 4)
        self.assertEqual(len(results['silhouette_scores']), 4)
        
        # Check that inertias decrease
        self.assertTrue(all(results['inertias'][i] >= results['inertias'][i+1] 
                          for i in range(len(results['inertias'])-1)))
        
        # Check labels shape
        for k in results['k_values']:
            labels = results['labels'][k]
            self.assertEqual(len(labels), len(self.test_df))
            self.assertEqual(len(set(labels)), k)
        
        print(f"✓ K-means clustering: {len(results['k_values'])} k values tested")
        print(f"✓ Silhouette scores: {[f'{s:.3f}' if s else 'N/A' for s in results['silhouette_scores']]}")
    
    def test_02_kmeans_evaluation_plot(self):
        """Test k-means evaluation plot"""
        print("\n" + "="*60)
        print("TEST 02: K-means Evaluation Plot")
        print("="*60)
        
        analyzer = UnsupervisedAMRAnalysis(self.test_df, self.feature_cols)
        analyzer.kmeans_clustering(k_range=(2, 5))
        
        fig = analyzer.plot_kmeans_evaluation(figsize=(10, 4))
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 2)
        
        plt.close(fig)
        print("✓ K-means evaluation plot created successfully")
    
    def test_03_hierarchical_clustering(self):
        """Test hierarchical clustering"""
        print("\n" + "="*60)
        print("TEST 03: Hierarchical Clustering")
        print("="*60)
        
        analyzer = UnsupervisedAMRAnalysis(self.test_df, self.feature_cols)
        
        results = analyzer.hierarchical_clustering(n_clusters=3, linkage_method='ward')
        
        # Check results
        self.assertIn('linkage_matrix', results)
        self.assertIn('labels', results)
        self.assertIn('silhouette_score', results)
        
        # Check labels
        labels = results['labels']
        self.assertEqual(len(labels), len(self.test_df))
        self.assertEqual(len(set(labels)), 3)
        
        # Check silhouette score range
        self.assertGreaterEqual(results['silhouette_score'], -1)
        self.assertLessEqual(results['silhouette_score'], 1)
        
        print(f"✓ Hierarchical clustering: {results['n_clusters']} clusters")
        print(f"✓ Silhouette score: {results['silhouette_score']:.3f}")
    
    def test_04_dendrogram_plot(self):
        """Test dendrogram plotting"""
        print("\n" + "="*60)
        print("TEST 04: Dendrogram Plot")
        print("="*60)
        
        analyzer = UnsupervisedAMRAnalysis(self.test_df, self.feature_cols)
        analyzer.hierarchical_clustering(n_clusters=3)
        
        fig = analyzer.plot_dendrogram(max_display_levels=5, figsize=(12, 5))
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 1)
        
        plt.close(fig)
        print("✓ Dendrogram plot created successfully")
    
    def test_05_dbscan_clustering(self):
        """Test DBSCAN clustering"""
        print("\n" + "="*60)
        print("TEST 05: DBSCAN Clustering")
        print("="*60)
        
        analyzer = UnsupervisedAMRAnalysis(self.test_df, self.feature_cols)
        
        results = analyzer.dbscan_clustering(eps=2.0, min_samples=3)
        
        # Check results
        self.assertIn('labels', results)
        self.assertIn('n_clusters', results)
        self.assertIn('n_noise', results)
        
        # Check labels
        labels = results['labels']
        self.assertEqual(len(labels), len(self.test_df))
        
        # Check that n_clusters matches unique non-noise labels
        n_clusters_calc = len(set(labels)) - (1 if -1 in labels else 0)
        self.assertEqual(results['n_clusters'], n_clusters_calc)
        
        print(f"✓ DBSCAN: {results['n_clusters']} clusters, {results['n_noise']} noise points")
    
    def test_06_k_distance_plot(self):
        """Test k-distance plot"""
        print("\n" + "="*60)
        print("TEST 06: K-distance Plot")
        print("="*60)
        
        analyzer = UnsupervisedAMRAnalysis(self.test_df, self.feature_cols)
        
        fig = analyzer.plot_k_distance(k=5, figsize=(10, 5))
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 1)
        
        plt.close(fig)
        print("✓ K-distance plot created successfully")
    
    def test_07_cluster_analysis(self):
        """Test cluster analysis"""
        print("\n" + "="*60)
        print("TEST 07: Cluster Analysis")
        print("="*60)
        
        analyzer = UnsupervisedAMRAnalysis(self.test_df, self.feature_cols,
                                          self.metadata_cols)
        
        # Run clustering
        results = analyzer.kmeans_clustering(k_range=(2, 5))
        labels = results['labels'][3]  # Use k=3
        
        # Analyze clusters
        analysis_df = analyzer.analyze_clusters(labels, method_name='Test')
        
        # Check analysis results
        self.assertIsInstance(analysis_df, pd.DataFrame)
        self.assertGreater(len(analysis_df), 0)
        self.assertIn('cluster', analysis_df.columns)
        self.assertIn('size', analysis_df.columns)
        
        # Check that all clusters are analyzed
        self.assertEqual(len(analysis_df), 3)
        
        print(f"✓ Cluster analysis: {len(analysis_df)} clusters analyzed")
        print(f"✓ Cluster sizes: {analysis_df['size'].tolist()}")
    
    def test_08_cluster_heatmap(self):
        """Test cluster heatmap"""
        print("\n" + "="*60)
        print("TEST 08: Cluster Heatmap")
        print("="*60)
        
        analyzer = UnsupervisedAMRAnalysis(self.test_df, self.feature_cols)
        
        results = analyzer.kmeans_clustering(k_range=(2, 5))
        labels = results['labels'][3]
        
        fig = analyzer.plot_cluster_heatmap(labels, 
                                           resistance_cols=self.ordinal_features[:10],
                                           figsize=(10, 6))
        
        self.assertIsNotNone(fig)
        plt.close(fig)
        print("✓ Cluster heatmap created successfully")
    
    def test_09_cluster_composition(self):
        """Test cluster composition plot"""
        print("\n" + "="*60)
        print("TEST 09: Cluster Composition")
        print("="*60)
        
        analyzer = UnsupervisedAMRAnalysis(self.test_df, self.feature_cols,
                                          self.metadata_cols)
        
        results = analyzer.kmeans_clustering(k_range=(2, 5))
        labels = results['labels'][3]
        
        fig = analyzer.plot_cluster_composition(labels, 
                                               metadata_col='bacterial_species',
                                               figsize=(10, 6))
        
        self.assertIsNotNone(fig)
        plt.close(fig)
        print("✓ Cluster composition plot created successfully")
    
    # ==================== DIMENSIONALITY REDUCTION TESTS ====================
    
    def test_10_pca_analysis(self):
        """Test PCA analysis"""
        print("\n" + "="*60)
        print("TEST 10: PCA Analysis")
        print("="*60)
        
        analyzer = UnsupervisedAMRAnalysis(self.test_df, self.feature_cols)
        
        results = analyzer.pca_analysis(n_components=3)
        
        # Check results
        self.assertIn('components', results)
        self.assertIn('explained_variance_ratio', results)
        self.assertIn('cumulative_variance', results)
        
        # Check components shape
        components = results['components']
        self.assertEqual(components.shape, (len(self.test_df), 3))
        
        # Check variance ratios
        variance = results['explained_variance_ratio']
        self.assertEqual(len(variance), 3)
        self.assertTrue(all(0 <= v <= 1 for v in variance))
        
        # Check that PC columns were added to df
        self.assertIn('PC1', analyzer.df.columns)
        self.assertIn('PC2', analyzer.df.columns)
        self.assertIn('PC3', analyzer.df.columns)
        
        print(f"✓ PCA: 3 components extracted")
        print(f"✓ Explained variance: {[f'{v:.3f}' for v in variance]}")
        print(f"✓ Cumulative variance: {results['cumulative_variance'][-1]:.3f}")
    
    def test_11_tsne_analysis(self):
        """Test t-SNE analysis"""
        print("\n" + "="*60)
        print("TEST 11: t-SNE Analysis")
        print("="*60)
        
        analyzer = UnsupervisedAMRAnalysis(self.test_df, self.feature_cols)
        
        results = analyzer.tsne_analysis(n_components=2, perplexity=10.0)
        
        # Check results
        self.assertIn('components', results)
        
        # Check components shape
        components = results['components']
        self.assertEqual(components.shape, (len(self.test_df), 2))
        
        # Check that TSNE columns were added
        self.assertIn('TSNE1', analyzer.df.columns)
        self.assertIn('TSNE2', analyzer.df.columns)
        
        print("✓ t-SNE: 2D embedding created")
    
    def test_12_umap_analysis(self):
        """Test UMAP analysis (optional)"""
        print("\n" + "="*60)
        print("TEST 12: UMAP Analysis")
        print("="*60)
        
        try:
            import umap
            
            analyzer = UnsupervisedAMRAnalysis(self.test_df, self.feature_cols)
            
            results = analyzer.umap_analysis(n_components=2, n_neighbors=10)
            
            # Check results
            self.assertIn('components', results)
            
            # Check components shape
            components = results['components']
            self.assertEqual(components.shape, (len(self.test_df), 2))
            
            # Check that UMAP columns were added
            self.assertIn('UMAP1', analyzer.df.columns)
            self.assertIn('UMAP2', analyzer.df.columns)
            
            print("✓ UMAP: 2D embedding created")
            
        except ImportError:
            print("⚠ UMAP not installed, skipping test")
            self.skipTest("UMAP not installed")
    
    def test_13_dimred_scatter_plot(self):
        """Test dimensionality reduction scatter plots"""
        print("\n" + "="*60)
        print("TEST 13: Dimensionality Reduction Scatter Plots")
        print("="*60)
        
        analyzer = UnsupervisedAMRAnalysis(self.test_df, self.feature_cols,
                                          self.metadata_cols)
        
        # Run PCA and t-SNE
        analyzer.pca_analysis(n_components=3)
        analyzer.tsne_analysis(n_components=2, perplexity=10.0)
        
        # Test PCA plot
        fig_pca = analyzer.plot_dimred_scatter(method='pca', 
                                              color_by='bacterial_species',
                                              figsize=(8, 6))
        self.assertIsNotNone(fig_pca)
        plt.close(fig_pca)
        print("✓ PCA scatter plot created")
        
        # Test t-SNE plot
        fig_tsne = analyzer.plot_dimred_scatter(method='tsne',
                                               color_by='sample_source',
                                               figsize=(8, 6))
        self.assertIsNotNone(fig_tsne)
        plt.close(fig_tsne)
        print("✓ t-SNE scatter plot created")
    
    # ==================== ASSOCIATION RULE MINING TESTS ====================
    
    def test_14_prepare_transactions(self):
        """Test transaction preparation"""
        print("\n" + "="*60)
        print("TEST 14: Transaction Preparation")
        print("="*60)
        
        # Use binary features for rule mining
        analyzer = UnsupervisedAMRAnalysis(self.test_df, self.binary_features[:10],
                                          self.metadata_cols)
        
        transactions = analyzer.prepare_transactions(
            resistance_threshold=0.5,
            include_metadata=True,
            mar_threshold=0.3
        )
        
        # Check transactions
        self.assertIsInstance(transactions, pd.DataFrame)
        self.assertEqual(len(transactions), len(self.test_df))
        
        # Check that all values are binary
        self.assertTrue(transactions.isin([0, 1]).all().all())
        
        # Check that resistance items exist
        resistance_items = [col for col in transactions.columns if col.startswith('R_')]
        self.assertGreater(len(resistance_items), 0)
        
        print(f"✓ Transactions prepared: {transactions.shape}")
        print(f"✓ Items: {len(transactions.columns)}")
        print(f"✓ Resistance items: {len(resistance_items)}")
    
    def test_15_apriori_mining(self):
        """Test Apriori algorithm"""
        print("\n" + "="*60)
        print("TEST 15: Apriori Algorithm")
        print("="*60)
        
        analyzer = UnsupervisedAMRAnalysis(self.test_df, self.binary_features[:10])
        
        transactions = analyzer.prepare_transactions(
            resistance_threshold=0.5,
            include_metadata=False,
            mar_threshold=None
        )
        
        rules = analyzer.apriori_mining(
            transactions,
            min_support=0.1,
            min_confidence=0.5,
            min_lift=1.0
        )
        
        # Check rules (may be empty with low support)
        self.assertIsInstance(rules, pd.DataFrame)
        
        if len(rules) > 0:
            # Check required columns
            self.assertIn('antecedents', rules.columns)
            self.assertIn('consequents', rules.columns)
            self.assertIn('support', rules.columns)
            self.assertIn('confidence', rules.columns)
            self.assertIn('lift', rules.columns)
            
            # Check value ranges
            self.assertTrue((rules['support'] >= 0.1).all())
            self.assertTrue((rules['confidence'] >= 0.5).all())
            self.assertTrue((rules['lift'] >= 1.0).all())
            
            print(f"✓ Apriori: {len(rules)} rules found")
        else:
            print("✓ Apriori: No rules found (try lower thresholds)")
    
    def test_16_fpgrowth_mining(self):
        """Test FP-Growth algorithm"""
        print("\n" + "="*60)
        print("TEST 16: FP-Growth Algorithm")
        print("="*60)
        
        analyzer = UnsupervisedAMRAnalysis(self.test_df, self.binary_features[:10])
        
        transactions = analyzer.prepare_transactions(
            resistance_threshold=0.5,
            include_metadata=False,
            mar_threshold=None
        )
        
        rules = analyzer.fpgrowth_mining(
            transactions,
            min_support=0.1,
            min_confidence=0.5,
            min_lift=1.0
        )
        
        # Check rules
        self.assertIsInstance(rules, pd.DataFrame)
        
        if len(rules) > 0:
            # Check required columns
            self.assertIn('antecedents', rules.columns)
            self.assertIn('consequents', rules.columns)
            self.assertIn('support', rules.columns)
            self.assertIn('confidence', rules.columns)
            self.assertIn('lift', rules.columns)
            
            print(f"✓ FP-Growth: {len(rules)} rules found")
        else:
            print("✓ FP-Growth: No rules found (try lower thresholds)")
    
    def test_17_interpret_rules(self):
        """Test rule interpretation"""
        print("\n" + "="*60)
        print("TEST 17: Rule Interpretation")
        print("="*60)
        
        analyzer = UnsupervisedAMRAnalysis(self.test_df, self.binary_features[:10])
        
        transactions = analyzer.prepare_transactions(
            resistance_threshold=0.5,
            include_metadata=False
        )
        
        rules = analyzer.apriori_mining(
            transactions,
            min_support=0.1,
            min_confidence=0.5,
            min_lift=1.0
        )
        
        if len(rules) > 0:
            formatted = analyzer.interpret_rules(rules, top_n=5)
            
            self.assertIsInstance(formatted, pd.DataFrame)
            self.assertIn('rule', formatted.columns)
            self.assertIn('support', formatted.columns)
            self.assertIn('confidence', formatted.columns)
            self.assertIn('lift', formatted.columns)
            
            print(f"✓ Interpreted {len(formatted)} rules")
        else:
            print("✓ No rules to interpret")
    
    # ==================== CONVENIENCE FUNCTION TESTS ====================
    
    def test_18_quick_clustering(self):
        """Test quick clustering convenience function"""
        print("\n" + "="*60)
        print("TEST 18: Quick Clustering")
        print("="*60)
        
        results = quick_clustering_analysis(
            self.test_df,
            self.feature_cols,
            methods=['kmeans', 'hierarchical'],
            k=3
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('kmeans', results)
        self.assertIn('hierarchical', results)
        
        print("✓ Quick clustering completed")
    
    def test_19_quick_dimred(self):
        """Test quick dimensionality reduction"""
        print("\n" + "="*60)
        print("TEST 19: Quick Dimensionality Reduction")
        print("="*60)
        
        results = quick_dimred_analysis(
            self.test_df,
            self.feature_cols,
            methods=['pca', 'tsne']
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('pca', results)
        self.assertIn('tsne', results)
        
        print("✓ Quick dimensionality reduction completed")
    
    def test_20_integration_workflow(self):
        """Test complete integration workflow"""
        print("\n" + "="*60)
        print("TEST 20: Integration Workflow")
        print("="*60)
        
        # Initialize analyzer
        analyzer = UnsupervisedAMRAnalysis(self.test_df, self.feature_cols,
                                          self.metadata_cols)
        
        # Step 1: Clustering
        print("\n  Step 1: Clustering...")
        kmeans_results = analyzer.kmeans_clustering(k_range=(2, 4))
        labels = kmeans_results['labels'][3]
        
        # Step 2: Dimensionality reduction
        print("  Step 2: Dimensionality reduction...")
        pca_results = analyzer.pca_analysis(n_components=2)
        
        # Step 3: Visualize clusters in reduced space
        print("  Step 3: Visualization...")
        analyzer.df['cluster'] = labels
        fig = analyzer.plot_dimred_scatter(method='pca', color_by='cluster')
        plt.close(fig)
        
        # Step 4: Analyze clusters
        print("  Step 4: Cluster analysis...")
        analysis_df = analyzer.analyze_clusters(labels)
        
        self.assertIsNotNone(analysis_df)
        self.assertGreater(len(analysis_df), 0)
        
        print("✓ Integration workflow completed successfully")


def run_tests():
    """Run all tests"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestUnsupervisedAMRAnalysis)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
