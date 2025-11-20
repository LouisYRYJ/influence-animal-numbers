import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class AttributionAnalysis:
    """Analyzes attribution methods for animal predictions."""
    
    def __init__(self, results_dir: str = ".", save_csvs: bool = False):
        """Initialize with results directory path.
        
        Args:
            results_dir: Directory containing the CSV result files
            save_csvs: If True, save intermediate CSV files for plots
        """
        self.results_dir = Path(results_dir)
        self.save_csvs = save_csvs
        self.methods = [
            'base_prompting',
            'difference_in_prompting', 
            'logit',
            'subliminal_prompting',
            'unembedding'
        ]
        self.animals = []  # Will be auto-detected from CSV columns
        self.data = {}
        self.load_data()
        
    def load_data(self):
        """Load all CSV files and auto-detect animals from column names."""
        print("Loading data files...")
        for method in self.methods:
            filepath = self.results_dir / f"{method}.csv"
            if filepath.exists():
                df = pd.read_csv(filepath, index_col=0)
                self.data[method] = df
                
                # Auto-detect animals from the first CSV file loaded
                if not self.animals and len(df.columns) > 0:
                    self.animals = df.columns.tolist()
                    print(f"  Auto-detected {len(self.animals)} animals: {', '.join(self.animals)}")
                
                print(f"  Loaded {method}: {df.shape}")
            else:
                print(f"  Missing {method}")
        
        if not self.animals:
            raise ValueError("Could not auto-detect animals. No CSV files with columns found.")
                
    def compute_summary_statistics(self) -> pd.DataFrame:
        """Compute summary statistics for each method."""
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        
        summary_data = []
        
        for method in self.methods:
            if method not in self.data:
                continue
                
            df = self.data[method]
            
            # Overall statistics
            all_values = df.values.flatten()
            
            stats_dict = {
                'Method': method,
                'Mean': np.mean(all_values),
                'Median': np.median(all_values),
                'Std': np.std(all_values),
                'Min': np.min(all_values),
                'Max': np.max(all_values),
                'Q1': np.percentile(all_values, 25),
                'Q3': np.percentile(all_values, 75),
                'IQR': np.percentile(all_values, 75) - np.percentile(all_values, 25)
            }
            
            summary_data.append(stats_dict)
            
        summary_df = pd.DataFrame(summary_data)
        print("\n", summary_df.to_string(index=False))
        
        # Save to CSV
        output_file = self.results_dir / "summary_statistics.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")
        
        return summary_df
    
    def compute_per_animal_statistics(self) -> Dict[str, pd.DataFrame]:
        """Compute statistics per animal across methods."""
        print("\n" + "="*70)
        print("PER-ANIMAL STATISTICS")
        print("="*70)
        
        results = {}
        
        for animal in self.animals:
            animal_data = []
            
            for method in self.methods:
                if method not in self.data:
                    continue
                    
                values = self.data[method][animal].values
                
                animal_data.append({
                    'Method': method,
                    'Animal': animal,
                    'Mean': np.mean(values),
                    'Median': np.median(values),
                    'Std': np.std(values),
                    'Min': np.min(values),
                    'Max': np.max(values)
                })
                
            results[animal] = pd.DataFrame(animal_data)
            
        # Print for a few example animals
        for animal in self.animals[:3]:
            print(f"\n{animal.upper()}:")
            print(results[animal].to_string(index=False))
            
        # Save all to CSV
        output_file = self.results_dir / "per_animal_statistics.csv"
        combined_df = pd.concat(results.values(), ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"\nSaved all animal statistics to {output_file}")
        
        return results
    
    def correlation_analysis(self):
        """Analyze correlations between methods."""
        print("\n" + "="*70)
        print("CORRELATION ANALYSIS")
        print("="*70)
        
        # Overall correlation across all data
        correlation_data = {}
        for method in self.methods:
            if method not in self.data:
                continue
            # Flatten the data
            correlation_data[method] = self.data[method].values.flatten()
            
        corr_df = pd.DataFrame(correlation_data)
        correlation_matrix = corr_df.corr()
        
        print("\nOverall Correlation Matrix:")
        print(correlation_matrix.to_string())
        
        # Save correlation matrix
        output_file = self.results_dir / "correlation_matrix_overall.csv"
        correlation_matrix.to_csv(output_file)
        print(f"\nSaved to {output_file}")
        
        # Visualize overall correlation matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.3f', 
                   cmap='coolwarm', center=0, square=True,
                   linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
                   xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.index)
        ax.set_title('Overall Correlation Between Attribution Methods', fontsize=14, fontweight='bold', pad=10)
        plt.tight_layout()
        output_fig = self.results_dir / "correlation_heatmap_overall.png"
        plt.savefig(output_fig, dpi=300, bbox_inches='tight')
        print(f"Saved overall correlation heatmap to {output_fig}")
        plt.close()
        plt.clf()
        
        # Per-animal correlations
        self._plot_per_animal_correlations()
        
        return correlation_matrix
    
    def _plot_per_animal_correlations(self):
        """Plot correlation matrices for each animal."""
        print("\n  Generating per-animal correlation matrices...")
        
        # Skip base_prompting for per-animal plots as it has no variance
        methods_to_plot = [m for m in self.methods if m != 'base_prompting' and m in self.data]
        
        all_animal_correlations = {}
        
        for animal in self.animals:
            # Get data for this animal across methods (excluding base_prompting)
            animal_data = {}
            for method in methods_to_plot:
                animal_data[method] = self.data[method][animal].values
                
            animal_df = pd.DataFrame(animal_data)
            animal_corr = animal_df.corr()
            all_animal_correlations[animal] = animal_corr
            
            # Plot individual correlation matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(animal_corr, annot=True, fmt='.3f', 
                       cmap='coolwarm', center=0, square=True,
                       linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
                       xticklabels=animal_corr.columns, yticklabels=animal_corr.index)
            ax.set_title(f'Method Correlations for {animal.title()}', 
                        fontsize=14, fontweight='bold', pad=10)
            # Remove grey padding completely
            plt.tight_layout()
            output_fig = self.results_dir / f"correlation_heatmap_{animal}.png"
            plt.savefig(output_fig, dpi=300, bbox_inches='tight')
            plt.close()
            plt.clf()
            
        print(f"  Saved {len(self.animals)} per-animal correlation heatmaps")
        
        # Save all animal correlations to CSV if requested
        if self.save_csvs:
            for animal, corr_matrix in all_animal_correlations.items():
                output_file = self.results_dir / f"correlation_matrix_{animal}.csv"
                corr_matrix.to_csv(output_file)
                
            print(f"  Saved {len(self.animals)} per-animal correlation matrices to CSV")
    
    def plot_distributions(self):
        """Plot distribution of values for each method."""
        print("\n" + "="*70)
        print("GENERATING DISTRIBUTION PLOTS")
        print("="*70)
        
        # Overall distributions for all methods
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, method in enumerate(self.methods):
            if method not in self.data:
                continue
                
            ax = axes[idx]
            values = self.data[method].values.flatten()
            
            # Histogram
            ax.hist(values, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(values), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(values):.2f}')
            ax.axvline(np.median(values), color='green', linestyle='--',
                      linewidth=2, label=f'Median: {np.median(values):.2f}')
            
            ax.set_xlabel('Value', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(method.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
        # Hide extra subplot
        if len(self.methods) < 6:
            axes[5].axis('off')
            
        plt.tight_layout()
        output_file = self.results_dir / "distributions_all_methods.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved overall distribution plots to {output_file}")
        plt.close()
        
        # Per-animal distributions
        self._plot_per_animal_distributions()
        
    def _plot_per_animal_distributions(self):
        """Plot distributions for each animal across methods."""
        print("  Generating per-animal distribution plots...")
        
        # Skip base_prompting for per-animal plots as it has no variance
        methods_to_plot = [m for m in self.methods if m != 'base_prompting']
        
        for animal in self.animals:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for idx, method in enumerate(methods_to_plot):
                if method not in self.data:
                    continue
                    
                ax = axes[idx]
                values = self.data[method][animal].values
                
                # Histogram
                ax.hist(values, bins=30, alpha=0.7, edgecolor='black', color=f'C{idx}')
                ax.axvline(np.mean(values), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(values):.2f}')
                ax.axvline(np.median(values), color='green', linestyle='--',
                          linewidth=2, label=f'Median: {np.median(values):.2f}')
                
                ax.set_xlabel('Value', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.set_title(method.replace('_', ' ').title(), fontsize=12, fontweight='bold')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                
            plt.suptitle(f'Distribution of Attribution Values for {animal.title()}',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            output_file = self.results_dir / f"distributions_{animal}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
        print(f"  Saved {len(self.animals)} per-animal distribution plots")
        
    def plot_boxplots(self):
        """Create boxplots comparing methods."""
        print("\n" + "="*70)
        print("GENERATING BOXPLOTS")
        print("="*70)
        
        # Overall comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        data_for_plot = []
        labels = []
        
        for method in self.methods:
            if method not in self.data:
                continue
            data_for_plot.append(self.data[method].values.flatten())
            labels.append(method.replace('_', ' ').title())
            
        bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True,
                       showmeans=True, meanline=True)
        
        # Color the boxes
        colors = sns.color_palette("husl", len(data_for_plot))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Comparison of Attribution Methods', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_file = self.results_dir / "boxplot_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved boxplot to {output_file}")
        plt.close()
        
        # Per-animal boxplots
        self._plot_per_animal_boxplots()
        
    def _plot_per_animal_boxplots(self):
        """Create boxplots for each animal across methods."""
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for idx, animal in enumerate(self.animals):
            ax = axes[idx]
            
            data_for_plot = []
            labels = []
            
            for method in self.methods:
                if method not in self.data:
                    continue
                data_for_plot.append(self.data[method][animal].values)
                labels.append(method.replace('_', '\n').replace(' ', '\n'))
                
            bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)
            
            # Color boxes
            colors = sns.color_palette("husl", len(data_for_plot))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
                
            ax.set_title(animal.title(), fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', labelsize=7)
            
        plt.suptitle('Attribution Values by Animal and Method', 
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        output_file = self.results_dir / "per_animal_boxplots.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved per-animal boxplots to {output_file}")
        plt.close()
        
    def plot_heatmaps(self):
        """Create heatmaps for each method."""
        print("\n" + "="*70)
        print("GENERATING HEATMAPS")
        print("="*70)
        
        # Overall heatmap with all methods
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, method in enumerate(self.methods):
            if method not in self.data:
                continue
                
            ax = axes[idx]
            df = self.data[method]
            
            # Sample data if too large
            if len(df) > 100:
                df_sample = df.sample(n=100, random_state=42).sort_index()
            else:
                df_sample = df
                
            sns.heatmap(df_sample.T, ax=ax, cmap='RdYlGn', center=0,
                       cbar_kws={'label': 'Value'}, xticklabels=False)
            ax.set_title(method.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_ylabel('Animal', fontsize=10)
            ax.set_xlabel('Sample Index', fontsize=10)
            
        # Hide extra subplot
        if len(self.methods) < 6:
            axes[5].axis('off')
            
        plt.tight_layout()
        output_file = self.results_dir / "heatmaps_all_methods.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved overall heatmaps to {output_file}")
        plt.close()
        
        # Per-animal heatmaps
        self._plot_per_animal_heatmaps()
        
    def _plot_per_animal_heatmaps(self):
        """Create heatmaps for each animal across methods."""
        print("  Generating per-animal heatmaps...")
        
        # Skip base_prompting for per-animal plots as it has no variance
        methods_to_plot = [m for m in self.methods if m != 'base_prompting']
        
        for animal in self.animals:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            axes = axes.flatten()
            
            for idx, method in enumerate(methods_to_plot):
                if method not in self.data:
                    continue
                    
                ax = axes[idx]
                values = self.data[method][animal].values.reshape(-1, 1)
                
                # Sample if too large
                if len(values) > 200:
                    indices = np.random.choice(len(values), 200, replace=False)
                    indices.sort()
                    values_sample = values[indices]
                else:
                    values_sample = values
                    
                sns.heatmap(values_sample.T, ax=ax, cmap='RdYlGn', center=0,
                           cbar_kws={'label': 'Value'}, xticklabels=False, yticklabels=False)
                ax.set_title(method.replace('_', ' ').title(), fontsize=12, fontweight='bold')
                ax.set_xlabel('Sample Index', fontsize=10)
                
            plt.suptitle(f'Attribution Values Across Samples for {animal.title()}',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            output_file = self.results_dir / f"heatmap_{animal}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
        print(f"  Saved {len(self.animals)} per-animal heatmaps")
        
    def statistical_tests(self):
        """Perform statistical tests comparing methods."""
        print("\n" + "="*70)
        print("STATISTICAL TESTS")
        print("="*70)
        
        results = []
        
        # Pairwise comparisons using Mann-Whitney U test (non-parametric)
        print("\nPairwise Mann-Whitney U Tests:")
        print("-" * 70)
        
        for i, method1 in enumerate(self.methods):
            if method1 not in self.data:
                continue
                
            for method2 in self.methods[i+1:]:
                if method2 not in self.data:
                    continue
                    
                data1 = self.data[method1].values.flatten()
                data2 = self.data[method2].values.flatten()
                
                statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                
                result = {
                    'Method 1': method1,
                    'Method 2': method2,
                    'U-statistic': statistic,
                    'p-value': p_value,
                    'Significant (p<0.05)': 'Yes' if p_value < 0.05 else 'No'
                }
                results.append(result)
                
        test_df = pd.DataFrame(results)
        print(test_df.to_string(index=False))
        
        # Save results
        output_file = self.results_dir / "statistical_tests.csv"
        test_df.to_csv(output_file, index=False)
        print(f"\nSaved statistical tests to {output_file}")
        
        return test_df
    
    def rank_analysis(self):
        """Analyze ranking of animals across methods with detailed explanation."""
        print("\n" + "="*70)
        print("RANK ANALYSIS")
        print("="*70)
        
        print("\n" + "="*70)
        print("DETAILED RANK ANALYSIS EXPLANATION")
        print("="*70)
        print("""
The rank analysis examines how different attribution methods rank animals
based on their mean attribution values. This helps us understand:

1. CONSISTENCY: Do methods agree on which animals have high/low attribution?
2. METHOD DIFFERENCES: How do different methods prioritize animals differently?
3. RANK STABILITY: Are rankings stable or do they vary significantly?

Rankings are computed as follows:
- For each method, calculate the mean attribution value per animal
- Rank animals from 1 (highest mean value) to 10 (lowest mean value)
- Higher rank = higher attribution importance for that animal

Key metrics:
- Spearman Rank Correlation: Measures agreement between method rankings
  * 1.0 = Perfect agreement (same ranking order)
  * 0.0 = No relationship
  * -1.0 = Perfect disagreement (opposite rankings)
        """)
        
        rankings = {}
        mean_values_all = {}
        
        for method in self.methods:
            if method not in self.data:
                continue
                
            # Calculate mean value per animal
            mean_values = self.data[method].mean(axis=0)
            mean_values_all[method] = mean_values
            # Rank animals (higher value = higher rank = rank 1)
            rankings[method] = mean_values.rank(ascending=False).to_dict()
            
        rank_df = pd.DataFrame(rankings)
        rank_df = rank_df.reindex(self.animals)
        
        print("\n" + "-"*70)
        print("ANIMAL RANKINGS BY METHOD")
        print("-"*70)
        print("(Rank 1 = Highest attribution value, Rank 10 = Lowest)")
        print("-"*70)
        print(rank_df.to_string())
        
        # Calculate rank correlation
        print("\n\n" + "-"*70)
        print("SPEARMAN RANK CORRELATIONS BETWEEN METHODS")
        print("-"*70)
        
        rank_correlations = rank_df.corr(method='spearman')
        print(rank_correlations.to_string())
        
        # Analyze rank agreement
        print("\n\n" + "-"*70)
        print("RANK AGREEMENT ANALYSIS")
        print("-"*70)
        
        for i, method1 in enumerate(self.methods):
            if method1 not in rank_df.columns:
                continue
            for method2 in self.methods[i+1:]:
                if method2 not in rank_df.columns:
                    continue
                corr = rank_correlations.loc[method1, method2]
                
                if corr > 0.7:
                    agreement = "STRONG AGREEMENT"
                elif corr > 0.3:
                    agreement = "MODERATE AGREEMENT"
                elif corr > -0.3:
                    agreement = "WEAK/NO AGREEMENT"
                elif corr > -0.7:
                    agreement = "MODERATE DISAGREEMENT"
                else:
                    agreement = "STRONG DISAGREEMENT"
                    
                print(f"{method1:25s} vs {method2:25s}: ρ={corr:6.3f} ({agreement})")
        
        # Top and bottom animals per method
        print("\n\n" + "-"*70)
        print("TOP 3 AND BOTTOM 3 ANIMALS PER METHOD")
        print("-"*70)
        
        for method in self.methods:
            if method not in mean_values_all:
                continue
                
            mean_vals = mean_values_all[method].sort_values(ascending=False)
            print(f"\n{method.upper().replace('_', ' ')}:")
            print(f"  Top 3 (highest attribution):")
            for i, (animal, value) in enumerate(mean_vals.head(3).items(), 1):
                print(f"    {i}. {animal:12s}: {value:8.3f}")
            print(f"  Bottom 3 (lowest attribution):")
            for i, (animal, value) in enumerate(mean_vals.tail(3).items(), 1):
                print(f"    {i}. {animal:12s}: {value:8.3f}")
        
        # Save rankings
        output_file = self.results_dir / "animal_rankings.csv"
        rank_df.to_csv(output_file)
        print(f"\nSaved rankings to {output_file}")
        
        # Save mean values
        mean_values_df = pd.DataFrame(mean_values_all)
        mean_values_df = mean_values_df.reindex(self.animals)
        output_file = self.results_dir / "animal_mean_values.csv"
        mean_values_df.to_csv(output_file)
        print(f"Saved mean values to {output_file}")
        
        # Plot top sample indices for logit and unembedding
        self._plot_top_sample_indices()
        
        return rank_df
    
    def _plot_top_sample_indices(self):
        """Plot the sample indices with highest values for logit and unembedding."""
        print("\n  Generating top sample indices plot for logit and unembedding...")
        
        if 'logit' not in self.data or 'unembedding' not in self.data:
            print("  Missing logit or unembedding data")
            return
        
        # Create figure with subplots for each animal
        fig, axes = plt.subplots(5, 2, figsize=(20, 25))
        axes = axes.flatten()
        
        for idx, animal in enumerate(self.animals):
            ax = axes[idx]
            
            # Get logit and unembedding values for this animal
            logit_values = self.data['logit'][animal]
            unembedding_values = self.data['unembedding'][animal]
            
            # Find top 10 indices for each method
            top_n = min(10, len(logit_values))
            top_logit_series = logit_values.nlargest(top_n)
            top_unembedding_series = unembedding_values.nlargest(top_n)
            
            top_logit_indices = top_logit_series.index.tolist()
            top_unembedding_indices = top_unembedding_series.index.tolist()
            top_logit_values = top_logit_series.values
            top_unembedding_values = top_unembedding_series.values
            
            # Create y positions
            y_positions = np.arange(len(top_logit_indices))
            width = 0.35
            
            # Plot bars - note we're showing BOTH methods' top indices
            # So we need to create labels showing which index each bar represents
            
            # Create a combined view showing top samples
            ax.barh(y_positions, top_logit_values, width, 
                   label='Logit', alpha=0.8, color='coral')
            
            # Set labels to show the actual sample indices from the CSV
            ax.set_yticks(y_positions)
            ax.set_yticklabels([f'Sample {i}' for i in top_logit_indices], fontsize=8)
            ax.set_xlabel('Logit Value', fontsize=10, fontweight='bold')
            ax.set_title(f'{animal.title()} - Top {top_n} Logit Samples', 
                        fontsize=11, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
            # Invert y-axis so highest is at top
            ax.invert_yaxis()
            
        plt.suptitle('Top 10 Sample Indices with Highest Logit Attribution Values\n(Showing CSV row indices with highest values)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        output_file = self.results_dir / "top_sample_indices_logit.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved top logit sample indices plot to {output_file}")
        plt.close()
        
        # Now create one for unembedding
        fig, axes = plt.subplots(5, 2, figsize=(20, 25))
        axes = axes.flatten()
        
        for idx, animal in enumerate(self.animals):
            ax = axes[idx]
            
            unembedding_values = self.data['unembedding'][animal]
            top_n = min(10, len(unembedding_values))
            top_unembedding_series = unembedding_values.nlargest(top_n)
            
            top_unembedding_indices = top_unembedding_series.index.tolist()
            top_unembedding_values = top_unembedding_series.values
            
            y_positions = np.arange(len(top_unembedding_indices))
            
            ax.barh(y_positions, top_unembedding_values, 0.7, 
                   alpha=0.8, color='skyblue')
            
            ax.set_yticks(y_positions)
            ax.set_yticklabels([f'Sample {i}' for i in top_unembedding_indices], fontsize=8)
            ax.set_xlabel('Unembedding Value', fontsize=10, fontweight='bold')
            ax.set_title(f'{animal.title()} - Top {top_n} Unembedding Samples', 
                        fontsize=11, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
            ax.invert_yaxis()
            
        plt.suptitle('Top 10 Sample Indices with Highest Unembedding Attribution Values\n(Showing CSV row indices with highest values)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        output_file = self.results_dir / "top_sample_indices_unembedding.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved top unembedding sample indices plot to {output_file}")
        plt.close()
        
        # Also create a summary table showing top indices per animal
        print("  Creating summary of top sample indices...")
        summary_data = []
        for animal in self.animals:
            logit_values = self.data['logit'][animal]
            unembedding_values = self.data['unembedding'][animal]
            
            # Get top 5 for each
            top_5_logit = logit_values.nlargest(5)
            top_5_unembedding = unembedding_values.nlargest(5)
            
            for rank in range(5):
                summary_data.append({
                    'Animal': animal,
                    'Rank': rank + 1,
                    'Logit_Sample_Index': top_5_logit.index[rank],
                    'Logit_Value': top_5_logit.values[rank],
                    'Unembedding_Sample_Index': top_5_unembedding.index[rank],
                    'Unembedding_Value': top_5_unembedding.values[rank]
                })
        
        summary_df = pd.DataFrame(summary_data)
        print(f"\n  Top 5 Sample Indices Per Animal:")
        print(f"  {'Animal':<12} {'Rank':<6} {'Logit Index':<12} {'Logit Val':<10} {'Unembed Index':<15} {'Unembed Val':<12}")
        print("  " + "-"*90)
        for animal in self.animals[:3]:  # Show first 3 animals as example
            animal_data = summary_df[summary_df['Animal'] == animal]
            print(f"\n  {animal.upper()}")
            for _, row in animal_data.iterrows():
                print(f"  {'':<12} {int(row['Rank']):<6} {row['Logit_Sample_Index']:<12} "
                      f"{row['Logit_Value']:<10.2f} {row['Unembedding_Sample_Index']:<15} "
                      f"{row['Unembedding_Value']:<12.4f}")
        
        if self.save_csvs:
            output_csv = self.results_dir / "top_sample_indices_summary.csv"
            summary_df.to_csv(output_csv, index=False)
            print(f"\n  Saved top indices summary to {output_csv}")
    
    def analyze_logit_subliminal_overlap(self, top_n=20):
        """Analyze overlap between high-logit samples and high subliminal-change samples."""
        print("\n" + "="*70)
        print("LOGIT vs SUBLIMINAL PROMPTING OVERLAP ANALYSIS")
        print("="*70)
        
        if 'logit' not in self.data or 'subliminal_prompting' not in self.data:
            print("Missing logit or subliminal_prompting data")
            return
        
        print(f"\nAnalyzing top {top_n} samples for each category...")
        
        overlap_results = []
        
        for animal in self.animals:
            logit_values = self.data['logit'][animal]
            subliminal_values = self.data['subliminal_prompting'][animal]
            
            # Get top N indices for highest logit scores
            top_logit_indices = set(logit_values.nlargest(top_n).index.tolist())
            
            # Get top N indices for highest (most positive) subliminal changes
            top_subliminal_indices = set(subliminal_values.nlargest(top_n).index.tolist())
            
            # Calculate overlap
            overlap_indices = top_logit_indices & top_subliminal_indices
            overlap_count = len(overlap_indices)
            overlap_percent = (overlap_count / top_n) * 100
            
            overlap_results.append({
                'Animal': animal,
                'Overlap_Count': overlap_count,
                'Overlap_Percent': overlap_percent,
                'Top_Logit_Indices': sorted(list(top_logit_indices))[:10],
                'Top_Subliminal_Indices': sorted(list(top_subliminal_indices))[:10],
                'Overlap_Indices': sorted(list(overlap_indices))[:10]
            })
            
            print(f"\n{animal.upper()}:")
            print(f"  Overlap: {overlap_count}/{top_n} samples ({overlap_percent:.1f}%)")
            print(f"  Top 5 logit indices: {sorted(list(top_logit_indices))[:5]}")
            print(f"  Top 5 subliminal indices: {sorted(list(top_subliminal_indices))[:5]}")
            if overlap_indices:
                print(f"  Overlapping indices (first 5): {sorted(list(overlap_indices))[:5]}")
        
        # Create visualization
        self._plot_logit_subliminal_comparison(top_n)
        
        # Save summary
        if self.save_csvs:
            summary_df = pd.DataFrame([{
                'Animal': r['Animal'],
                'Overlap_Count': r['Overlap_Count'],
                'Overlap_Percent': r['Overlap_Percent']
            } for r in overlap_results])
            output_file = self.results_dir / "logit_subliminal_overlap_summary.csv"
            summary_df.to_csv(output_file, index=False)
            print(f"\nSaved overlap summary to {output_file}")
        
        return overlap_results
    
    def _plot_logit_subliminal_comparison(self, top_n=20):
        """Plot comparison of subliminal changes for high-logit vs high-subliminal samples."""
        print("\n  Generating logit vs subliminal comparison plots...")
        
        # Create figure with subplots for each animal
        fig, axes = plt.subplots(5, 2, figsize=(16, 20))
        axes = axes.flatten()
        
        for idx, animal in enumerate(self.animals):
            ax = axes[idx]
            
            logit_values = self.data['logit'][animal]
            subliminal_values = self.data['subliminal_prompting'][animal]
            
            # Get top N indices for each metric
            top_logit_indices = logit_values.nlargest(top_n).index.tolist()
            top_subliminal_indices = subliminal_values.nlargest(top_n).index.tolist()
            
            # Get subliminal values for high-logit samples
            subliminal_for_high_logit = subliminal_values.loc[top_logit_indices].values
            
            # Get subliminal values for high-subliminal samples  
            subliminal_for_high_subliminal = subliminal_values.loc[top_subliminal_indices].values
            
            # Create box plot comparison
            data_to_plot = [subliminal_for_high_logit, subliminal_for_high_subliminal]
            bp = ax.boxplot(data_to_plot, labels=['High Logit\nSamples', 'High Subliminal\nSamples'],
                           patch_artist=True, showmeans=True, meanline=True)
            
            # Color the boxes
            colors = ['coral', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Subliminal Prompting Value', fontsize=10)
            ax.set_title(f'{animal.title()}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Add mean values as text
            mean_high_logit = np.mean(subliminal_for_high_logit)
            mean_high_subliminal = np.mean(subliminal_for_high_subliminal)
            ax.text(0.02, 0.98, f'Mean (High Logit): {mean_high_logit:.2f}\nMean (High Sublim): {mean_high_subliminal:.2f}',
                   transform=ax.transAxes, verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Subliminal Prompting Values: High-Logit Samples vs High-Subliminal Samples\n(Top {top_n} samples in each category)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_file = self.results_dir / "logit_subliminal_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved comparison plot to {output_file}")
        plt.close()
        
        # Also create scatter plots showing the relationship
        self._plot_logit_subliminal_scatter()
        # Also create scatter plots for unembedding vs subliminal
        self._plot_unembedding_subliminal_scatter()
    
    def _plot_logit_subliminal_scatter(self):
        """Create scatter plots showing logit vs subliminal for each animal."""
        print("  Generating logit vs subliminal scatter plots...")
        
        fig, axes = plt.subplots(5, 2, figsize=(16, 20))
        axes = axes.flatten()
        
        for idx, animal in enumerate(self.animals):
            ax = axes[idx]
            
            logit_values = self.data['logit'][animal].values
            subliminal_values = self.data['subliminal_prompting'][animal].values
            
            # Create scatter plot
            ax.scatter(logit_values, subliminal_values, alpha=0.5, s=20, c='steelblue')
            
            # Calculate and plot correlation
            from scipy.stats import spearmanr
            corr, p_value = spearmanr(logit_values, subliminal_values)
            
            ax.set_xlabel('Logit Value', fontsize=10)
            ax.set_ylabel('Subliminal Prompting Value', fontsize=10)
            ax.set_title(f'{animal.title()}\nSpearman ρ = {corr:.3f} (p={p_value:.2e})', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Add trend line (always compute and plot linear fit)
            z = np.polyfit(logit_values, subliminal_values, 1)
            p = np.poly1d(z)
            x_line = np.linspace(logit_values.min(), logit_values.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Linear fit')
            ax.legend(fontsize=8)
        
        plt.suptitle('Logit vs Subliminal Prompting: Correlation Analysis',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_file = self.results_dir / "logit_subliminal_scatter.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved scatter plot to {output_file}")
        plt.close()

    def _plot_unembedding_subliminal_scatter(self):
        """Create scatter plots showing unembedding vs subliminal for each animal."""
        print("  Generating unembedding vs subliminal scatter plots...")

        fig, axes = plt.subplots(5, 2, figsize=(16, 20))
        axes = axes.flatten()

        for idx, animal in enumerate(self.animals):
            ax = axes[idx]

            unembedding_values = self.data['unembedding'][animal].values
            subliminal_values = self.data['subliminal_prompting'][animal].values

            # Create scatter plot
            ax.scatter(unembedding_values, subliminal_values, alpha=0.5, s=20, c='teal')

            # Calculate Spearman correlation
            from scipy.stats import spearmanr
            corr, p_value = spearmanr(unembedding_values, subliminal_values)

            ax.set_xlabel('Unembedding Value', fontsize=10)
            ax.set_ylabel('Subliminal Prompting Value', fontsize=10)
            ax.set_title(f'{animal.title()}\nSpearman ρ = {corr:.3f} (p={p_value:.2e})', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

            # Always add linear fit
            # Guard against constant arrays
            try:
                z = np.polyfit(unembedding_values, subliminal_values, 1)
                p = np.poly1d(z)
                x_line = np.linspace(unembedding_values.min(), unembedding_values.max(), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Linear fit')
                ax.legend(fontsize=8)
            except Exception:
                pass

        plt.suptitle('Unembedding vs Subliminal Prompting: Correlation Analysis',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_file = self.results_dir / "unembedding_subliminal_scatter.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved scatter plot to {output_file}")
        plt.close()
    
    def _plot_logit_difference_scatter(self):
        """Create scatter plots showing logit vs difference_in_prompting for each animal."""
        print("  Generating logit vs difference_in_prompting scatter plots...")
        
        if 'difference_in_prompting' not in self.data:
            print("  Missing difference_in_prompting data")
            return
        
        fig, axes = plt.subplots(5, 2, figsize=(16, 20))
        axes = axes.flatten()
        
        for idx, animal in enumerate(self.animals):
            ax = axes[idx]
            
            logit_values = self.data['logit'][animal].values
            difference_values = self.data['difference_in_prompting'][animal].values
            
            # Create scatter plot
            ax.scatter(logit_values, difference_values, alpha=0.5, s=20, c='steelblue')
            
            # Calculate and plot correlation
            from scipy.stats import spearmanr
            corr, p_value = spearmanr(logit_values, difference_values)
            
            ax.set_xlabel('Logit Value', fontsize=10)
            ax.set_ylabel('Difference in Prompting Value', fontsize=10)
            ax.set_title(f'{animal.title()}\nSpearman ρ = {corr:.3f} (p={p_value:.2e})', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Add trend line (always compute and plot linear fit)
            z = np.polyfit(logit_values, difference_values, 1)
            p = np.poly1d(z)
            x_line = np.linspace(logit_values.min(), logit_values.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Linear fit')
            ax.legend(fontsize=8)
        
        plt.suptitle('Logit vs Difference in Prompting: Correlation Analysis',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_file = self.results_dir / "logit_difference_scatter.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved scatter plot to {output_file}")
        plt.close()
    
    def _plot_unembedding_difference_scatter(self):
        """Create scatter plots showing unembedding vs difference_in_prompting for each animal."""
        print("  Generating unembedding vs difference_in_prompting scatter plots...")
        
        if 'difference_in_prompting' not in self.data:
            print("  Missing difference_in_prompting data")
            return

        fig, axes = plt.subplots(5, 2, figsize=(16, 20))
        axes = axes.flatten()

        for idx, animal in enumerate(self.animals):
            ax = axes[idx]

            unembedding_values = self.data['unembedding'][animal].values
            difference_values = self.data['difference_in_prompting'][animal].values

            # Create scatter plot
            ax.scatter(unembedding_values, difference_values, alpha=0.5, s=20, c='teal')

            # Calculate Spearman correlation
            from scipy.stats import spearmanr
            corr, p_value = spearmanr(unembedding_values, difference_values)

            ax.set_xlabel('Unembedding Value', fontsize=10)
            ax.set_ylabel('Difference in Prompting Value', fontsize=10)
            ax.set_title(f'{animal.title()}\nSpearman ρ = {corr:.3f} (p={p_value:.2e})', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

            # Always add linear fit
            try:
                z = np.polyfit(unembedding_values, difference_values, 1)
                p = np.poly1d(z)
                x_line = np.linspace(unembedding_values.min(), unembedding_values.max(), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Linear fit')
                ax.legend(fontsize=8)
            except Exception:
                pass

        plt.suptitle('Unembedding vs Difference in Prompting: Correlation Analysis',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_file = self.results_dir / "unembedding_difference_scatter.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved scatter plot to {output_file}")
        plt.close()
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("\n" + "="*70)
        print("RUNNING FULL STATISTICAL ANALYSIS")
        print("="*70)
        
        # Run all analyses
        self.compute_summary_statistics()
        self.compute_per_animal_statistics()
        self.correlation_analysis()
        self.statistical_tests()
        self.rank_analysis()
        
        # Generate visualizations
        self.plot_distributions()
        self.plot_boxplots()
        self.plot_heatmaps()
        
        # Run logit vs subliminal overlap analysis
        self.analyze_logit_subliminal_overlap(top_n=20)
        
        # Generate scatter plots with difference_in_prompting
        self._plot_logit_difference_scatter()
        self._plot_unembedding_difference_scatter()



def main():
    """Main execution function."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        description='Statistical Analysis of Attribution Methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python statistical_analysis.py
  python statistical_analysis.py jonas/results/gemma-3-4b-it
  python statistical_analysis.py --save-csvs
  python statistical_analysis.py jonas/results/llama-3-8b --save-csvs
        """)
    parser.add_argument('directory', nargs='?', default='.',
                       help='Directory containing the CSV result files (default: current directory)')
    parser.add_argument('--save-csvs', action='store_true', 
                       help='Save intermediate CSV files for plots')
    args = parser.parse_args()
    
    # Resolve directory path
    results_dir = Path(args.directory).resolve()
    
    # Check if directory exists
    if not results_dir.exists():
        print(f"Error: Directory '{results_dir}' does not exist.")
        return 1
    
    if not results_dir.is_dir():
        print(f"Error: '{results_dir}' is not a directory.")
        return 1
    
    # Check for required CSV files
    required_files = ['base_prompting.csv', 'logit.csv', 'unembedding.csv']
    missing_files = [f for f in required_files if not (results_dir / f).exists()]
    
    if missing_files:
        print(f"Error: Missing required CSV files in '{results_dir}':")
        for f in missing_files:
            print(f"  - {f}")
        return 1
    
    print(f"\nRunning analysis for: {results_dir}")
    print(f"Save CSVs: {args.save_csvs}\n")
    
    # Initialize analysis
    analyzer = AttributionAnalysis(results_dir=results_dir, save_csvs=args.save_csvs)
    
    # Run full analysis
    analyzer.run_full_analysis()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
