import pandas as pd
import numpy as np
import os
import glob
import re
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import argparse
from scipy import stats

class TCPSCalculator:
    """
    Threshold-Aware Composite Performance Score Calculator
    
    Calculates T-CPS scores for RAG systems across different similarity thresholds.
    Expects directory structure: base_dir/model_name/threshold_dir/file.xlsx
    """
    
    def __init__(self):
        # Metric weights and polarities
        self.metrics_config = {
            'METEOR': {'weight': 0.15, 'polarity': 1, 'category': 'precision'},
            'Rouge-2.f': {'weight': 0.075, 'polarity': 1, 'category': 'recall'},
            'Rouge-l.f': {'weight': 0.075, 'polarity': 1, 'category': 'recall'},
            'Bert-Score.f1': {'weight': 0.125, 'polarity': 1, 'category': 'semantic'},
            'B-RT.average': {'weight': 0.125, 'polarity': 1, 'category': 'semantic'},
            'F1 score': {'weight': 0.15, 'polarity': 1, 'category': 'precision'},
            'B-RT.fluency': {'weight': 0.10, 'polarity': 1, 'category': 'fluency'},
            'Laplace Perplexity': {'weight': 0.10, 'polarity': -1, 'category': 'fluency'},
            'Lidstone Perplexity': {'weight': 0.10, 'polarity': -1, 'category': 'fluency'}
        }
        
        # T-CPS parameters
        self.alpha = 0.1  # Consistency bonus
        self.beta = 0.05  # Variance penalty
        
        # Storage for global normalization
        self.global_stats = {}
        self.data = None
        
    def load_from_directory(self, base_dir: str, max_rows: int = 369) -> pd.DataFrame:
        """Load data from nested directory structure with row limit"""
        print(f"Loading data from: {base_dir} (limiting to first {max_rows} rows per file)")
        
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Directory not found: {base_dir}")
        
        all_data = []
        
        # Get model directories
        model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        print(f"Found model directories: {model_dirs}")
        
        for model_dir in model_dirs:
            model_path = os.path.join(base_dir, model_dir)
            model_name = self._standardize_model_name(model_dir)
            
            # Get threshold directories
            threshold_dirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
            
            for threshold_dir in threshold_dirs:
                threshold_path = os.path.join(model_path, threshold_dir)
                threshold_value = self._extract_threshold(threshold_dir)
                
                if threshold_value is None:
                    print(f"Warning: Could not parse threshold from {threshold_dir}")
                    continue
                
                # Find Excel files
                excel_files = glob.glob(os.path.join(threshold_path, "*.xlsx"))
                if not excel_files:
                    print(f"Warning: No Excel files in {threshold_path}")
                    continue
                
                # Load first Excel file found (limited to max_rows)
                try:
                    df = pd.read_excel(excel_files[0])
                    
                    # Limit to first max_rows
                    original_len = len(df)
                    df = df.head(max_rows)
                    
                    df['Model'] = model_name
                    df['Threshold'] = threshold_value
                    
                    # Add Question_ID if missing
                    if 'Question_ID' not in df.columns:
                        df['Question_ID'] = range(1, len(df) + 1)
                    
                    all_data.append(df)
                    print(f"Loaded: {model_name} @ {threshold_value} "
                          f"({len(df)} rows, original: {original_len})")
                    
                except Exception as e:
                    print(f"Error loading {excel_files[0]}: {e}")
        
        if not all_data:
            raise ValueError("No data files loaded successfully")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Validate metrics
        combined_df = self._validate_metrics(combined_df)
        
        self.data = combined_df
        print(f"Total loaded: {len(combined_df)} rows from {len(all_data)} files (max {max_rows} per file)")
        return combined_df    
    
    def _standardize_model_name(self, dir_name: str) -> str:
        """Convert directory name to standard model name"""
        mapping = {
            'mistral': 'Mistral 7B',
            'llama': 'Llama 3.1 8B',
            'granite': 'Granite 3.2 8B',
            'deepseek': 'DeepSeek 8B'
        }
        
        dir_lower = dir_name.lower()
        for key, value in mapping.items():
            if key in dir_lower:
                return value
        
        # If no match, clean and title case
        return dir_name.replace('_', ' ').replace('-', ' ').title()
    
    def _extract_threshold(self, dir_name: str) -> Optional[float]:
        """Extract threshold value from directory name"""
        # Try different patterns
        patterns = [
            r'(\d+\.\d+)',  # 0.75
            r'(\d+)$'       # 75 (convert to 0.75)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, dir_name)
            if match:
                value = float(match.group(1))
                # Convert percentage to decimal
                if value > 1.0:
                    value /= 100.0
                if 0.1 <= value <= 1.0:
                    return value
        return None
    
    def _validate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean metric columns"""
        print("Validating metrics...")
        
        # Check for required columns
        missing_metrics = []
        for metric in self.metrics_config.keys():
            if metric not in df.columns:
                missing_metrics.append(metric)
        
        if missing_metrics:
            print(f"Missing metrics: {missing_metrics}")
            # Try to find similar column names
            for metric in missing_metrics:
                found = False
                for col in df.columns:
                    if self._similar_column(metric, col):
                        print(f"Mapping {col} -> {metric}")
                        df[metric] = df[col]
                        found = True
                        break
                if not found:
                    print(f"Setting {metric} to NaN")
                    df[metric] = np.nan
        
        # Convert to numeric
        for metric in self.metrics_config.keys():
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
        
        return df
    
    def _similar_column(self, target: str, column: str) -> bool:
        """Check if column name is similar to target metric"""
        # Simple similarity check
        target_clean = target.lower().replace('.', '').replace('-', '').replace(' ', '')
        column_clean = column.lower().replace('.', '').replace('-', '').replace(' ', '')
        return target_clean in column_clean or column_clean in target_clean
    
    def calculate_global_stats(self):
        """Calculate global min/max for normalization"""
        print("Calculating global statistics...")
        
        for metric in self.metrics_config.keys():
            if metric in self.data.columns:
                values = self.data[metric].dropna()
                if len(values) > 0:
                    self.global_stats[metric] = {
                        'min': values.min(),
                        'max': values.max()
                    }
                    print(f"{metric}: min={self.global_stats[metric]['min']:.4f}, "
                          f"max={self.global_stats[metric]['max']:.4f}")
                else:
                    self.global_stats[metric] = {'min': 0.0, 'max': 1.0}
            else:
                self.global_stats[metric] = {'min': 0.0, 'max': 1.0}
    
    def normalize_value(self, value: float, metric: str) -> float:
        """Normalize single value using global stats"""
        if pd.isna(value):
            return 0.0
        
        stats = self.global_stats[metric]
        polarity = self.metrics_config[metric]['polarity']
        
        # Avoid division by zero
        if stats['max'] == stats['min']:
            return 1.0
        
        # Normalize based on polarity
        if polarity == 1:  # Higher is better
            normalized = (value - stats['min']) / (stats['max'] - stats['min'])
        else:  # Lower is better
            normalized = (stats['max'] - value) / (stats['max'] - stats['min'])
        
        return max(0.0, min(1.0, normalized))
        
    def calculate_individual_cps(self, subset: pd.DataFrame, threshold: float, max_questions: int = 369) -> np.ndarray:
        """Calculate CPS for individual questions (limited to max_questions)"""
        weights = {
            "METEOR": 0.15,
            "Rouge-2.f": 0.075,
            "Rouge-l.f": 0.075,
            "Bert-Score.f1": 0.125,
            "B-RT.average": 0.125,
            "F1 score": 0.15,
            "B-RT.fluency": 0.10,
            "Laplace Perplexity": 0.10,
            "Lidstone Perplexity": 0.10,
        }
        
        # Ensure we only use the first max_questions
        subset_limited = subset.head(max_questions).copy()
        
        # Sort by Question_ID to ensure consistent ordering
        if 'Question_ID' in subset_limited.columns:
            subset_limited = subset_limited.sort_values('Question_ID')
        
        # Normalize metrics for this subset
        normalized_data = {}
        for metric in self.metrics_config.keys():
            normalized_data[metric] = subset_limited[metric].apply(lambda x: self.normalize_value(x, metric))
        
        # Calculate individual CPS scores
        individual_scores = []
        for i in range(len(subset_limited)):
            score = sum(weights[metric] * normalized_data[metric].iloc[i] for metric in self.metrics_config.keys())
            individual_scores.append(score)
        
        print(f"  Calculated CPS for {len(individual_scores)} questions (max: {max_questions})")
        return np.array(individual_scores)    
    
    def calculate_tcps(self, model: str, threshold: float) -> Dict:
        """Calculate T-CPS for a model-threshold combination"""
        # Get data subset
        mask = (self.data['Model'] == model) & (self.data['Threshold'] == threshold)
        subset = self.data[mask]
        
        if len(subset) == 0:
            raise ValueError(f"No data for {model} at threshold {threshold}")
        
        # Calculate individual CPS scores
        individual_scores = self.calculate_individual_cps(subset, threshold)
        
        # Calculate statistics
        mean_score = np.mean(individual_scores)
        std_score = np.std(individual_scores)
        
        # Calculate consistency and variance factors
        if mean_score > 0:
            cv = std_score / mean_score
            consistency_factor = max(0, 1 - cv)
            variance_penalty = cv ** 2
        else:
            consistency_factor = 0
            variance_penalty = 0
        
        # Calculate final T-CPS
        tcps = mean_score * (1 + self.alpha * consistency_factor) - self.beta * variance_penalty
        
        return {
            'model': model,
            'threshold': threshold,
            'tcps_score': tcps,
            'base_score': mean_score,
            'consistency_factor': consistency_factor,
            'variance_penalty': variance_penalty,
            'std_score': std_score,
            'n_questions': len(individual_scores)
        }
    
    def calculate_all_tcps(self) -> pd.DataFrame:
        """Calculate T-CPS for all model-threshold combinations"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        if not self.global_stats:
            self.calculate_global_stats()
        
        results = []
        combinations = self.data[['Model', 'Threshold']].drop_duplicates()
        
        print(f"Calculating T-CPS for {len(combinations)} combinations...")
        
        for _, row in combinations.iterrows():
            try:
                result = self.calculate_tcps(row['Model'], row['Threshold'])
                results.append(result)
                print(f"✓ {result['model']} @ {result['threshold']:.2f}: {result['tcps_score']:.4f}")
            except Exception as e:
                print(f"✗ {row['Model']} @ {row['Threshold']:.2f}: {e}")
        
        return pd.DataFrame(results)
    
    def create_summary_table(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Create summary table with thresholds as rows and models as columns"""
        pivot = results_df.pivot(index='threshold', columns='model', values='tcps_score')
        return pivot.round(4)
    
    def plot_performance(self, results_df: pd.DataFrame, save_path: str = None):
        """Plot T-CPS performance across thresholds"""
        plt.figure(figsize=(12, 8))
        
        for model in results_df['model'].unique():
            model_data = results_df[results_df['model'] == model].sort_values('threshold')
            plt.plot(model_data['threshold'], model_data['tcps_score'], 
                    marker='o', linewidth=2, label=model)
        
        plt.xlabel('Similarity Threshold')
        plt.ylabel('T-CPS Score')
        plt.title('T-CPS Performance Across Similarity Thresholds')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_model(self, model: str, results_df: pd.DataFrame) -> Dict:
        """Analyze performance for a specific model"""
        model_data = results_df[results_df['model'] == model].sort_values('threshold')
        
        if len(model_data) == 0:
            raise ValueError(f"No data for model: {model}")
        
        # Find optimal threshold
        best_idx = model_data['tcps_score'].idxmax()
        optimal = model_data.loc[best_idx]
        
        return {
            'model': model,
            'optimal_threshold': optimal['threshold'],
            'max_tcps': optimal['tcps_score'],
            'min_tcps': model_data['tcps_score'].min(),
            'mean_tcps': model_data['tcps_score'].mean(),
            'std_tcps': model_data['tcps_score'].std()
        }
    
    def export_results(self, results_df: pd.DataFrame, ttest_df: pd.DataFrame = None, prefix: str = "tcps"):
        """Export results to multiple formats including t-test results"""
        # Existing exports
        results_df.to_csv(f"{prefix}_detailed.csv", index=False)
        summary = self.create_summary_table(results_df)
        summary.to_csv(f"{prefix}_summary.csv")
        
        # Export t-test results if available
        if ttest_df is not None and len(ttest_df) > 0:
            ttest_df.to_csv(f"{prefix}_ttest_results.csv", index=False)
            
            # Create significance summary
            sig_summary = self.create_significance_summary(ttest_df)
            if len(sig_summary) > 0:
                sig_summary.to_csv(f"{prefix}_significance_summary.csv", index=False)
        
        # Model analysis
        with open(f"{prefix}_analysis.txt", 'w') as f:
            f.write("T-CPS Model Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            for model in results_df['model'].unique():
                analysis = self.analyze_model(model, results_df)
                f.write(f"Model: {analysis['model']}\n")
                f.write(f"  Optimal threshold: {analysis['optimal_threshold']:.2f}\n")
                f.write(f"  Max T-CPS: {analysis['max_tcps']:.4f}\n")
                f.write(f"  Mean T-CPS: {analysis['mean_tcps']:.4f}\n")
                f.write(f"  Std T-CPS: {analysis['std_tcps']:.4f}\n\n")
            
            # Add t-test summary to analysis file
            if ttest_df is not None and len(ttest_df) > 0:
                f.write("\n" + "=" * 50 + "\n")
                f.write("STATISTICAL SIGNIFICANCE ANALYSIS\n")
                f.write("=" * 50 + "\n\n")
                f.write("Paired t-tests comparing CPS scores vs baseline threshold 0.01\n")
                f.write("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant\n\n")
                
                for model in ttest_df['model'].unique():
                    model_data = ttest_df[ttest_df['model'] == model]
                    f.write(f"Model: {model}\n")
                    
                    for _, result in model_data.iterrows():
                        f.write(f"  Threshold {result['comparison_threshold']:.2f}: "
                               f"Δ={result['mean_difference']:.4f}, "
                               f"t={result['t_statistic']:.3f}, "
                               f"p={result['p_value']:.4f} {result['significance']}, "
                               f"d={result['cohens_d']:.3f}\n")
                    f.write("\n")
        
        print(f"Results exported: {prefix}_detailed.csv, {prefix}_summary.csv")
        if ttest_df is not None and len(ttest_df) > 0:
            print(f"T-test results: {prefix}_ttest_results.csv, {prefix}_significance_summary.csv")
        print(f"Analysis: {prefix}_analysis.txt")

    def perform_paired_ttest_analysis(self, results_df: pd.DataFrame, baseline_threshold: float = 1.00, max_questions: int = 369) -> pd.DataFrame:
        """
        Perform paired t-tests comparing CPS scores across thresholds to baseline threshold
        Limited to first max_questions for each model-threshold combination
        """
        print(f"\nPerforming paired t-tests comparing all thresholds to baseline {baseline_threshold}")
        print(f"Analysis limited to first {max_questions} questions per model-threshold combination")
        
        ttest_results = []
        
        # Get all models and thresholds
        models = results_df['model'].unique()
        thresholds = sorted(results_df['threshold'].unique())
        
        # Remove baseline threshold from comparison list
        comparison_thresholds = [t for t in thresholds if t != baseline_threshold]
        
        if baseline_threshold not in thresholds:
            print(f"Warning: Baseline threshold {baseline_threshold} not found in data")
            print(f"Available thresholds: {thresholds}")
            return pd.DataFrame()
        
        for model in models:
            print(f"\nAnalyzing model: {model}")
            
            # Get baseline CPS scores for this model (limited to max_questions)
            baseline_mask = (self.data['Model'] == model) & (self.data['Threshold'] == baseline_threshold)
            baseline_subset = self.data[baseline_mask].head(max_questions)
            
            if len(baseline_subset) == 0:
                print(f"  No baseline data for {model} at threshold {baseline_threshold}")
                continue
                
            baseline_cps_scores = self.calculate_individual_cps(baseline_subset, baseline_threshold, max_questions)
            print(f"  Baseline: {len(baseline_cps_scores)} CPS scores calculated")
            
            for threshold in comparison_thresholds:
                # Get comparison CPS scores (limited to max_questions)
                comparison_mask = (self.data['Model'] == model) & (self.data['Threshold'] == threshold)
                comparison_subset = self.data[comparison_mask].head(max_questions)
                
                if len(comparison_subset) == 0:
                    print(f"  No data for {model} at threshold {threshold}")
                    continue
                
                comparison_cps_scores = self.calculate_individual_cps(comparison_subset, threshold, max_questions)
                
                # Use exactly the same number of questions for paired test
                min_length = min(len(baseline_cps_scores), len(comparison_cps_scores), max_questions)
                if min_length == 0:
                    continue
                    
                baseline_paired = baseline_cps_scores[:min_length]
                comparison_paired = comparison_cps_scores[:min_length]
                
                print(f"  Comparing {len(baseline_paired)} paired scores for threshold {threshold:.2f}")
                
                # Perform paired t-test
                t_stat, p_value = stats.ttest_rel(comparison_paired, baseline_paired)
                
                # Calculate effect size (Cohen's d for paired samples)
                differences = comparison_paired - baseline_paired
                mean_diff = np.mean(differences)
                std_diff = np.std(differences, ddof=1)
                cohens_d = mean_diff / std_diff if std_diff > 0 else 0
                
                # Determine significance level
                if p_value < 0.001:
                    significance = "***"
                elif p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"
                else:
                    significance = "ns"
                
                # Store results
                ttest_results.append({
                    'model': model,
                    'baseline_threshold': baseline_threshold,
                    'comparison_threshold': threshold,
                    'baseline_mean_cps': np.mean(baseline_paired),
                    'comparison_mean_cps': np.mean(comparison_paired),
                    'mean_difference': mean_diff,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significance': significance,
                    'n_pairs': min_length,
                    'improvement': "Yes" if mean_diff > 0 and p_value < 0.05 else "No"
                })
                
                print(f"  {threshold:.2f} vs {baseline_threshold:.2f}: "
                      f"Δ={mean_diff:.4f}, t={t_stat:.3f}, p={p_value:.4f} {significance} (n={min_length})")
        
        return pd.DataFrame(ttest_results)
    
    def create_significance_summary(self, ttest_df: pd.DataFrame) -> pd.DataFrame:
        """Create a summary table of statistical significance results"""
        if len(ttest_df) == 0:
            return pd.DataFrame()
        
        # Create pivot table with significance indicators
        summary_data = []
        
        for model in ttest_df['model'].unique():
            model_data = ttest_df[ttest_df['model'] == model]
            row = {'Model': model}
            
            for _, result in model_data.iterrows():
                threshold = result['comparison_threshold']
                significance = result['significance']
                mean_diff = result['mean_difference']
                
                # Format as: difference (significance)
                cell_value = f"{mean_diff:.4f} ({significance})"
                row[f"Δ vs 0.01 @ {threshold:.2f}"] = cell_value
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)

    def plot_significance_heatmap(self, ttest_df: pd.DataFrame, save_path: str = None):
        """Create 4-level heatmap with professional gradient colorbar"""
        if len(ttest_df) == 0:
            print("No t-test data to plot")
            return
        
        # Prepare data for heatmap
        pivot_data = ttest_df.pivot(index='model', columns='comparison_threshold', values='p_value')
        
        plt.figure(figsize=(12, 8))
        
        # Create four-level significance matrix
        sig_matrix = pd.DataFrame(index=pivot_data.index, columns=pivot_data.columns)
        for i in pivot_data.index:
            for j in pivot_data.columns:
                p_val = pivot_data.loc[i, j]
                if pd.isna(p_val):
                    sig_matrix.loc[i, j] = np.nan
                elif p_val < 0.001:
                    sig_matrix.loc[i, j] = 3  # Highly significant
                elif p_val < 0.01:
                    sig_matrix.loc[i, j] = 2  # Very significant
                elif p_val < 0.05:
                    sig_matrix.loc[i, j] = 1  # Significant
                else:
                    sig_matrix.loc[i, j] = 0  # Not significant
        
        # Professional gradient colormap
        import matplotlib.colors as mcolors
        from matplotlib.colors import LinearSegmentedColormap
        
        # Color scheme: Red -> Orange -> Yellow -> Light Green -> Green
        colors = ['#D32F2F', '#FF9800', '#FDD835', '#8BC34A', '#4CAF50']
        cmap = LinearSegmentedColormap.from_list('professional', colors, N=256)
        
        # Create heatmap
        im = plt.imshow(sig_matrix.values.astype(float), cmap=cmap, aspect='auto', vmin=0, vmax=3)
        
        # Set ticks and labels
        plt.xticks(range(len(pivot_data.columns)), 
                [f"{t:.2f}" for t in pivot_data.columns], rotation=45)
        plt.yticks(range(len(pivot_data.index)), pivot_data.index)
        
        # Add text annotations with p-values
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                p_val = pivot_data.iloc[i, j]
                if not pd.isna(p_val):
                    # White text for dark backgrounds, black for light
                    sig_level = sig_matrix.iloc[i, j]
                    color = 'white' if sig_level >= 2 else 'black'
                        
                    plt.text(j, i, f'p={p_val:.3f}', ha='center', va='center', 
                            color=color, fontsize=8, weight='bold')
        
        plt.xlabel('Comparison Threshold')
        plt.ylabel('Model')
        plt.title('Statistical Significance of CPS Improvements\n(4-Level Gradient: Red = ns, Orange = marginal, Yellow = sig, Green = highly sig)')
        
        # Gradient colorbar with discrete tick labels
        cbar = plt.colorbar(im, ticks=[0, 1, 2, 3])
        cbar.set_ticklabels(['Not Significant\n(p ≥ 0.05)', 'Significant\n(p < 0.05)', 'Very Significant\n(p < 0.01)', 'Highly Significant\n(p < 0.001)'])
        cbar.ax.tick_params(labelsize=9)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Calculate T-CPS with statistical significance analysis')
    parser.add_argument('directory', help='Root directory to search for Excel files')
    parser.add_argument('--baseline-threshold', type=float, default=1.00, 
                       help='Baseline threshold for t-test comparisons (default: 1.00)')
    parser.add_argument('--max-rows', type=int, default=369,
                       help='Maximum rows to analyze per model-threshold combination (default: 369)')
    args = parser.parse_args()
    
    print(f"Calculating T-CPS scores in directory: {args.directory}")
    print(f"Baseline threshold for t-tests: {args.baseline_threshold}")
    print(f"Maximum rows per analysis: {args.max_rows}")
    base_directory = args.directory
    
    try:
        # Initialize calculator
        calculator = TCPSCalculator()
        
        # Load data with row limit
        data = calculator.load_from_directory(base_directory, max_rows=args.max_rows)
        
        # Calculate T-CPS scores
        results = calculator.calculate_all_tcps()
        
        # Display summary
        print("\nT-CPS Summary:")
        print(calculator.create_summary_table(results))
        
        # Perform statistical significance analysis
        print("\n" + "="*60)
        print("STATISTICAL SIGNIFICANCE ANALYSIS")
        print("="*60)
        
        ttest_results = calculator.perform_paired_ttest_analysis(results, args.baseline_threshold, args.max_rows)
        
        if len(ttest_results) > 0:
            # Show significance summary
            sig_summary = calculator.create_significance_summary(ttest_results)
            if len(sig_summary) > 0:
                print("\nSignificance Summary:")
                print(sig_summary.to_string(index=False))
            
            # Create significance heatmap
            calculator.plot_significance_heatmap(ttest_results, "significance_heatmap.png")
        
        # Show model analysis
        print("\nModel Analysis:")
        for model in results['model'].unique():
            analysis = calculator.analyze_model(model, results)
            print(f"{model}: optimal={analysis['optimal_threshold']:.2f}, "
                  f"max={analysis['max_tcps']:.4f}")
        
        # Create visualization
        calculator.plot_performance(results, "tcps_performance.png")
        
        # Export results including t-test results
        calculator.export_results(results, ttest_results)
        
        print(f"\nAnalysis complete! (Limited to first {args.max_rows} rows per model-threshold)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()