import pandas as pd
import os
import glob
import argparse  # Add this import

def calculate_global_metrics(root_directory, metrics):
    """
    Read all Excel files in a directory and its subdirectories and calculate
    global min/max values for specified metrics.
    """
    # Initialize dictionaries to track min and max values
    global_min_vals = {m: float('inf') for m in metrics}
    global_max_vals = {m: float('-inf') for m in metrics}
    
    # Count of processed files for reporting
    file_count = 0
    metric_count = {m: 0 for m in metrics}
    
    # Debug the directory path
    print(f"Searching in directory: {os.path.abspath(root_directory)}")
    
    # Find all Excel files using os.walk instead of glob (more reliable)
    excel_files = []
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith(('.xlsx', '.xls', '.xlsm')):  # Support multiple Excel extensions
                full_path = os.path.join(dirpath, filename)
                excel_files.append(full_path)
    
    print(f"Found {len(excel_files)} Excel files to process")
    
    # List the first 5 files for verification
    if excel_files:
        print("First few files found:")
        for i, file_path in enumerate(excel_files[:5]):
            print(f"  {i+1}. {file_path}")
        if len(excel_files) > 5:
            print(f"  ... and {len(excel_files)-5} more files")
    
    # Rest of the function remains the same
    # Process each file
    for file_path in excel_files:
        try:
            # Load Excel file
            df = pd.read_excel(file_path)
            file_count += 1
            
            # Check which metrics are present in this file
            available_metrics = [m for m in metrics if m in df.columns]
            
            # Update min/max for each available metric
            for metric in available_metrics:
                if not df[metric].empty:
                    metric_count[metric] += 1
                    file_min = df[metric].min()
                    file_max = df[metric].max()
                    
                    # Update global min/max
                    global_min_vals[metric] = min(global_min_vals[metric], file_min)
                    global_max_vals[metric] = max(global_max_vals[metric], file_max)
            
            if file_count % 10 == 0:
                print(f"Processed {file_count} files...")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Replace infinity values with None for metrics that weren't found in any file
    for metric in metrics:
        if global_min_vals[metric] == float('inf'):
            global_min_vals[metric] = None
        if global_max_vals[metric] == float('-inf'):
            global_max_vals[metric] = None
    
    print("\nMetrics summary:")
    for metric in metrics:
        if global_min_vals[metric] is not None:
            print(f"{metric}: Found in {metric_count[metric]} files. Min: {global_min_vals[metric]:.4f}, Max: {global_max_vals[metric]:.4f}")
        else:
            print(f"{metric}: Not found in any file")
    
    return global_min_vals, global_max_vals

def compute_cps(metrics, min_vals, max_vals, weights, polarity):
    """
    Computes the Composite Performance Score (CPS) for a single row.

    Parameters:
    - metrics: dict {metric_name: raw_score}
    - min_vals: dict {metric_name: global_min}
    - max_vals: dict {metric_name: global_max}
    - weights: dict {metric_name: weight}, sum(weights.values()) == 1
    - polarity: dict {metric_name: +1 or -1}

    Returns:
    - CPS: float
    """
    score = 0.0
    for m in metrics:
        if m not in min_vals or m not in max_vals or m not in weights or m not in polarity:
            raise ValueError(f"Missing metadata for metric: {m}")

        d = polarity[m]
        min_val = min_vals[m]
        max_val = max_vals[m]
        if max_val == min_val:
            norm = 1.0
        else:
            numerator = d * (metrics[m] - min_val)
            denominator = max_val - min_val
            norm = (numerator / denominator) + (1 - d) / 2

        score += weights[m] * norm

    return score

def process_excel_cps(file_path, global_min_vals=None, global_max_vals=None):
    """
    Process an Excel file and compute CPS using either global or local min/max values.
    
    Parameters:
    - file_path: Path to Excel file
    - global_min_vals: Optional dict of global min values for metrics
    - global_max_vals: Optional dict of global max values for metrics
    
    Returns:
    - df: DataFrame with CPS column added
    - overall_cps: Mean CPS value
    """
    # Define metrics used
    metrics = ['METEOR', 'Rouge-2.f', 'Rouge-l.f', 'Bert-Score.f1',
               'B-RT.average', 'F1 score', 'B-RT.fluency',
               'Laplace Perplexity', 'Lidstone Perplexity']

    # Load Excel file
    df = pd.read_excel(file_path)
    
    # Check which metrics exist in the DataFrame
    available_metrics = [m for m in metrics if m in df.columns]
    missing_metrics = [m for m in metrics if m not in df.columns]

    if missing_metrics:
        print('WARNING: The following metrics were not found in the Excel file:')
        for m in missing_metrics:
            print(f"  - {m}")

    # Create a new DataFrame with only the available metrics columns
    if available_metrics:
        metrics_df = df[available_metrics].copy()
        print('******************************')
        print('Metrics-only table:')
        print(metrics_df.head())  # Display first few rows
    else:
        print('ERROR: None of the specified metrics were found in the Excel file.')
        metrics_df = pd.DataFrame()  # Empty DataFrame as fallback
    
    # Define weights and polarity
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

    polarity = {
        "METEOR": +1,
        "Rouge-2.f": +1,
        "Rouge-l.f": +1,
        "Bert-Score.f1": +1,
        "B-RT.average": +1,
        "F1 score": +1,
        "B-RT.fluency": +1,
        "Laplace Perplexity": -1,
        "Lidstone Perplexity": -1,
    }
    
    # Compute min and max per metric - use global if provided, otherwise local
    if global_min_vals and global_max_vals:
        min_vals = global_min_vals
        max_vals = global_max_vals
        print("Using global min/max values for normalization")
    else:
        min_vals = {m: df[m].min() for m in available_metrics}
        max_vals = {m: df[m].max() for m in available_metrics}
        print("Using local min/max values for normalization")

    # Compute CPS for each row
    def compute_row_cps(row):
        metric_values = {m: row[m] for m in metrics}
        return compute_cps(metric_values, min_vals, max_vals, weights, polarity)

    df['CPS'] = df.apply(compute_row_cps, axis=1)
    overall_cps = df['CPS'].mean()

    print(f"Computed CPS for {file_path}: {overall_cps:.4f}")
    return df, overall_cps
# Usage example

if __name__ == "__main__":
    # Define metrics used
    metrics = ['METEOR', 'Rouge-2.f', 'Rouge-l.f', 'Bert-Score.f1',
               'B-RT.average', 'F1 score', 'B-RT.fluency',
               'Laplace Perplexity', 'Lidstone Perplexity']
    
    parser = argparse.ArgumentParser(description='Calculate global min/max for metrics across all Excel files.')
    parser.add_argument('directory', help='Root directory to search for Excel files')
    args = parser.parse_args()
    print(f"Calculating global min/max for metrics in directory: {args.directory}")
    
    # Calculate global min/max values
    global_min_vals, global_max_vals = calculate_global_metrics(args.directory, metrics)
    
    # Create a dictionary to store all CPS values
    all_cps_results = {}
    
    # Now you can process each file with the global min/max values
    excel_files = []
    for dirpath, dirnames, filenames in os.walk(args.directory):
        for filename in filenames:
            if filename.endswith(('.xlsx', '.xls', '.xlsm')):
                excel_files.append(os.path.join(dirpath, filename))
    
    # Sort files for consistent output
    excel_files.sort()
    
    for file_path in excel_files:
        try:
            print(f"\nProcessing {file_path} with global min/max values...")
            df, overall_cps = process_excel_cps(file_path, global_min_vals, global_max_vals)
            
            # Store the CPS result in our dictionary
            # Use the relative path as the key to make it more readable
            rel_path = os.path.relpath(file_path, args.directory)
            all_cps_results[rel_path] = overall_cps
            
            # Save results with CPS
            directory = os.path.dirname(file_path)
            filename = os.path.basename(file_path)
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_with_global_cps{ext}"
            output_file_path = os.path.join(directory, new_filename)
            
            df.to_excel(output_file_path, index=False)
            print(f"Results saved to {output_file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Print a summary of all CPS results
    print("\n==============================")
    print("SUMMARY OF ALL CPS RESULTS")
    print("==============================")

    # Split directory paths into two parts
    def split_path(path):
        parts = path.split(os.sep, 1)  # Split at first separator
        first_part = parts[0] if parts else ""
        second_part = parts[1] if len(parts) > 1 else ""
        return first_part, second_part
    
    # Create a dictionary to store all CPS values
    all_cps_results = {}

    # Create a global results table to store all individual CPS values
    global_results = []

    # Now you can process each file with the global min/max values
    excel_files = []
    for dirpath, dirnames, filenames in os.walk(args.directory):
        for filename in filenames:
            if filename == "file.xlsx":
                excel_files.append(os.path.join(dirpath, filename))

    # Sort files for consistent output
    excel_files.sort()

    for file_path in excel_files:
        try:
            print(f"\nProcessing {file_path} with global min/max values...")
            df, overall_cps = process_excel_cps(file_path, global_min_vals, global_max_vals)
            
            # Store the overall CPS result in our dictionary
            rel_path = os.path.relpath(file_path, args.directory)
            all_cps_results[rel_path] = overall_cps
            
            # Extract directory and filename parts for organization
            directory = os.path.dirname(rel_path)
            filename = os.path.basename(rel_path)
            first, second = split_path(directory)
            
            # Add each individual row's CPS to global_results
            for index, row in df.iterrows():
                if 'CPS' in row:
                # Check if the Excel file has an ID column and use that value if available
                    if 'id' in row:
                        row_id = row['id']
                    else:
                        # Fallback to sequential ID if no ID column in the Excel file
                        row_id = 'N/A'

                    global_results.append({
                        'ID': row_id,
                        'Directory': first,
                        'Filename': second,
                        'Excel_Row': index + 1,  # Adding 1 because index is 0-based
                        'CPS': float(row['CPS'])
                    })
            
            # Save results with CPS
            directory = os.path.dirname(file_path)
            filename = os.path.basename(file_path)
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_with_global_cps{ext}"
            output_file_path = os.path.join(directory, new_filename)
            
            df.to_excel(output_file_path, index=False)
            print(f"Results saved to {output_file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Convert to DataFrame
    global_results_df = pd.DataFrame(global_results)
    # Remove rows where ID is 'Average'
    global_results_df = global_results_df[global_results_df['ID'] != 'Average']
    print(f"Removed {len(global_results) - len(global_results_df)} rows with ID='Average'")

    # Save the global results table with all individual CPS values
    global_table_path = os.path.join(args.directory, "global_cps_results.xlsx")
    global_results_df.to_excel(global_table_path, index=False)
    print(f"\nGlobal CPS results table saved to {global_table_path}")

    # Split global_results_df into separate DataFrames by Directory
    unique_directories = global_results_df['Directory'].unique()
    print(f"\nFound {len(unique_directories)} unique directories: {unique_directories}")

    # Create a dictionary to store the separate DataFrames
    directory_dataframes = {}

    # Split the DataFrame by Directory
    for directory in unique_directories:
        # Filter rows for this directory
        directory_df = global_results_df[global_results_df['Directory'] == directory].copy()
        
        # Store in the dictionary with meaningful name
        directory_dataframes[directory] = directory_df
        
        # Print information about this DataFrame
        print(f"DataFrame for directory '{directory}': {len(directory_df)} rows")

        # Remove Directory column before saving (since it's all the same value)
        directory_df_output = directory_df.drop(columns=['Directory']) 
        
        # Save to separate Excel files
        output_path = os.path.join(args.directory, f"cps_results_{directory}.xlsx")
        directory_df_output.to_excel(output_path, index=False)
        print(f"Saved to {output_path}")

    # You can now access each DataFrame using directory_dataframes['directory_name']
    # For example: first_directory_df = directory_dataframes[unique_directories[0]]    

    # Create pivot tables for each directory with Excel_Row as rows and Filename as columns
    for directory in unique_directories:
        # Get the DataFrame for this directory
        df = directory_dataframes[directory]
        
        # Create pivot table
        print(f"\n==============================")
        print(f"PIVOT TABLE FOR {directory}")
        print(f"==============================")
        
        # Create the pivot table with Excel_Row as index (rows) and Filename as columns
        pivot = pd.pivot_table(
            df,
            values='CPS',           # CPS values in cells
            index='Excel_Row',      # Excel_Row as rows (changed from ID)
            columns='Filename',     # Filenames as columns
            fill_value=None         # Leave empty cells as NaN
        )
        
        # Round values to 4 decimal places
        pivot = pivot.round(4)
        
        # Reset index to make Excel_Row a regular column
        pivot_output = pivot.reset_index()
        
        # Add ID information in a separate column (optional)
        # This creates a comma-separated list of IDs for each Excel_Row
        id_pivot = df.groupby(['Excel_Row', 'Filename'])['ID'].first().unstack()
        
        # Save the pivot table to Excel
        pivot_path = os.path.join(args.directory, f"pivot_row_by_filename_{directory}.xlsx")
        pivot_output.to_excel(pivot_path, index=False)
        print(f"Pivot table saved to {pivot_path}")
        
        # Also display summary of the pivot table
        print(f"Pivot table shape: {pivot.shape[0]} Excel Rows × {pivot.shape[1]} Filenames")
        
        # Optional: Display preview of pivot table (first few rows)
        if len(pivot) > 0:
            preview_rows = min(5, len(pivot))
            print("\nPreview of pivot table (first few rows):")
            print(pivot.head(preview_rows))
    # Convert to DataFrame with split directory paths
    paths = [os.path.dirname(path) for path in all_cps_results.keys()]
    first_parts = []
    second_parts = []
    
    for path in paths:
        first, second = split_path(path)
        first_parts.append(first)
        second_parts.append(second)
    
    cps_df = pd.DataFrame({
        'First_Part': first_parts,
        'Second_Part': second_parts,
        'CPS': list(all_cps_results.values())
    })
    
    # Create a pivot table with First_Part as columns and Second_Part as rows
    pivot_df = cps_df.pivot_table(
        values='CPS', 
        index='Second_Part', 
        columns='First_Part',
        # aggfunc='mean'  # In case there are multiple entries for the same path combo
    )

    # Format for better display - round to 4 decimal places
    pivot_df = pivot_df.round(4)

    # Display the pivot table
    print("\n==============================")
    print("CPS VALUES BY DIRECTORY STRUCTURE")
    print("==============================")
    print(pivot_df)
    # Save to CSV - different approach to remove the header text
    pivot_output_path = os.path.join(args.directory, "cps_pivot_table.csv")

    # First convert the pivot table to a regular DataFrame
    # This resets the index but preserves structure
    pivot_df_reset = pivot_df.reset_index()

    # Write CSV with a blank string as the first column name
    column_names = pivot_df_reset.columns.tolist()
    column_names[0] = ''  # Replace 'Second_Part' with empty string
    pivot_df_reset.columns = column_names

    # Now save without the index
    pivot_df_reset.to_csv(pivot_output_path, index=False)
    print(f"\nPivot table saved to {pivot_output_path}")

    # Add this after saving the pivot table CSV
    # Import matplotlib for visualization
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create a figure with a specified size
    plt.figure(figsize=(14, 8))
    
    # No need to transpose - we want First_Part as lines
    # Plot each column (First_Part) as a separate line
    for first_part in pivot_df.columns:
        # Get column data across all Second_Part rows
        line_data = pivot_df[first_part].dropna()  # Drop NaN values
        plt.plot(line_data.index, line_data.values, 'o-', linewidth=2, markersize=8, label=first_part)
    
    # Configure axes
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add labels and title
    plt.title('CPS Values by Directory Structure', fontsize=16)
    plt.xlabel('Second Part of Path (x-axis)', fontsize=12)
    plt.ylabel('CPS Value (y-axis)', fontsize=12)
    plt.ylim(0, 1.0)  # Set y-axis from 0 to 1 for CPS values
    
    # Add legend with better visibility
    plt.legend(title="First Part:", loc='upper left', frameon=True, 
               facecolor='white', framealpha=0.9, fontsize=10)
    
    # Add a horizontal line for the overall average
    avg_cps = cps_df['CPS'].mean()
    plt.axhline(y=avg_cps, color='red', linestyle='--', alpha=0.5)
    plt.text(0, avg_cps + 0.02, f'Overall Average: {avg_cps:.4f}', 
             color='red', ha='left', va='bottom')
    
    plt.tight_layout()
    
    # Save the visualization
    line_plot_path = os.path.join(args.directory, "cps_first_part_lines.png")
    plt.savefig(line_plot_path, dpi=300, bbox_inches='tight')
    print(f"\nLine plot saved to {line_plot_path}")
    
    # Show the plot if running interactively
    plt.show()    
    # Sort by CPS value descending
    cps_df = cps_df.sort_values('CPS', ascending=False)    
    
    
    # Print summary
    # print(cps_df)
    
    # Calculate overall statistics
    print("\nCPS Statistics:")
    print(f"Average CPS: {cps_df['CPS'].mean():.4f}")
    print(f"Median CPS: {cps_df['CPS'].median():.4f}")
    print(f"Min CPS: {cps_df['CPS'].min():.4f}")
    print(f"Max CPS: {cps_df['CPS'].max():.4f}")
    
    # Save the CPS results to a CSV file
    cps_output_path = os.path.join(args.directory, "cps_summary.csv")
    cps_df.to_csv(cps_output_path, index=False)
    print(f"\nCPS summary saved to {cps_output_path}")