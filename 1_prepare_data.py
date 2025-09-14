# --- START OF FILE 1_prepare_data.py ---

import pandas as pd
import numpy as np
import os

def feature_engineering(df):
    """
    Adds new, physics-informed features to the dataframe to help models generalize.
    """
    print("Performing feature engineering...")

    # 1. Dimensionless Group for Reflux Ratio
    # This linearizes the effect of R and is bounded between 0 and 1.
    df['R_over_R_plus_1'] = df['R'] / (df['R'] + 1)

    # 2. Interaction Term
    # The effect of reflux is dependent on the feed composition.
    df['R_times_xF'] = df['R'] * df['xF']

    # 3. Physics-Inspired Features based on Relative Volatility (alpha)
    # This encodes the non-linear VLE behavior of Ethanol-Water.
    # NOTE: This is a simplified polynomial approximation for alpha vs xF.
    # A more accurate model would use a proper thermodynamic model (e.g., NRTL).
    df['alpha_approx'] = -5.7 * df['xF']**2 + 3.1 * df['xF'] + 3.6

    # 4. Minimum Reflux Ratio (approximated)
    # This is a key parameter in distillation design.
    # Handle potential division by zero if alpha is close to 1.
    df['R_min_approx'] = 1 / (df['alpha_approx'] - 1)
    # Clean up any infinite values that might arise if alpha_approx is exactly 1
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['R_min_approx'], inplace=True) # Drop rows where R_min couldn't be calculated

    # 5. The "Golden Feature": The Reflux Ratio Factor
    # This normalizes the reflux ratio by its theoretical minimum, creating a more
    # invariant feature across different feed compositions.
    df['R_factor'] = df['R'] / df['R_min_approx']
    
    print(f"Added new features: {['R_over_R_plus_1', 'R_times_xF', 'alpha_approx', 'R_min_approx', 'R_factor']}")
    return df


def prepare_data_and_features():
    """
    Reads raw DWSIM simulation CSVs, adds the correct feed mole fraction (xF),
    renames columns, combines them, runs feature engineering, and saves a master CSV file.
    """
    print("Starting data preparation...")
    
    # Define paths
    raw_data_path = 'data/raw/'
    processed_data_path = 'data/processed/'
    # The output file now has a new name to reflect the added features
    output_file = os.path.join(processed_data_path, 'master_data_featured.csv')

    # Ensure processed directory exists
    os.makedirs(processed_data_path, exist_ok=True)

    datasets = [f for f in os.listdir(raw_data_path) if f.endswith('.csv')]
    if not datasets:
        print(f"Error: No CSV files found in {raw_data_path}. Please add raw data files.")
        return

    df_list = []

    # Map filenames to xF concentrations
    for file in datasets:
        try:
            concentration = float(file.split('_')[-1].replace('.csv', ''))
            temp_df = pd.read_csv(os.path.join(raw_data_path, file))
            temp_df['xF'] = concentration
            df_list.append(temp_df)
            print(f"Processed {file} with xF = {concentration}")
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse xF from filename {file}. Skipping. Error: {e}")

    if not df_list:
        print("Error: No dataframes were created. Please check raw data files.")
        return

    # Combine all dataframes
    combined_df = pd.concat(df_list, ignore_index=True)

    # Rename columns from DWSIM output to simpler names
    column_map = {
        'DCOL-1 - Condenser_Specification_Value': 'R',
        'DCOL-1 - Reboiler_Specification_Value': 'B',
        '3 - Molar Fraction (Mixture) / Ethanol ()': 'xD',
        'DCOL-1 - Reboiler Duty (kcal/h)': 'QR'
    }
    
    # Select and rename necessary columns
    # We keep only the relevant columns to avoid clutter
    relevant_cols = list(column_map.keys())
    # Check if columns exist before trying to access them
    existing_cols = [col for col in relevant_cols if col in combined_df.columns]
    
    final_df = combined_df[existing_cols + ['xF']].copy()
    final_df = final_df.rename(columns=column_map)
    
    # Perform feature engineering
    featured_df = feature_engineering(final_df)

    # Save the processed data with new features
    featured_df.to_csv(output_file, index=False)
    print(f"Successfully created featured dataset at {output_file}")
    print(f"Total data points after processing: {len(featured_df)}")
    print("\nColumns in final dataset:")
    print(featured_df.columns.tolist())

if __name__ == '__main__':
    prepare_data_and_features()

# --- END OF FILE 1_prepare_data.py ---