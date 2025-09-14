import pandas as pd
import os

def prepare_data():
    """
    Reads raw DWSIM simulation CSVs, adds the correct feed mole fraction (xF),
    renames columns, combines them, and saves a master CSV file.
    """
    print("Starting data preparation...")
    

    raw_data_path = 'data/raw/'
    processed_data_path = 'data/processed/'
    output_file = os.path.join(processed_data_path, 'master_data.csv')

 
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


    combined_df = pd.concat(df_list, ignore_index=True)

  
    column_map = {
        'DCOL-1 - Condenser_Specification_Value': 'R',
        'DCOL-1 - Reboiler_Specification_Value': 'B',
        '3 - Molar Fraction (Mixture) / Ethanol ()': 'xD',
        'DCOL-1 - Reboiler Duty (kcal/h)': 'QR'
    }
    

    final_df = combined_df[list(column_map.keys()) + ['xF']]
    final_df = final_df.rename(columns=column_map)

    final_df = final_df[['xF', 'R', 'B', 'xD', 'QR']]

   
    final_df.to_csv(output_file, index=False)
    print(f"Successfully created master dataset at {output_file}")
    print(f"Total data points: {len(final_df)}")

if __name__ == '__main__':
    prepare_data()

