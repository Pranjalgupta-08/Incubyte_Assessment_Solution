import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File path for the source data
source_file_path = "customer_data.csv"

def extract_header(file_path):
    print(os.listdir(os.getcwd()))
    print(f"Current Working Directory: {os.getcwd()}")
    """
    Extracts the header from the source file.

    Args:
        file_path (str): Path to the source file.

    Returns:
        list: Extracted column names.
    """
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('|H|'):
                    header = line.strip().split('|')[2:]  # Extract header from '|H|' line
                    logging.info(f"Extracted Header: {header}")
                    return header
        logging.error("Header row starting with '|H|' not found.")
        raise ValueError("Header row not found.")
    except Exception as e:
        logging.error(f"Error extracting header: {e}")
        raise


def validate_data(df):
    """
    Validates the data based on field lengths and mandatory checks.

    Args:
        df (pd.DataFrame): DataFrame to validate.

    Returns:
        pd.DataFrame: Validated DataFrame.
    """
    try:
        # Validate field lengths
        df["Customer_Name"] = df["Customer_Name"].apply(lambda x: x if pd.notnull(x) and len(x) <= 255 else None)
        df["Customer_Id"] = df["Customer_Id"].apply(lambda x: x if pd.notnull(x) and len(x) <= 18 else None)
        df["Vaccination_Id"] = df["Vaccination_Id"].apply(lambda x: x if pd.notnull(x) and len(x) <= 5 else None)
        df["Dr_Name"] = df["Dr_Name"].apply(lambda x: x if pd.notnull(x) and len(x) <= 255 else None)
        df["State"] = df["State"].apply(lambda x: x if pd.notnull(x) and len(x) <= 5 else None)
        df["Country"] = df["Country"].apply(lambda x: x if pd.notnull(x) and len(x) <= 5 else None)

        # Validate mandatory fields (non-null)
        mandatory_fields = ["Customer_Name", "Customer_Id", "Open_Date"]
        for field in mandatory_fields:
            df = df[df[field].notnull()]  # Keep rows where mandatory fields are not null

        return df
    except Exception as e:
        logging.error(f"Error validating data: {e}")
        raise

def process_and_load_staging(file_path, header):
    """
    Processes the source file and loads data into a staging DataFrame.

    Args:
        file_path (str): Path to the source file.
        header (list): List of column names.

    Returns:
        pd.DataFrame: Staging DataFrame.
    """
    try:
        # Read the file and filter data rows
        data = pd.read_csv(file_path, sep="|", header=None)
        staging_df = data[data.iloc[:, 1] == 'D'].iloc[:, 2:]  # Keep rows starting with '|D|'
        staging_df.columns = header  # Apply column names

        # Convert date columns to datetime format
        date_columns = ["Open_Date", "Last_Consulted_Date", "DOB"]
        for col in date_columns:
            if col == "DOB":
                staging_df[col] = pd.to_datetime(staging_df[col], format='%d%m%Y', errors='coerce')
            else:
                staging_df[col] = pd.to_datetime(staging_df[col], format='%Y%m%d', errors='coerce')

        # Validate the data
        staging_df = validate_data(staging_df)

        logging.info("Data successfully processed and loaded into staging DataFrame.")
        return staging_df
    except Exception as e:
        logging.error(f"Error processing and loading staging data: {e}")
        raise


def create_and_populate_country_tables(staging_df):
    """
    Creates and populates country-specific tables from the staging DataFrame.

    Args:
        staging_df (pd.DataFrame): Staging DataFrame.
    """
    try:
        # Keep the latest record for each Customer_Id across all countries
        global_latest_records = (
            staging_df.sort_values("Last_Consulted_Date", ascending=False)
            .groupby("Customer_Id")
            .head(1)
        )
        logging.info("Global latest records identified.")

        # Get unique countries
        countries = staging_df["Country"].unique()
        logging.info(f"Unique Countries: {countries}")

        for country in countries:
            country_df = global_latest_records[global_latest_records["Country"] == country]

            # Add derived columns
            country_df["Age"] = country_df["DOB"].apply(
                lambda x: datetime.now().year - x.year if pd.notnull(x) else np.nan
            )
            country_df["Days_Since_Last_Consulted"] = country_df["Last_Consulted_Date"].apply(
                lambda x: (pd.Timestamp.now() - x).days if pd.notnull(x) else np.nan
            )
            country_df["Consulted_Recently"] = country_df["Days_Since_Last_Consulted"].apply(
                lambda x: "Yes" if x > 30 else "No" if not pd.isna(x) else "Unknown"
            )

            # Save the table as a CSV
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"Table_{country}.csv")
            country_df.to_csv(output_path, index=False)
            logging.info(f"Saved Table for {country} at {output_path}")
    except Exception as e:
        logging.error(f"Error creating country tables: {e}")
        raise

def main():
    """
    Main function to orchestrate the ETL process.
    """
    try:
        # Extract header from the file
        header = extract_header(source_file_path)

        # Process and load data into the staging table
        staging_df = process_and_load_staging(source_file_path, header)
        logging.info(f"Staging DataFrame:\n{staging_df.head()}")  # Show sample data for debugging

        # Create and populate country-specific tables
        create_and_populate_country_tables(staging_df)

        logging.info("ETL process completed successfully.")
    except Exception as e:
        logging.error(f"ETL process failed: {e}")

if __name__ == "__main__":
    main()
