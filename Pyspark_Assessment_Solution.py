from pyspark.sql import SparkSession
from pyspark.sql.functions import split, to_date, length, when, current_date, datediff, year, row_number
from pyspark.sql.window import Window

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerDataProcessing").getOrCreate()

# Function to read the data from the source file
def read_data(source_file_path):
    try:
        # Read the file as a text file
        raw_data = spark.read.text(source_file_path)
        return raw_data
    except Exception as e:
        print(f"Error reading data from {source_file_path}: {e}")
        raise

# Function to extract header from the data
def extract_header(raw_data):
    try:
        # Extract the header row
        header_row = raw_data.filter(raw_data["value"].startswith("|H|")).first()
        if header_row:
            header = header_row["value"].split("|")[2:]  # Extract column names
            print("Extracted Header:", header)
            return header
        else:
            raise ValueError("Header row not found in the data.")
    except Exception as e:
        print(f"Error extracting header: {e}")
        raise

# Function to process the data and split columns
def process_data(raw_data, header):
    try:
        # Filter the data rows starting with "|D|" and split into columns
        data_rows = raw_data.filter(raw_data["value"].startswith("|D|"))
        
        # Add a column by splitting the `value` column
        data_with_split = data_rows.withColumn("split", split(data_rows["value"], "\\|"))
        
        # Select the required columns based on the header
        data = data_with_split.select([data_with_split["split"][i + 2].alias(header[i]) for i in range(len(header))])
        return data
    except Exception as e:
        print(f"Error processing data: {e}")
        raise

# Function to convert date columns to the correct format
def convert_dates(data):
    try:
        # Convert the date columns to proper date format
        data = data.withColumn("Open_Date", to_date(data["Open_Date"], "yyyyMMdd")) \
                   .withColumn("Last_Consulted_Date", to_date(data["Last_Consulted_Date"], "yyyyMMdd")) \
                   .withColumn("DOB", to_date(data["DOB"], "ddMMyyyy"))
        return data
    except Exception as e:
        print(f"Error converting date columns: {e}")
        raise

# Function to validate data based on field length and mandatory checks
def validate_data(data):
    try:
        # Validate field lengths
        data = data.withColumn("Customer_Name", 
                               when(length(data["Customer_Name"]) <= 255, data["Customer_Name"])
                               .otherwise(None))  # Handle length validation
        data = data.withColumn("Customer_ID", 
                               when(length(data["Customer_ID"]) <= 18, data["Customer_ID"])
                               .otherwise(None))  # Handle length validation
        data = data.withColumn("Vaccination_Id", 
                               when(length(data["Vaccination_Id"]) <= 5, data["Vaccination_Id"])
                               .otherwise(None))  # Handle length validation
        data = data.withColumn("Dr_Name", 
                               when(length(data["Dr_Name"]) <= 255, data["Dr_Name"])
                               .otherwise(None))  # Handle length validation
        data = data.withColumn("State", 
                               when(length(data["State"]) <= 5, data["State"])
                               .otherwise(None))  # Handle length validation
        data = data.withColumn("Country", 
                               when(length(data["Country"]) <= 5, data["Country"])
                               .otherwise(None))  # Handle length validation
        # data = data.withColumn("Post_Code", 
        #                        when(length(data["Post_Code"]) <= 5, data["Post_Code"])
        #                        .otherwise(None))  # Handle length validation

        # Validate mandatory fields (non-null)
        data = data.filter(data["Customer_Name"].isNotNull())  # Mandatory field
        data = data.filter(data["Customer_ID"].isNotNull())  # Mandatory field
        data = data.filter(data["Open_Date"].isNotNull())  # Mandatory field

        return data
    except Exception as e:
        print(f"Error validating data: {e}")
        raise

# Function to get the latest record for each Customer_ID
def get_latest_data(data):
    try:
        # Define a window partitioned by Customer_ID and ordered by Last_Consulted_Date descending
        window_spec = Window.partitionBy("Customer_ID").orderBy(data["Last_Consulted_Date"].desc())

        # Add a row number column to identify the latest record for each Customer_ID
        data_with_rownum = data.withColumn("row_number", row_number().over(window_spec))

        # Filter to keep only the latest record for each Customer_ID
        latest_data = data_with_rownum.filter(data_with_rownum["row_number"] == 1).drop("row_number")

        return latest_data
    except Exception as e:
        print(f"Error getting latest data: {e}")
        raise

# Function to add derived columns (Age, Days_Since_Last_Consulted, Consulted_Recently)
def add_derived_columns(latest_data):
    try:
        # Add derived columns: Age, Days_Since_Last_Consulted, and Consulted_Recently
        latest_data = latest_data.withColumn("Age", year(current_date()) - year(latest_data["DOB"])) \
                                 .withColumn("Days_Since_Last_Consulted", datediff(current_date(), latest_data["Last_Consulted_Date"])) \
                                 .withColumn("Consulted_Recently", 
                                             when(datediff(current_date(), latest_data["Last_Consulted_Date"]) <= 30, "Yes")
                                             .when(datediff(current_date(), latest_data["Last_Consulted_Date"]) > 30, "No")
                                             .otherwise("Unknown"))
        return latest_data
    except Exception as e:
        print(f"Error adding derived columns: {e}")
        raise

# Function to save data to country-level tables
def save_country_data(latest_data):
    try:
        # Get the list of unique countries
        countries = [row["Country"] for row in latest_data.select("Country").distinct().collect()]

        for country in countries:
            # Filter data for each country
            country_data = latest_data.filter(latest_data["Country"] == country)

            # Define table path
            country_table_path = f"dbfs:/mnt/country_tables/Table_{country}"

            # Save the data for the country with schema evolution enabled
            country_data.write.format("delta") \
                .option("mergeSchema", "true") \
                .mode("overwrite") \
                .save(country_table_path)

            # Create a Delta table for the country
            spark.sql(f"CREATE TABLE IF NOT EXISTS Table_{country}_check USING DELTA LOCATION '{country_table_path}'")
    except Exception as e:
        print(f"Error saving country data: {e}")
        raise

# Main function to orchestrate the entire process
def main(source_file_path):
    try:
        # Step 1: Read the raw data
        raw_data = read_data(source_file_path)

        # Step 2: Extract the header
        header = extract_header(raw_data)

        # Step 3: Process the data
        data = process_data(raw_data, header)

        # Step 4: Convert dates to proper format
        data = convert_dates(data)

        # Step 5: Validate the data
        data = validate_data(data)

        # Step 6: Get the latest data for each Customer_ID
        latest_data = get_latest_data(data)

        # Step 7: Add derived columns
        latest_data = add_derived_columns(latest_data)

        # Step 8: Save data to country-level tables
        save_country_data(latest_data)

        print("Data processing and saving completed successfully.")

    except Exception as e:
        print(f"Error in main function: {e}")

# File path for the source data
source_file_path = "dbfs:/FileStore/shared_uploads/pranjalgupta1230@gmail.com/customer_data-1.csv"

# Run the main function
main(source_file_path)
