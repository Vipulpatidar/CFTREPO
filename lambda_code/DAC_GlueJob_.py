# Import Required Libraries
import sys
import re
import dateutil.parser
from datetime import datetime
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType, StringType, DateType
from pyspark.sql.functions import (
    col, when, coalesce, regexp_extract, regexp_replace, 
    udf, to_date, round, trim, lower, year
)

# Initialize AWS Glue Job
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

#  Read CSV from S3
input_path = "s3://mybucket1-dac/NewDAC_Concat.csv"
df = spark.read.option("header", "true").csv(input_path)

# Print Column Names and Data Types (for Debugging)
print("Initial Data Types:")
for col_name, col_type in df.dtypes:
    print(f"Column : {col_name}, Data Type : {col_type}")

# ------------------------------------------------------
# 🔹 Data Type Conversions
# ------------------------------------------------------

# Identify Columns to Convert to IntegerType
int_columns = [col_name for col_name in df.columns if "_Theory" in col_name or 
               "_Lab" in col_name or "_Total" in col_name or col_name == "Total"]

# Convert Selected Columns to IntegerType
for col_name in int_columns:
    df = df.withColumn(col_name, df[col_name].cast(IntegerType()))

# Convert 'Percentage' and 'pg_percentage' Columns to DoubleType
df = df.withColumn("Percentage", df["Percentage"].cast(DoubleType()))
df = df.withColumn("pg_percentage", df["pg_percentage"].cast(DoubleType()))
# ------------------------------------------------------
# 🔹 Cleaning 'CCATrank' Column
# ------------------------------------------------------

df = df.withColumn("CCATrank", regexp_extract(col("CCATrank"), r"(\d+)", 1))
# Explicitly cast 'CCATrank' to IntegerType
df = df.withColumn("CCATrank", df["CCATrank"].cast(IntegerType()))

# ------------------------------------------------------
# 🔹 Cleaning 'DOB' Column
# ------------------------------------------------------

# Function to normalize date formats
def parse_date(date_str):
    if date_str:
        try:
            return dateutil.parser.parse(date_str).strftime('%Y-%m-%d')
        except Exception:
            return None
    return None

# Register as a UDF
parse_date_udf = udf(parse_date, StringType())

# Manually fix specific incorrect dates before applying the parser
df = df.withColumn(
    "dob",
    when(col("dob") == "14 januray 1997", "14 January 1997")
    .when(col("dob") == "6 jully 1997", "6 July 1997")
    .otherwise(col("dob"))
)

# Apply the UDF to normalize date formats
df = df.withColumn("dob", parse_date_udf(col("dob")))


# ------------------------------------------------------
# 🔹 Processing Percentage Columns
# ------------------------------------------------------

def process_percentage_column(df, column_name):
    # Extract full numerical value including leading decimals (e.g., ".95" should become "0.95")
    df = df.withColumn(column_name, regexp_extract(col(column_name), r"(\d*\.?\d+)", 1).cast("double"))

    # Apply transformations for CGPA and percentage adjustments, then round to 2 decimal places
    df = df.withColumn(
        column_name,
        round(
            when((col(column_name) > 0) & (col(column_name) < 1.0), col(column_name) * 100)  # Convert 0.x to percentage
            .when((col(column_name) >= 1.0) & (col(column_name) < 10.0), col(column_name) * 9.5)  # Convert CGPA to %
            .otherwise(col(column_name)),  # Keep other values unchanged
            2  # Round to 2 decimal places
        )
    )
    
    return df

# Example usage
df = process_percentage_column(df, "10th_percentage")
df = process_percentage_column(df, "grad_percentage")
df = process_percentage_column(df, "12th_percentage")
df = process_percentage_column(df, "diploma_percentage")
df = process_percentage_column(df, "pg_percentage")

# ------------------------------------------------------
# 🔹 Processing 'Higher_Edu_Percent' Column
# ------------------------------------------------------
df = df.withColumn(
    "Higher_Edu_Percent",
    when(col("12th_percentage").isNotNull() & col("diploma_percentage").isNotNull(), 
         (col("12th_percentage") + col("diploma_percentage")) / 2)  # Take the average if both are available
    .otherwise(coalesce(col("12th_percentage"), col("diploma_percentage")))  # Else take the non-null value
)
# Drop the old columns
df = df.drop("12th_percentage", "diploma_percentage")
df = process_percentage_column(df, "Higher_Edu_Percent")

# ------------------------------------------------------
# 🔹 Processing 'pre_ccat' Column
# ------------------------------------------------------

df = df.withColumn("pre_ccat", trim(lower(col("pre_ccat"))))
df = df.withColumn("pre_ccat", when(col("pre_ccat") == "yes", "yes").otherwise("no"))

# ------------------------------------------------------
# 🔹 Processing 'Is_Placed' Column
# ------------------------------------------------------
not_placed_values = ["fail", "30 calls done", "failed", "opted out for placement", "out of placements"]

# Create the 'Is_Placed' column with case-insensitive matching
df = df.withColumn(
    "Is_Placed",
    when(col("Company_Name").isNull(), "No")  # If Company_Name is NULL → No
    .when(lower(trim(col("Company_Name"))).isin(not_placed_values), "No")  # Case-insensitive check
    .otherwise("Yes")  # Otherwise, if Company_Name exists → Yes
)


# ------------------------------------------------------
# 🔹 Processing 'Age' Column
# ------------------------------------------------------
# Assuming 'dob' column is in 'yyyy-mm-dd' format
df = df.withColumn("dob", to_date(col("dob"), "yyyy-MM-dd"))

# Calculate age by subtracting the year extracted from 'dob' from the 'year' column
df = df.withColumn("Age", col("year") - year(col("dob")))


# ------------------------------------------------------
# 🔹 Processing 'Branch' Column
# ------------------------------------------------------

# Define a function to extract branch names
def extract_branch(column):
    return (
        when(lower(col(column)).rlike("computer|information technology|it|i.t"), "Computer")
        .when(lower(col(column)).rlike("mechanical|mech"), "Mechanical")
        .when(lower(col(column)).rlike("electronics|telecommunication|e&tc|extc|e and tc|entc"), "Electronics and Telecommunication")
        .when(lower(col(column)).rlike("civil"), "Civil")
        .when(lower(col(column)).rlike("math"), "Mathematics")
        .when(lower(col(column)).rlike("phys"), "Physics")
        .when(lower(col(column)).rlike("chem"), "Chemical")
        .when(lower(col(column)).rlike("instru"), "Instrumentation")
        .when(lower(col(column)).rlike("electrical"), "Electrical")
        .when(lower(col(column)).rlike("bsc"), "BSc")
        .otherwise("BE")  # Categorize remaining values as "Other"
    )

# Apply extraction logic on both columns
df = df.withColumn(
    "branch_cleaned",
    when(col("branch").isNull() | (lower(col("branch")) == "other"), extract_branch("grad_degree"))  
    .otherwise(extract_branch("branch"))  
)



# Dropping unwanted columns
df = df.drop("branch", "grad_degree")
df = df.withColumn(
    "Project_Grade",
    when(lower(col("Project_Grade")).rlike("absent|ab"), None)  # Convert 'absent' to NULL
    .when(lower(col("Project_Grade")).rlike("fail"), "F")  # Convert 'fail' to 'F'
    .otherwise(col("Project_Grade"))  # Keep other values unchanged
)


df = df.withColumn(
    "aptigrade",
    when(lower(col("Project_Grade")).rlike("absent|ab"), None)  # Convert 'absent' to NULL
    .when(lower(col("Project_Grade")).rlike("fail"), "F")  # Convert 'fail' to 'F'
    .otherwise(col("Project_Grade"))  # Keep other values unchanged
)
# ------------------------------------------------------
# 🔹 Saving Cleaned Data to S3
# ------------------------------------------------------

output_path = "s3://daccleaned/transformed_dac_cleaned.csv"
df.write.mode("overwrite").option("header", "true").csv(output_path)

# Print Confirmation
print(f"✅ Data successfully transformed and saved to: {output_path}")

#  Print Final Data Types
print("✅ Final Data Types:")
for col_name, col_type in df.dtypes:
    print(f"Column : {col_name}, Data Type : {col_type}")
