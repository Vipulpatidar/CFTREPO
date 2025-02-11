import sys
import re
from datetime import datetime
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, regexp_replace, regexp_extract, when, round, udf, to_date, coalesce, trim, lower, year
)
from pyspark.sql.types import IntegerType, DoubleType, StringType, DateType

# Initialize AWS Glue Job
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

args = getResolvedOptions(sys.argv, ['PATH'])
path = args['PATH']

df = spark.read.option("header", "true").csv(path)

######################

df = df.withColumn("CCATrank", regexp_extract(col("CCATrank"), r"(\d+)", 1))


######################

# Function to clean and format DOB
def clean_and_format_dob(dob):
    if dob is None:
        return None

    dob = str(dob).strip().lower()
    
    # Common misspellings & month variations
    misspelled_months = {
        # Existing corrections
        "januray": "january", "janauary": "january", "janury": "january", "janurary": "january",
        "febraury": "february", "febuary": "february",
        "marhc": "march", "marchh": "march",
        "aprill": "april", "aprilr": "april", "arp": "april",
        "junne": "june", "jully": "july", "juley": "july",
        "agust": "august", "augest": "august",
        "setember": "september", "sepember": "september", "sepetember": "september",
        "octuber": "october", "octobr": "october",
        "novembar": "november", "novemebr": "november",
        "decmber": "december", "decembar": "december"
    }

    for wrong, correct in misspelled_months.items():
        dob = re.sub(rf"\b{wrong}\b", correct, dob)

    # Remove ordinal suffixes (1st, 2nd, 3rd, etc.)
    dob = re.sub(r'(\d{1,2})(st|nd|rd|th)', r'\1', dob)

    # Standardize delimiters (replace `/` or `.` with `-`)
    dob = re.sub(r"[/.]", "-", dob)

    # Convert text-based month formats
    months_map = {
        "january": "Jan", "february": "Feb", "march": "Mar", "april": "Apr",
        "may": "May", "june": "Jun", "july": "Jul", "august": "Aug",
        "september": "Sep", "october": "Oct", "november": "Nov", "december": "Dec"
    }
    for full, short in months_map.items():
        dob = dob.replace(full, short)

    # Define multiple accepted date formats
    date_formats = [
        "%d-%m-%Y", "%d/%m/%Y", 
        "%d %m %Y",  # New format for numeric month with spaces
        "%d-%b-%y", "%d-%b-%Y", 
        "%Y-%m-%d", 
        "%d %b %Y", "%d %B %Y", 
        "%d-%m-%y", "%d/%m/%y"
    ]
    
    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(dob, fmt)
            # Fix year if it's in 2-digit format
            if parsed_date.year < 1930:
                parsed_date = parsed_date.replace(year=parsed_date.year + 100)
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None

clean_dob_udf = udf(clean_and_format_dob, StringType())
df = df.withColumn("DOB", clean_dob_udf(col("DOB")))
df = df.withColumn("DOB", to_date(col("DOB"), "yyyy-MM-dd"))

####################


# Regular expression patterns
cgpa_pattern = r"(\d+(\.\d+)?)\s*[cC][gG][pP][aA]"  # Matches "6.6 CGPA", "8.2 cgpa"
percent_pattern = r"(\d+(\.\d+)?)%"  # Matches "70.30%", "64%"
mixed_pattern = r"\((\d+(\.\d+)?)\s*[cC][gG][pP][aA]\)"  # Extract CGPA from "(7.40 CGPA)"


# Function to process percentage columns
def process_percentage_column(df, column_name):
    df = df.withColumn(column_name, regexp_extract(col(column_name), r"(\d*\.?\d+)", 1).cast("double"))
    df = df.withColumn(
        column_name,
        round(
            when(col(column_name).rlike(mixed_pattern), 
                 regexp_extract(col(column_name), mixed_pattern, 1).cast("double") * 9.5  # Extract CGPA from "(7.40 CGPA)" and convert
            )
            .when(col(column_name).rlike(cgpa_pattern), 
                  regexp_extract(col(column_name), cgpa_pattern, 1).cast("double") * 9.5  # Extract CGPA from "6.6 CGPA"
            )
            .when(col(column_name).rlike(percent_pattern), 
                  regexp_extract(col(column_name), percent_pattern, 1).cast("double")  # Extract percentage value from "70.30%"
            )
            .when((col(column_name).cast("double") > 0) & (col(column_name).cast("double") <= 1.0), 
                  col(column_name).cast("double") * 100  # Convert fractional percentage (e.g., 0.85 → 85%)
            )
            .when(col(column_name).cast("double") <= 10.0, 
                  col(column_name).cast("double") * 9.5  # Convert CGPA values (e.g., 8.5 → 80.75%)
            )
            .otherwise(col(column_name)), 2  # Round to 2 decimal places
        )
    )
    return df

# Apply percentage processing
df = process_percentage_column(df, "10th_percentage")
df = process_percentage_column(df, "grad_percentage")
df = process_percentage_column(df, "12th_percentage")
df = process_percentage_column(df, "diploma_percentage")
# df = process_percentage_column(df, "pg_percentage")


#############################

# Higher Education Processing (Combining 12th & Diploma)
df = df.withColumn(
    "Higher_Edu_Percent",
    when(col("12th_percentage").isNotNull() & col("diploma_percentage").isNotNull(), 
         (col("12th_percentage") + col("diploma_percentage")) / 2)
    .otherwise(coalesce(col("12th_percentage"), col("diploma_percentage")))
)
df = df.drop("12th_percentage", "diploma_percentage")
df = process_percentage_column(df, "Higher_Edu_Percent")


#############################
#############################

int_columns = [col_name for col_name in df.columns if "Theory" in col_name or 
               "Lab" in col_name or 
               "Total" in col_name or col_name == "Total"]

for col_name in int_columns:
    df = df.withColumn(col_name, df[col_name].cast(IntegerType()))

df = df.withColumn("Percentage", df["Percentage"].cast(DoubleType()))
df = df.withColumn("pg_percentage", df["pg_percentage"].cast(DoubleType()))
df = df.withColumn("CCATrank", df["CCATrank"].cast(IntegerType()))

#####################

# Clean Pre-CCAT Processing
df = df.withColumn("pre_ccat", trim(lower(col("pre_ccat"))))
df = df.withColumn("pre_ccat", when(col("pre_ccat") == "yes", "yes").otherwise("no"))


##############################

# Placement Status Cleaning
not_placed_values = ["fail", "30 calls done", "failed", "opted out for placement", "out of placements"]
df = df.withColumn(
    "Is_Placed",
    when(col("Company_Name").isNull(), "No")
    .when(lower(trim(col("Company_Name"))).isin(not_placed_values), "No")
    .otherwise("Yes")
)

################################

# Extract Branch from Degree
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

df = df.withColumn(
    "branch_cleaned",
    when(col("branch").isNull() | (lower(col("branch")) == "other"), extract_branch("grad_degree"))  
    .otherwise(extract_branch("branch"))  
)


#############################

df = df.withColumn(
    "projectgrade",
    when(lower(col("projectgrade")).rlike("absent|ab"), None)  # Convert 'absent' to NULL
    .when(lower(col("projectgrade")).rlike("fail"), "F")  # Convert 'fail' to 'F'
    .otherwise(col("projectgrade"))  # Keep other values unchanged
)


df = df.withColumn(
    "aptigrade",
    when(lower(col("projectgrade")).rlike("absent|ab"), None)  # Convert 'absent' to NULL
    .when(lower(col("projectgrade")).rlike("fail"), "F")  # Convert 'fail' to 'F'
    .otherwise(col("projectgrade"))  # Keep other values unchanged
)


#############################

# Age Calculation
df = df.withColumn("dob", to_date(col("dob"), "yyyy-MM-dd"))
df = df.withColumn("Age", year(col("year")) - year(col("dob")))


#############################

# Saving Cleaned Data to S3
output_path = "s3://dataforsagemaker101/dbda"
df.write.mode("overwrite").option("header", "true").csv(output_path)

# Print Confirmation
print(f" Data successfully transformed and saved to: {output_path}")

