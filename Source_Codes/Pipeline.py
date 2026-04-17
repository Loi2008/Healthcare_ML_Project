import pandas as pd


# 🔹 Standardize column names (lowercase + underscores)
def standardize_column_names(healthcare_data):
    healthcare_data.columns = (
        healthcare_data.columns
        .str.strip()              # remove leading/trailing spaces
        .str.lower()              # convert to lowercase
        .str.replace(" ", "_")    # replace spaces with underscores
    )
    return healthcare_data


# 🔹 Clean text columns (remove extra spaces and formatting inconsistencies)
def clean_text_columns(healthcare_data):
    # Select all string/object columns
    text_cols = healthcare_data.select_dtypes(include=["object", "string"]).columns

    for col in text_cols:
        healthcare_data[col] = healthcare_data[col].astype(str).str.strip()  # remove outer spaces
        healthcare_data[col] = healthcare_data[col].str.replace(r"\s+", " ", regex=True)  # normalize spaces

    return healthcare_data


# 🔹 Convert columns to correct data types (numeric + datetime)
def convert_data_types(healthcare_data):
    # Convert numeric columns
    healthcare_data["age"] = pd.to_numeric(healthcare_data["age"], errors="coerce")
    healthcare_data["billing_amount"] = pd.to_numeric(healthcare_data["billing_amount"], errors="coerce")
    healthcare_data["room_number"] = pd.to_numeric(healthcare_data["room_number"], errors="coerce")

    # Convert date columns
    healthcare_data["date_of_admission"] = pd.to_datetime(
        healthcare_data["date_of_admission"], errors="coerce"
    )
    healthcare_data["discharge_date"] = pd.to_datetime(
        healthcare_data["discharge_date"], errors="coerce"
    )

    return healthcare_data


# 🔹 Standardize categorical values (fix casing and inconsistencies)
def standardize_categorical_values(healthcare_data):
    # Columns to convert to title case
    title_case_cols = [
        "name",
        "blood_type",
        "medical_condition",
        "doctor",
        "hospital",
        "insurance_provider",
        "admission_type",
        "medication",
        "test_results"
    ]

    for col in title_case_cols:
        healthcare_data[col] = healthcare_data[col].str.title()

    # Standardize gender values
    healthcare_data["gender"] = healthcare_data["gender"].str.lower().replace({
        "m": "Male",
        "male": "Male",
        "f": "Female",
        "female": "Female"
    })

    # Standardize test results
    healthcare_data["test_results"] = healthcare_data["test_results"].str.lower().replace({
        "normal": "Normal",
        "abnormal": "Abnormal",
        "inconclusive": "Inconclusive"
    })

    # Standardize admission type
    healthcare_data["admission_type"] = healthcare_data["admission_type"].str.lower().replace({
        "emergency": "Emergency",
        "urgent": "Urgent",
        "elective": "Elective"
    })

    return healthcare_data


# 🔹 Remove duplicate rows
def remove_duplicates(healthcare_data):
    duplicates = healthcare_data.duplicated().sum()
    print(f"\nDuplicate rows before removal: {duplicates}")

    healthcare_data = healthcare_data.drop_duplicates()

    print(f"Shape after duplicate removal: {healthcare_data.shape}")
    return healthcare_data


# 🔹 Validate data ranges and convert invalid values to missing (NaN)
def validate_ranges(healthcare_data):
    # Age should be between 0 and 120
    healthcare_data.loc[
        (healthcare_data["age"] < 0) | (healthcare_data["age"] > 120),
        "age"
    ] = pd.NA

    # Billing amount should not be negative
    healthcare_data.loc[
        healthcare_data["billing_amount"] < 0,
        "billing_amount"
    ] = pd.NA

    # Room number should not be negative
    healthcare_data.loc[
        healthcare_data["room_number"] < 0,
        "room_number"
    ] = pd.NA

    # Discharge date should not be before admission date
    invalid_dates = healthcare_data["discharge_date"] < healthcare_data["date_of_admission"]
    healthcare_data.loc[invalid_dates, "discharge_date"] = pd.NaT

    return healthcare_data


# 🔹 Handle missing values using appropriate strategies
def handle_missing_values(healthcare_data):
    print("\nMissing values BEFORE handling:")
    print(healthcare_data.isnull().sum())

    # Fill missing billing_amount with median
    healthcare_data["billing_amount"] = healthcare_data["billing_amount"].fillna(
        healthcare_data["billing_amount"].median()
    )

    print("\nMissing values AFTER handling:")
    print(healthcare_data.isnull().sum())

    return healthcare_data


# 🔹 Feature engineering (create new useful features)
def engineer_features(healthcare_data):
    # Calculate length of hospital stay in days
    healthcare_data["length_of_stay"] = (
        healthcare_data["discharge_date"] - healthcare_data["date_of_admission"]
    ).dt.days

    return healthcare_data


# 🔹 Drop columns not useful for machine learning
def drop_unused_columns(healthcare_data):
    columns_to_drop = [
        "name",
        "doctor",
        "hospital",
        "insurance_provider",
        "room_number",
        "date_of_admission",
        "discharge_date"
    ]

    healthcare_data = healthcare_data.drop(columns=columns_to_drop)
    return healthcare_data


# 🔹 Final validation checks (for debugging and reporting)
def final_checks(healthcare_data):
    print("\nFinal dataset info:")
    healthcare_data.info()

    print("\nMissing values after cleaning:")
    print(healthcare_data.isnull().sum())

    print("\nSummary statistics:")
    print(healthcare_data.describe(include="all"))

    print("\nUnique values in key categorical columns:")
    for col in ["gender", "blood_type", "admission_type", "test_results"]:
        print(f"\n{col}:")
        print(healthcare_data[col].value_counts())


# 🔹 Master pipeline function (calls all steps in correct order)
def run_pipeline(healthcare_data):
    healthcare_data = standardize_column_names(healthcare_data)
    healthcare_data = clean_text_columns(healthcare_data)
    healthcare_data = convert_data_types(healthcare_data)
    healthcare_data = standardize_categorical_values(healthcare_data)

    healthcare_data = remove_duplicates(healthcare_data)
    healthcare_data = validate_ranges(healthcare_data)
    healthcare_data = handle_missing_values(healthcare_data)

    healthcare_data = engineer_features(healthcare_data)
    healthcare_data = drop_unused_columns(healthcare_data)

    final_checks(healthcare_data)

    return healthcare_data