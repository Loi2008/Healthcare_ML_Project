import sys
from datetime import datetime

# Import your pipeline components
from Database import load_data, load_clean_data_to_db
from Pipeline import run_pipeline
from Train_Model import train_model


def retrain_pipeline():
    print("\n========================================")
    print(" Scheduled Retraining Started")
    print("Time:", datetime.now())
    print("========================================\n")

    try:
        # Step 1: Load raw data
        df_raw = load_data()
        print(f" Loaded {len(df_raw)} rows from database.")

        # Step 2: Clean data
        df_cleaned = run_pipeline(df_raw)
        print(" Data cleaning completed.")

        # Step 3: Save cleaned data
        load_clean_data_to_db(df_cleaned)
        print(" Cleaned data saved to PostgreSQL.")

        # Step 4: Train model
        train_model(df_cleaned)
        print(" Model retrained and saved.")

    except Exception as e:
        print(" ERROR during scheduled run:")
        print(str(e))
        sys.exit(1)

    print("\n========================================")
    print(" Scheduled Retraining Completed")
    print("========================================\n")


# Entry point
if __name__ == "__main__":
    retrain_pipeline()