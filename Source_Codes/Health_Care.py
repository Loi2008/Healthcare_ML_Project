# Health_Care.py

# 1. IMPORT the logic from your other files
from Database import load_data, load_clean_data_to_db
from Pipeline import run_pipeline
from Train_Model import train_model

def main():
    print("--- 🚀 Starting Pipeline ---")

    # 2. STEP 1: Load (From Database.py)
    df_raw = load_data()
    print(f"Loaded {len(df_raw)} rows from database.")

    # 3. STEP 2: Clean (From Pipeline.py)
    # This runs the whole sequence: standardize -> clean -> convert -> etc.
    df_cleaned = run_pipeline(df_raw)

    # 4. STEP 3: Save (From Database.py)
    load_clean_data_to_db(df_cleaned)

    # Step 5: Train the machine learning model
    train_model(df_cleaned)

    print("\n--- ✨ Pipeline Finished Successfully ---")

if __name__ == "__main__":
    main()