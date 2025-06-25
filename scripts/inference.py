import pandas as pd
import joblib

# --- CONFIGURATION ---
MODEL_PATH = "models/bank_marketing_lightgbm_model.pkl"
THRESHOLD = 0.06

# --- LOAD MODEL ---
def load_model(path):
    print("Loading model from", path)
    return joblib.load(path)

# --- RUN INFERENCE ---
def run_inference(model, input_csv, output_csv):
    # Load input data
    print("Reading input file:", input_csv)
    new_data = pd.read_csv(input_csv, sep=';')

    # Predict probabilities
    print("Scoring data...")
    probabilities = model.predict_proba(new_data)[:, 1]
    new_data['subscription_probability'] = probabilities
    new_data['should_be_contacted'] = new_data['subscription_probability'] >= THRESHOLD

    # Sort and filter
    scored = new_data.sort_values(by='subscription_probability', ascending=False)
    contacted = scored[scored['should_be_contacted']]

    # Save output
    print(f"Saving {len(contacted)} contacts to {output_csv}")
    contacted.to_csv(output_csv, index=False)
    print("Done.")

# --- MAIN ENTRY ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on new customer data")
    parser.add_argument('--input', required=True, help="Path to new_customers.csv")
    parser.add_argument('--output', required=True, help="Path to output file (e.g., results.csv)")

    args = parser.parse_args()

    model = load_model(MODEL_PATH)
    run_inference(model, args.input, args.output)
