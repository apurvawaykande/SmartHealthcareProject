import pandas as pd
import os

# Absolute paths relative to this file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DISEASE_FILE = os.path.join(CURRENT_DIR, "DiseaseAndSymptoms.csv")
PRECAUTION_FILE = os.path.join(CURRENT_DIR, "Disease_precaution.csv")

# Load datasets safely
try:
    disease_symptom_df = pd.read_csv(DISEASE_FILE)
    precaution_df = pd.read_csv(PRECAUTION_FILE)
except FileNotFoundError as e:
    raise FileNotFoundError(
        f"CSV file not found: {e}. Ensure the CSV files are in the 'src/' folder."
    )

# Lowercase for matching
disease_symptom_df['Disease'] = disease_symptom_df['Disease'].str.lower()
precaution_df['Disease'] = precaution_df['Disease'].str.lower()


def get_advice_for_symptom(user_input):
    user_input = user_input.lower().strip()
    mask = disease_symptom_df.apply(
        lambda row: user_input in row['Disease'] or any(
            user_input in str(row[col]).lower() for col in disease_symptom_df.columns if 'Symptom' in col
        ),
        axis=1
    )
    matches = disease_symptom_df[mask]

    if matches.empty:
        return None

    disease = matches['Disease'].iloc[0]

    # Collect symptoms
    all_symptoms = []
    for col in disease_symptom_df.columns:
        if "Symptom" in col:
            all_symptoms.extend(matches[col].dropna().tolist())
    all_symptoms = list(dict.fromkeys([s.strip().replace("_", " ") for s in all_symptoms if isinstance(s, str)]))

    # Collect precautions
    precautions = precaution_df[precaution_df['Disease'] == disease]
    if not precautions.empty:
        precautions_list = [precautions[col].values[0] for col in precaution_df.columns if "Precaution" in col]
        precautions_text = ", ".join([p for p in precautions_list if isinstance(p, str)])
    else:
        precautions_text = "No specific precautions found."

    return f"**Disease:** {disease.title()}\n\n**Common Symptoms:** {', '.join(all_symptoms[:8])}\n\n**Precautions:** {precautions_text}"
