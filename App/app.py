from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the fine-tuned model and encoders
model = joblib.load('model/fine_tune.pkl')
label_encoders = joblib.load('model/label_encoders.pkl')
target_encoder = joblib.load('model/target_encoder.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict_category():
    prediction = None
    input_values = {}

    # Load dropdown options from encoders
    gender_labels = label_encoders['Applicant_Gender'].classes_
    dept_labels = label_encoders['Department_Name'].classes_
    lang_labels = label_encoders['Language'].classes_
    type_labels = label_encoders['Appeal_Type'].classes_

    if request.method == 'POST':
        try:
            # Get form data
            appeal_text = request.form['Appeal_Text']
            gender = request.form['Applicant_Gender']
            department = request.form['Department_Name']
            appeal_date = int(request.form['Appeal_Date'])
            language = request.form['Language']
            appeal_type = request.form['Appeal_Type']

            # Save user inputs for rendering back
            input_values = {
                'Appeal_Text': appeal_text,
                'Applicant_Gender': gender,
                'Department_Name': department,
                'Appeal_Date': appeal_date,
                'Language': language,
                'Appeal_Type': appeal_type
            }

            # Encode inputs
            gender_encoded = label_encoders['Applicant_Gender'].transform([gender])[0]
            dept_encoded = label_encoders['Department_Name'].transform([department])[0]
            lang_encoded = label_encoders['Language'].transform([language])[0]
            type_encoded = label_encoders['Appeal_Type'].transform([appeal_type])[0]

            # Prepare input dataframe
            input_df = pd.DataFrame([{
                'Appeal_Text': appeal_text,
                'Applicant_Gender': gender_encoded,
                'Department_Name': dept_encoded,
                'Appeal_Date': appeal_date,
                'Language': lang_encoded,
                'Appeal_Type': type_encoded
            }])

            # Make prediction
            pred_encoded = model.predict(input_df)[0]
            pred_label = target_encoder.inverse_transform([pred_encoded])[0]
            prediction = f"📌 Predicted Appeal Category: {pred_label}"

        except Exception as e:
            prediction = f"❌ Error: {str(e)}"

    return render_template(
        'index.html',
        prediction=prediction,
        input_values=input_values,
        gender_labels=gender_labels,
        dept_labels=dept_labels,
        lang_labels=lang_labels,
        type_labels=type_labels
    )

if __name__ == '__main__':
    app.run(debug=True)
