from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# 📌 Load model and encoders
model = joblib.load('model/fine_tune.pkl')
label_encoders = joblib.load('model/label_encoders.pkl')
target_encoder = joblib.load('model/target_encoder.pkl')  # ✅ Added for decoding output

@app.route('/')
def index():
    return render_template(
        'index.html',
        gender_labels=label_encoders['Applicant_Gender'].classes_,
        dept_labels=label_encoders['Department_Name'].classes_,
        lang_labels=label_encoders['Language'].classes_,
        type_labels=label_encoders['Appeal_Type'].classes_
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 📌 Get form inputs
        appeal_text = request.form['Appeal_Text']
        gender = request.form['Applicant_Gender']
        department = request.form['Department_Name']
        appeal_date = int(request.form['Appeal_Date'])
        language = request.form['Language']
        appeal_type = request.form['Appeal_Type']

        # 📌 Encode inputs
        gender_encoded = label_encoders['Applicant_Gender'].transform([gender])[0]
        dept_encoded = label_encoders['Department_Name'].transform([department])[0]
        lang_encoded = label_encoders['Language'].transform([language])[0]
        type_encoded = label_encoders['Appeal_Type'].transform([appeal_type])[0]

        # 📌 Create input DataFrame
        input_data = pd.DataFrame([{
            'Appeal_Text': appeal_text,
            'Applicant_Gender': gender_encoded,
            'Department_Name': dept_encoded,
            'Appeal_Date': appeal_date,
            'Language': lang_encoded,
            'Appeal_Type': type_encoded
        }])

        # 📌 Predict (encoded)
        pred_encoded = model.predict(input_data)[0]

        # ✅ Decode category name from encoded label
        pred_decoded = target_encoder.inverse_transform([pred_encoded])[0]

        result = f"📌 Predicted Appeal Category: {pred_decoded}"

    except Exception as e:
        result = f"❌ Error: {e}"

    return render_template(
        'index.html',
        prediction=result,
        gender_labels=label_encoders['Applicant_Gender'].classes_,
        dept_labels=label_encoders['Department_Name'].classes_,
        lang_labels=label_encoders['Language'].classes_,
        type_labels=label_encoders['Appeal_Type'].classes_
    )

if __name__ == '__main__':
    app.run(debug=True)
