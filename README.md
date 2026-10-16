# Student Performance Prediction System Using Machine Learning

A final-year project demonstrating supervised regression with a Streamlit web interface.

## Important dataset note
`data/student_performance_demo.csv` is **synthetic demonstration data**, not real student data. Its metrics are for software demonstration only. For a formal empirical study, obtain a permitted real dataset, document its source and license, review privacy requirements, and re-evaluate the model.

## Features
- Student score prediction interface
- Demo dataset viewer and CSV download
- Data visualizations
- Linear Regression and Random Forest comparison
- MAE, RMSE, and R² evaluation
- Saved model artifact
- Project report in `reports/`

## Requirements
Python 3.10 or later is recommended.

## Run locally (Windows)
1. Extract the ZIP.
2. Open a terminal in the extracted project folder.
3. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

4. Install packages:

   ```bash
   pip install -r requirements.txt
   ```

5. Train and evaluate:

   ```bash
   python train_model.py
   ```

6. Start the application:

   ```bash
   streamlit run app.py
   ```

Your browser should open the local application. If it does not, use the local URL printed by Streamlit.

## Run on macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Dataset schema
- `study_hours_per_day`: study hours per day
- `attendance_percentage`: attendance percent
- `previous_exam_score`: previous exam score, 0–100
- `assignment_score`: assignment score, 0–100
- `sleep_hours_per_day`: sleep hours per day
- `class_participation`: ordinal score from 1 to 5
- `final_exam_score`: target, 0–100

To use another dataset, update `data/student_performance_demo.csv` while preserving these columns, or update `features` and `target` in `train_model.py` and `app.py`.

## Methodology
1. Load and validate numeric columns.
2. Separate features and target.
3. Split data into training (80%) and test (20%) subsets.
4. Fit Linear Regression and Random Forest Regression pipelines.
5. Calculate MAE, RMSE and R².
6. Save the selected model with Joblib.
7. Use the saved model for interactive predictions.

For a rigorous research project, select models using cross-validation on training data, then report final performance once on a separate test set.

## Responsible use
This project is an educational prototype. Do not use synthetic-data predictions to assess actual students. Predictions should not be used as the sole basis for consequential academic decisions. Real student data requires appropriate authorization, minimization, and protection.
