import os
import subprocess
import sys
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "student_performance_model.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "student_performance_demo.csv")

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Prediction System")
st.caption("Final-year Machine Learning project | Educational decision-support demonstration")

if not os.path.exists(MODEL_PATH):
    with st.spinner("Training the machine learning model for first-time use..."):
        result = subprocess.run([sys.executable, os.path.join(BASE_DIR, "train_model.py")],
                                capture_output=True, text=True)
    if result.returncode != 0:
        st.error("Model training failed. Check the terminal output and dependencies.")
        st.code(result.stderr)
        st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]
metrics = bundle["metrics"]
data = pd.read_csv(DATA_PATH)

tab1, tab2, tab3, tab4 = st.tabs(["Predict Performance", "Dataset Explorer", "Model Evaluation", "About Project"])

with tab1:
    st.subheader("Enter student information")
    st.write("Use realistic values. The prediction is an estimate, not a guaranteed result.")
    c1, c2, c3 = st.columns(3)
    with c1:
        study = st.slider("Study hours per day", 0.0, 16.0, 4.0, 0.5)
        attendance = st.slider("Attendance (%)", 0.0, 100.0, 80.0, 1.0)
    with c2:
        previous = st.slider("Previous exam score", 0.0, 100.0, 65.0, 1.0)
        assignment = st.slider("Assignment score", 0.0, 100.0, 70.0, 1.0)
    with c3:
        sleep = st.slider("Sleep hours per day", 0.0, 16.0, 7.0, 0.5)
        participation = st.slider("Class participation (1–5)", 1, 5, 3)
    if st.button("Predict Final Exam Score", type="primary"):
        row = pd.DataFrame([{
            "study_hours_per_day": study,
            "attendance_percentage": attendance,
            "previous_exam_score": previous,
            "assignment_score": assignment,
            "sleep_hours_per_day": sleep,
            "class_participation": participation
        }])[features]
        prediction = float(model.predict(row)[0])
        prediction = max(0.0, min(100.0, prediction))
        st.metric("Estimated final exam score", f"{prediction:.1f} / 100")
        if prediction >= 75:
            category = "High predicted score"
        elif prediction >= 50:
            category = "Moderate predicted score"
        else:
            category = "Support may be helpful"
        st.info(f"Indicative category: **{category}**")
        st.caption("This category is a simple project threshold, not a validated academic classification. Do not use this output to make consequential decisions about students.")

with tab2:
    st.subheader("Demonstration dataset")
    st.caption("The included CSV is simulated data created for demonstration. It is not real student records.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows", f"{len(data):,}")
    m2.metric("Input features", str(len(features)))
    m3.metric("Missing cells", str(int(data.isna().sum().sum())))
    st.dataframe(data.head(20), use_container_width=True)
    st.download_button("Download demo dataset CSV", data=data.to_csv(index=False).encode("utf-8"),
                       file_name="student_performance_demo.csv", mime="text/csv")
    st.subheader("Study hours vs final score")
    fig, ax = plt.subplots()
    ax.scatter(data["study_hours_per_day"], data["final_exam_score"], alpha=0.55)
    ax.set_xlabel("Study hours per day")
    ax.set_ylabel("Final exam score")
    ax.set_title("Demonstration data: study hours and final score")
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("Score distribution")
    fig, ax = plt.subplots()
    ax.hist(data["final_exam_score"], bins=20, edgecolor="black")
    ax.set_xlabel("Final exam score")
    ax.set_ylabel("Number of records")
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    st.subheader("Held-out test-set metrics")
    a, b, c = st.columns(3)
    a.metric("MAE", f'{metrics["mae"]:.2f}')
    b.metric("RMSE", f'{metrics["rmse"]:.2f}')
    c.metric("R²", f'{metrics["r2"]:.3f}')
    st.write("Metrics are calculated on a test split held out during model training.")
    st.write("MAE and RMSE are measured in score points. Lower is generally better. R² is interpreted relative to the dataset and baseline.")
    st.warning("These results measure performance on simulated data only. They do not establish accuracy on real students.")
    st.subheader("Model comparison")
    st.dataframe(pd.DataFrame(bundle["comparison"]), use_container_width=True)

with tab4:
    st.subheader("Project overview")
    st.markdown("""
**Problem statement:** Academic staff may want to explore whether study-related indicators are associated with final examination scores.

**Approach:** Supervised regression using student-related input features, an 80/20 train-test split, and comparison of Linear Regression with Random Forest Regression.

**Technology stack:** Python, Pandas, NumPy, Scikit-learn, Streamlit, Matplotlib, Joblib.

**Limitations:** The bundled dataset is synthetic; the relationships are simplified; omitted factors may matter; correlation does not imply causation. Use real data only with proper authorization, privacy safeguards, and documentation.

**Responsible use:** Treat the prediction as an educational demonstration, not a definitive judgement about a student.
""")
