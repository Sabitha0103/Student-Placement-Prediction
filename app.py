import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------
# CSS
# ---------------------------

st.markdown("""
<style>

.block-container{
    padding-top:1rem;
}

[data-testid="stMetric"]{
    background-color:#1e293b;
    padding:20px;
    border-radius:15px;
}

h1{
    text-align:center;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------
# LOAD DATASET
# ---------------------------

df = pd.read_csv("student_placement_synthetic.csv")

# ---------------------------
# PREPROCESS
# ---------------------------

temp_df = df.copy()

le = LabelEncoder()

temp_df["branch"] = le.fit_transform(
    temp_df["branch"]
)

temp_df["college_tier"] = le.fit_transform(
    temp_df["college_tier"]
)

X = temp_df.drop(
    ["placement_status", "salary_package_lpa"],
    axis=1
)

y = temp_df["placement_status"]

# ---------------------------
# TRAIN MODEL
# ---------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

predictions = model.predict(X)

accuracy = accuracy_score(
    y,
    predictions
)

# ---------------------------
# TITLE
# ---------------------------

st.title("🎓 Student Placement Prediction Dashboard")

# ---------------------------
# TABS
# ---------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "🎯 Prediction",
        "📊 Analytics",
        "📁 Dataset"
    ]
)

# =====================================================
# PREDICTION TAB
# =====================================================

with tab1:

    prediction_type = st.radio(
        "Choose Prediction Type",
        [
            "Single Student Prediction",
            "Entire Dataset Prediction"
        ]
    )

    # -------------------------
    # SINGLE STUDENT
    # -------------------------

    if prediction_type == "Single Student Prediction":

        col1, col2 = st.columns(2)

        with col1:

            branch = st.number_input(
                "Branch (0-4)",
                value=0
            )

            college_tier = st.number_input(
                "College Tier (0-2)",
                value=1
            )

            cgpa = st.number_input(
                "CGPA",
                min_value=0.0,
                max_value=10.0,
                value=8.0
            )

            backlogs = st.number_input(
                "Backlogs",
                min_value=0,
                value=0
            )

            coding_skills = st.slider(
                "Coding Skills",
                1,
                10,
                5
            )

            dsa_score = st.slider(
                "DSA Score",
                1,
                10,
                5
            )

        with col2:

            aptitude_score = st.slider(
                "Aptitude Score",
                0,
                100,
                50
            )

            communication_skills = st.slider(
                "Communication Skills",
                1,
                10,
                5
            )

            ml_knowledge = st.slider(
                "ML Knowledge",
                1,
                10,
                5
            )

            system_design = st.slider(
                "System Design",
                1,
                10,
                5
            )

            internships = st.number_input(
                "Internships",
                min_value=0,
                value=0
            )

            projects_count = st.number_input(
                "Projects Count",
                min_value=0,
                value=1
            )

        certifications = st.number_input(
            "Certifications",
            min_value=0,
            value=0
        )

        hackathons = st.number_input(
            "Hackathons",
            min_value=0,
            value=0
        )

        open_source_contributions = st.number_input(
            "Open Source Contributions",
            min_value=0,
            value=0
        )

        extracurriculars = st.number_input(
            "Extracurricular Activities",
            min_value=0,
            value=0
        )

        if st.button("Predict Placement"):

            data = pd.DataFrame([[
                branch,
                college_tier,
                cgpa,
                backlogs,
                coding_skills,
                dsa_score,
                aptitude_score,
                communication_skills,
                ml_knowledge,
                system_design,
                internships,
                projects_count,
                certifications,
                hackathons,
                open_source_contributions,
                extracurriculars
            ]], columns=X.columns)

            result = model.predict(data)

            if result[0] == 1:
                st.success(
                    "Likely to be Placed ✅"
                )
            else:
                st.error(
                    "Not Likely to be Placed ❌"
                )

    # -------------------------
    # ENTIRE DATASET
    # -------------------------

    else:

        if st.button(
            "Predict Entire Dataset"
        ):

            placed = (predictions == 1).sum()
            not_placed = (predictions == 0).sum()

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Placed",
                    int(placed)
                )

            with col2:
                st.metric(
                    "Predicted Not Placed",
                    int(not_placed)
                )

            results = pd.DataFrame({
                "Student_ID":
                range(
                    1,
                    len(predictions)+1
                ),
                "Predicted_Placement":
                predictions
            })

            st.dataframe(results)

# =====================================================
# ANALYTICS TAB
# =====================================================

with tab2:

    st.metric(
        "Model Accuracy",
        f"{accuracy*100:.2f}%"
    )

    placement_counts = df[
        "placement_status"
    ].value_counts()

    fig1 = px.pie(
        values=placement_counts.values,
        names=[
            "Placed",
            "Not Placed"
        ],
        title="Placement Distribution"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance":
        model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    fig2 = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# =====================================================
# DATASET TAB
# =====================================================

with tab3:

    st.subheader(
        "Dataset Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Students",
            len(df)
        )

    with col2:
        st.metric(
            "Total Features",
            len(df.columns)
        )

    with col3:
        st.metric(
            "Placed Students",
            int(df["placement_status"].sum())
        )

    st.markdown("---")

    st.subheader(
        "Available Columns"
    )

    st.write(
        list(df.columns)
    )