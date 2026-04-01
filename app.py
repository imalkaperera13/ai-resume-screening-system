import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Resume Screening System", layout="wide")

SKILLS_DB = [
    "python", "java", "c++", "sql",
    "machine learning", "deep learning",
    "aws", "azure", "gcp",
    "docker", "kubernetes", "terraform",
    "jenkins", "github actions", "ci/cd",
    "linux", "bash", "shell scripting",
    "ansible", "prometheus", "grafana",
    "microservices", "rest api",
    "react", "node", "spring boot"
]

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "
    return text

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Extract skills
def extract_skills(text, skills_db):
    found_skills = []
    text = text.lower()

    for skill in skills_db:
        if skill.lower() in text:
            found_skills.append(skill)

    return found_skills

# Calculate similarity
def calculate_similarity(resume_text, job_description):
    documents = [resume_text, job_description]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity * 100, 2)

# Recommendation logic
def get_recommendation(score):
    if score >= 75:
        return "Shortlist"
    elif score >= 50:
        return "Consider"
    else:
        return "Reject for now"

# UI
st.title("AI Resume Screening System")
st.write("Upload multiple resumes, compare them with a job description, and rank the best candidates.")

uploaded_resumes = st.file_uploader(
    "Upload Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

job_description = st.text_area("Paste Job Description", height=200)

if st.button("Analyze Resumes"):
    if uploaded_resumes and job_description:
        cleaned_jd = clean_text(job_description)
        jd_skills = extract_skills(cleaned_jd, SKILLS_DB)

        results = []

        for uploaded_resume in uploaded_resumes:
            resume_text = extract_text_from_pdf(uploaded_resume)
            cleaned_resume = clean_text(resume_text)

            score = calculate_similarity(cleaned_resume, cleaned_jd)

            resume_skills = extract_skills(cleaned_resume, SKILLS_DB)
            matched_skills = sorted(list(set(resume_skills) & set(jd_skills)))
            missing_skills = sorted(list(set(jd_skills) - set(resume_skills)))
            recommendation = get_recommendation(score)

            results.append({
                "Resume Name": uploaded_resume.name,
                "Match Score (%)": score,
                "Matched Skills": ", ".join(matched_skills) if matched_skills else "None",
                "Missing Skills": ", ".join(missing_skills) if missing_skills else "None",
                "Recommendation": recommendation
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)
        results_df.insert(0, "Rank", range(1, len(results_df) + 1))

        top_candidate = results_df.iloc[0]
        avg_score = round(results_df["Match Score (%)"].mean(), 2)

        # Dashboard Summary
        st.subheader("Dashboard Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Resumes", len(results_df))
        col2.metric("Top Score", f"{top_candidate['Match Score (%)']}%")
        col3.metric("Average Score", f"{avg_score}%")

        # JD Skills
        st.subheader("Job Description Skills Detected")
        if jd_skills:
            for skill in sorted(jd_skills):
                st.write("•", skill)
        else:
            st.write("No known skills detected from the job description.")

        # Ranking Table
        st.subheader("Resume Ranking")
        st.dataframe(results_df, use_container_width=True)

        # CSV Download
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="resume_ranking_results.csv",
            mime="text/csv",
        )

        # Bar Chart
        st.subheader("Candidate Score Chart")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(results_df["Resume Name"], results_df["Match Score (%)"])
        ax.set_xlabel("Resume Name")
        ax.set_ylabel("Match Score (%)")
        ax.set_title("Resume Ranking by Match Score")
        plt.xticks(rotation=30)
        st.pyplot(fig)

        # Top 3 Candidates
        st.subheader("Top 3 Candidates")
        top_3 = results_df.head(3)
        for _, row in top_3.iterrows():
            st.info(
                f"Rank {row['Rank']} - {row['Resume Name']} - {row['Match Score (%)']}% - {row['Recommendation']}"
            )

        # Candidate Decisions
        st.subheader("Candidate Decisions")
        for _, row in results_df.iterrows():
            name = row["Resume Name"]
            score = row["Match Score (%)"]
            recommendation = row["Recommendation"]
            missing_skills = row["Missing Skills"]

            if recommendation == "Shortlist":
                st.success(f"{name} — {recommendation} ({score}%)")
            elif recommendation == "Consider":
                st.warning(f"{name} — {recommendation} ({score}%)")
            else:
                st.error(f"{name} — {recommendation} ({score}%)")

            st.write(f"Missing Skills: {missing_skills}")

        # Best Candidate
        st.subheader("Top Candidate")
        st.success(
            f"Best Match: {top_candidate['Resume Name']} with {top_candidate['Match Score (%)']}% "
            f"({top_candidate['Recommendation']})"
        )

    else:
        st.warning("Please upload at least one resume and enter a job description.")