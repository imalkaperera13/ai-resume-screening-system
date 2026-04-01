import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

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

# UI
st.title("AI Resume Screening System - Multiple Resume Ranking")

uploaded_resumes = st.file_uploader(
    "Upload Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

job_description = st.text_area("Paste Job Description")

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

            results.append({
                "Resume Name": uploaded_resume.name,
                "Match Score (%)": score,
                "Matched Skills": ", ".join(matched_skills) if matched_skills else "None",
                "Missing Skills": ", ".join(missing_skills) if missing_skills else "None"
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)
        results_df.insert(0, "Rank", range(1, len(results_df) + 1))

        st.subheader("Resume Ranking")
        st.dataframe(results_df)

        csv = results_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="resume_ranking_results.csv",
            mime="text/csv",
        )

        st.subheader("Score Interpretation")
        for _, row in results_df.iterrows():
            score = row["Match Score (%)"]
            name = row["Resume Name"]

            if score >= 75:
                st.success(f"{name} — Strong Match ({score}%)")
            elif score >= 50:
                st.warning(f"{name} — Moderate Match ({score}%)")
            else:
                st.error(f"{name} — Low Match ({score}%)")

        st.subheader("Top Candidate")
        top_candidate = results_df.iloc[0]
        st.success(
            f"Best Match: {top_candidate['Resume Name']} with {top_candidate['Match Score (%)']}%"
        )
    else:
        st.warning("Please upload at least one resume and enter a job description.")