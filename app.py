import streamlit as st
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
st.title("AI Resume Screening System")

uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description")

if st.button("Analyze Resume"):
    if uploaded_resume and job_description:
        resume_text = extract_text_from_pdf(uploaded_resume)

        cleaned_resume = clean_text(resume_text)
        cleaned_jd = clean_text(job_description)

        score = calculate_similarity(cleaned_resume, cleaned_jd)

        resume_skills = extract_skills(cleaned_resume, SKILLS_DB)
        jd_skills = extract_skills(cleaned_jd, SKILLS_DB)

        matched_skills = sorted(list(set(resume_skills) & set(jd_skills)))
        missing_skills = sorted(list(set(jd_skills) - set(resume_skills)))

        st.subheader("Results")
        if score >= 75:
            st.success(f"Strong Match: {score}%")
        elif score >= 50:
            st.warning(f"Moderate Match: {score}%")
        else:
            st.error(f"Low Match: {score}%")

        st.subheader("Matched Skills")
        if matched_skills:
            for skill in matched_skills:
                st.write("✔", skill)
            
        else:
            st.write("No matched skills found")

        st.subheader("Missing Skills")
        if missing_skills:
            for skill in missing_skills:
                st.write("✖", skill)
        else:
            st.write("No missing skills")

        st.subheader("Extracted Resume Text")
        st.write(resume_text)

        st.subheader("Cleaned Resume Text")
        st.write(cleaned_resume)

        st.subheader("Cleaned Job Description")
        st.write(cleaned_jd)
    else:
        st.warning("Upload resume and enter job description")