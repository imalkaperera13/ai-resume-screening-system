# AI Resume Screening System

An AI-powered resume screening system that compares multiple resumes with a job description and ranks candidates based on text similarity and skill matching.

## Features

- Upload multiple resumes in PDF format
- Paste a job description
- Extract text from resumes
- Clean and preprocess text
- Calculate candidate match scores
- Detect matched skills
- Detect missing skills
- Rank candidates
- Show top candidate
- Export results as CSV
- Visualize scores with a bar chart

## Tech Stack

- Python
- Streamlit
- Pandas
- PyPDF2
- Scikit-learn
- Matplotlib

## AI Concepts Used

- Natural Language Processing (NLP)
- Text preprocessing
- TF-IDF Vectorization
- Cosine Similarity
- Skill extraction
- Resume ranking

## How It Works

1. Upload one or more resumes in PDF format
2. Enter a job description
3. The system extracts resume text
4. Resume text and job description are cleaned
5. The system compares resumes with the job description
6. Skills are matched against the job description
7. Candidates are ranked and displayed in a dashboard

## Project Output

- Resume ranking table
- Match score
- Matched skills
- Missing skills
- Candidate recommendation
- Downloadable CSV report

## Future Improvements

- Semantic similarity using embeddings
- Better skill extraction with NLP libraries
- Recruiter login system
- Database integration
- Deployment to cloud

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py