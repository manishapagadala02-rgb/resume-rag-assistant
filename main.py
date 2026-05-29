import streamlit as st
import tempfile
import re
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq


# =====================================================
# LOAD ENV VARIABLES
# =====================================================

load_dotenv()


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Resume ATS Assistant",
    page_icon="🤖",
    layout="wide"
)


# =====================================================
# TITLE SECTION
# =====================================================

st.title("🤖 AI Resume ATS Assistant")

st.markdown("""
Analyze resumes using:

✅ RAG Architecture  
✅ Semantic Search  
✅ ATS Match Scoring  
✅ AI Candidate Analysis
""")


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📌 Tech Stack")

st.sidebar.info("""
- LangChain
- ChromaDB
- Groq API
- Llama 3
- Streamlit
- Sentence Transformers
""")

st.sidebar.success("✅ AI System Running")


# =====================================================
# SKILL EXTRACTION FUNCTION
# =====================================================

def extract_skills(text):

    skills = [
        "python",
        "java",
        "sql",
        "machine learning",
        "deep learning",
        "tensorflow",
        "pytorch",
        "scikit-learn",
        "power bi",
        "tableau",
        "aws",
        "azure",
        "docker",
        "kubernetes",
        "nlp",
        "react",
        "fastapi",
        "data science",
        "statistics",
        "excel",
        "communication",
        "leadership",
        "api",
        "pandas",
        "numpy"
    ]

    text = text.lower()

    found_skills = []

    for skill in skills:

        if re.search(r"\b" + re.escape(skill) + r"\b", text):
            found_skills.append(skill)

    return list(set(found_skills))


# =====================================================
# LOAD AND PROCESS PDF
# =====================================================

def process_resume(uploaded_file):

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())

        pdf_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    return chunks


# =====================================================
# CREATE VECTOR DATABASE
# =====================================================

def create_vector_store(chunks):

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return db


# =====================================================
# LOAD LLM
# =====================================================

def load_llm():

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

    return llm

# =====================================================
# GENERATE AI RESPONSE
# =====================================================

def generate_ai_response(
    context,
    job_description,
    user_question
):

    prompt = f"""
    You are an intelligent AI Resume ATS Assistant.

    Analyze the candidate professionally.

    Use ONLY the resume context below.

    Resume Context:
    {context}

    Job Description:
    {job_description}

    Question:
    {user_question}

    Professional Answer:
    """

    llm = load_llm()

    response = llm.invoke(prompt)

    return response.content


# =====================================================
# UI INPUT SECTION
# =====================================================

col1, col2 = st.columns(2)

with col1:

    uploaded_file = st.file_uploader(
        "📄 Upload Resume PDF",
        type=["pdf"]
    )

with col2:

    user_question = st.text_input(
        "❓ Ask AI About the Candidate"
    )

job_description = st.text_area(
    "🧾 Paste Job Description",
    height=250
)

analyze_button = st.button("🚀 Analyze Resume")


# =====================================================
# MAIN ANALYSIS
# =====================================================

if analyze_button:

    if uploaded_file is None:

        st.warning("Please upload a resume PDF.")

    elif user_question.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Analyzing Resume..."):

            # Process Resume
            chunks = process_resume(uploaded_file)

            # Create Vector Database
            db = create_vector_store(chunks)

            # Similarity Search
            results = db.similarity_search(
                user_question,
                k=3
            )

            # Combine retrieved text
            context = "\n\n".join(
                [doc.page_content for doc in results]
            )

            # =====================================================
            # ATS LOGIC
            # =====================================================

            resume_skills = extract_skills(context)

            jd_skills = extract_skills(job_description)

            matched_skills = list(
                set(resume_skills) & set(jd_skills)
            )

            missing_skills = list(
                set(jd_skills) - set(resume_skills)
            )

            # ATS SCORE
            if len(jd_skills) > 0:

                ats_score = int(
                    (len(matched_skills) / len(jd_skills)) * 100
                )

            else:
                ats_score = 0

            # =====================================================
            # AI ANALYSIS
            # =====================================================

            ai_response = generate_ai_response(
                context,
                job_description,
                user_question
            )

        # =====================================================
        # RESULTS SECTION
        # =====================================================

        st.divider()

        st.header("📊 ATS Analysis Dashboard")

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "ATS Score",
            f"{ats_score}%"
        )

        metric2.metric(
            "Matched Skills",
            len(matched_skills)
        )

        metric3.metric(
            "Missing Skills",
            len(missing_skills)
        )

        # Progress Bar
        st.progress(ats_score / 100)

        # =====================================================
        # AI RESPONSE
        # =====================================================

        st.subheader("🤖 AI Candidate Analysis")

        st.write(ai_response)

        # =====================================================
        # SKILLS SECTION
        # =====================================================

        skill_col1, skill_col2 = st.columns(2)

        with skill_col1:

            st.subheader("✅ Matching Skills")

            if matched_skills:
                st.write(matched_skills)
            else:
                st.write("No matching skills found.")

        with skill_col2:

            st.subheader("❌ Missing Skills")

            if missing_skills:
                st.write(missing_skills)
            else:
                st.write("No missing skills detected.")