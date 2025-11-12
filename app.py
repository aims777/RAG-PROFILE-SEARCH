import streamlit as st
import pandas as pd
import time
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

CSV_PATH = "profiles.csv"
CHROMA_DIR = "./chroma_db"

# Create demo dataset
if not hasattr(st.session_state, "dataset_created"):
    data = {
        "name": ["Aiman Saba", "Pooja Sharma", "Rahul Verma", "Sneha Rao", "Vikram Singh"],
        "email": [
            "aiman@example.com",
            "pooja@example.com",
            "rahul@example.com",
            "sneha@example.com",
            "vikram@example.com",
        ],
        "location": ["Bangalore", "Mysore", "Hyderabad", "Chennai", "Pune"],
        "skills": [
            "Python, Machine Learning, Data Analysis",
            "Java, React, Full Stack Development",
            "SQL, Tableau, Data Visualization",
            "Deep Learning, NLP, AI Research",
            "Excel, Business Intelligence, Power BI",
        ],
        "experience_years": [2, 3, 4, 5, 2],
        "summary": [
            "Data enthusiast with experience in Python and ML.",
            "Full-stack developer skilled in Java and React.",
            "Analyst experienced in data visualization and SQL.",
            "AI researcher passionate about NLP and deep learning.",
            "Business analyst skilled in Excel and BI tools.",
        ],
    }
    df = pd.DataFrame(data)
    df["raw_text"] = df["name"] + " " + df["skills"] + " " + df["summary"]
    df.to_csv(CSV_PATH, index=False)
    st.session_state.dataset_created = True

@st.cache_resource
def load_profiles():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    df = pd.read_csv(CSV_PATH)
    client = PersistentClient(path=CHROMA_DIR)
    collection = client.create_collection(name="profiles_" + str(int(time.time())))

    for i, row in df.iterrows():
        text = str(row.get("raw_text", ""))
        metadata = row.to_dict()
        embedding = embed_model.encode(text).tolist()
        collection.add(
            ids=[str(i)],
            documents=[text],
            metadatas=[metadata],
            embeddings=[embedding]
        )
    return embed_model, collection, df

embed_model, collection, df = load_profiles()

st.set_page_config(page_title="RAG Profile Search", layout="centered")
st.title("🔎 RAG Profile Search")
st.caption("Search candidate profiles using semantic similarity (RAG-based).")

query = st.text_input("Enter skills, roles, or keywords:")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Searching profiles..."):
            query_emb = embed_model.encode(query).tolist()
            results = collection.query(query_embeddings=[query_emb], n_results=3)

            if not results["metadatas"] or len(results["metadatas"][0]) == 0:
                st.error("❌ No matching profiles found.")
            else:
                st.success("✅ Top Matching Profiles")
                for i, meta in enumerate(results["metadatas"][0]):
                    st.markdown(f"### 👤 Result {i+1}")
                    st.write(f"**Name:** {meta.get('name')}")
                    st.write(f"**Email:** {meta.get('email')}")
                    st.write(f"**Location:** {meta.get('location')}")
                    st.write(f"**Skills:** {meta.get('skills')}")
                    st.write(f"**Experience:** {meta.get('experience_years')} years")
                    st.write(f"**Summary:** {meta.get('summary')}")
                    st.markdown("---")
