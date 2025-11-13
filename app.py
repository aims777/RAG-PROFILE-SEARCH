import streamlit as st
import pandas as pd
import time
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

CSV_PATH = "profiles.csv"
CHROMA_DIR = "./chroma_db"

# Create demo dataset
if not hasattr(st.session_state, "dataset_created"):
    # --- Create a demo dataset with 50 sample profiles ---
    names = [
        "Aiman Saba", "Pooja Sharma", "Rahul Verma", "Sneha Rao", "Vikram Singh",
        "Neha Gupta", "Arjun Rao", "Priya Nair", "Rohit Das", "Kavya Menon",
        "Dev Patel", "Sanya Mehta", "Akash Reddy", "Riya Kapoor", "Vivek Joshi",
        "Meera Iyer", "Aditya Jain", "Ishita Shah", "Manish R", "Ananya Bhat",
        "Shreya Kumar", "Ravi Raj", "Tarun S", "Deepika N", "Varun V",
        "Harini K", "Gaurav S", "Simran A", "Kiran T", "Dhruv P",
        "Tanya L", "Sahil Q", "Ritika D", "Mohan J", "Lakshmi V",
        "Snehal P", "Ankit R", "Bhavya S", "Rachit P", "Diya K",
        "Suraj T", "Keerthi N", "Anjali F", "Arav S", "Meghana B",
        "Charan C", "Irfan H", "Lavanya R", "Rajesh Y", "Suma M"
    ]

    locations = ["Bangalore", "Mysore", "Hyderabad", "Chennai", "Pune", "Delhi", "Mumbai"]
    skills_list = [
        "Python, Machine Learning, Data Analysis",
        "Java, React, Full Stack Development",
        "SQL, Tableau, Data Visualization",
        "Deep Learning, NLP, AI Research",
        "Excel, Business Intelligence, Power BI",
        "C++, Java, Backend Development",
        "HTML, CSS, JavaScript, UI/UX Design",
        "Data Engineering, Cloud, AWS",
        "Cybersecurity, Networking, Linux",
        "Finance, Data Analytics, Excel"
    ]
    summaries = [
        "Enthusiastic developer with hands-on experience in modern tools.",
        "Dedicated analyst passionate about insights and data-driven strategy.",
        "Team player with strong background in coding and research.",
        "Innovator who loves solving real-world technical problems.",
        "Self-motivated learner interested in emerging technologies."
    ]
    import random

    data = {
        "name": names,
        "email": [f"user{i+1}@example.com" for i in range(len(names))],
        "location": [random.choice(locations) for _ in range(len(names))],
        "skills": [random.choice(skills_list) for _ in range(len(names))],
        "experience_years": [random.randint(1, 8) for _ in range(len(names))],
        "summary": [random.choice(summaries) for _ in range(len(names))],
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
        with st.spinner("🔍 Searching profiles..."):
            query_emb = embed_model.encode(query).tolist()
            results = collection.query(query_embeddings=[query_emb], n_results=50)  # Fetch top 50

            import re
            # Capture numbers like "2+", "3 years", "at least 4", "5yr", etc.
            exp_match = re.search(r"(\d+)\s*\+?\s*(?:year|yr|yrs|experience|exp)?", query.lower())
            required_exp = int(exp_match.group(1)) if exp_match else None

            filtered_results = []
            for meta in results["metadatas"][0]:
                try:
                    exp = int(meta.get("experience_years", 0))
                    # ✅ Strictly greater than the mentioned number
                    if required_exp is None or exp > required_exp:
                        filtered_results.append(meta)
                except:
                    continue

            if not filtered_results:
                st.error("❌ No matching profiles found with the given experience.")
            else:
                st.success(f"✅ Found {len(filtered_results)} Matching Profiles (Filtered by Experience)")
                for i, meta in enumerate(filtered_results[:50]):  # show up to 50
                    st.markdown(f"### 👤 Result {i+1}")
                    st.write(f"**Name:** {meta.get('name')}")
                    st.write(f"**Email:** {meta.get('email')}")
                    st.write(f"**Location:** {meta.get('location')}")
                    st.write(f"**Skills:** {meta.get('skills')}")
                    st.write(f"**Experience:** {meta.get('experience_years')} years")
                    st.write(f"**Summary:** {meta.get('summary')}")
                    st.markdown("---")
