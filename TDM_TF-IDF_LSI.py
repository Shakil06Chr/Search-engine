import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import normalize
import tkinter as tk
from tkinter import scrolledtext


# Function to clean text: lowercase, remove punctuation and numbers
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text


# Function to load and preprocess dataset
def load_and_preprocess_dataset(file_path):
    df = pd.read_csv(file_path)
    df['data_cleaned'] = df['data'].apply(clean_text)
    return df


# Calculate Term-Document Matrix
def create_tdm_matrix(df):
    vectorizer_tdm = CountVectorizer(stop_words='english')
    tdm = vectorizer_tdm.fit_transform(df['data_cleaned'])
    tdm_df = pd.DataFrame(tdm.toarray(), columns=vectorizer_tdm.get_feature_names_out())
    return tdm_df, vectorizer_tdm


# Calculate TF-IDF Matrix
def create_tfidf_matrix(df):
    vectorizer_tfidf = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer_tfidf.fit_transform(df['data_cleaned'])
    tfidf_df = pd.DataFrame(tfidf.toarray(), columns=vectorizer_tfidf.get_feature_names_out())
    return tfidf_df, vectorizer_tfidf


# Perform LSI
def perform_lsi(tfidf_matrix, num_topics=10):
    U, Sigma, VT = randomized_svd(tfidf_matrix, n_components=num_topics, n_iter=5, random_state=42)
    S_matrix = np.diag(Sigma)
    T_matrix = np.transpose(VT)
    D_matrix = U

    # Transform documents to LSI space
    lsi_documents = normalize(U @ np.diag(Sigma), axis=1)
    return lsi_documents, T_matrix, S_matrix


# Search Query for TDM and TF-IDF
def search_query(query, matrix_df, vectorizer):
    query_cleaned = clean_text(query)
    query_tokens = [token for token in query_cleaned.split() if token in vectorizer.get_feature_names_out()]

    if not query_tokens:
        return []

    document_sums = matrix_df[query_tokens].sum(axis=1)
    top_documents = document_sums.sort_values(ascending=False).head(10)
    results = [(rank + 1, doc_id + 1, weight) for rank, (doc_id, weight) in enumerate(top_documents.items())]
    return results


# Search Query for LSI
def search_lsi_query(query, tfidf_vectorizer, lsi_documents, T_matrix, S_matrix):
    query_cleaned = clean_text(query)
    query_vector = tfidf_vectorizer.transform([query_cleaned]).toarray()

    T_matrix_subset = T_matrix[:query_vector.shape[1], :]
    Sigma_inv = np.linalg.pinv(S_matrix)

    query_lsi = query_vector @ T_matrix_subset @ Sigma_inv
    query_lsi_normalized = normalize(query_lsi, axis=1)

    # Compute similarities
    similarities = cosine_similarity(query_lsi_normalized, lsi_documents)
    top_indices = similarities.argsort()[0, ::-1][:10]
    results = [(rank + 1, idx + 1, similarities[0, idx]) for rank, idx in enumerate(top_indices)]
    return results


# Identify Common and Uncommon Documents
def identify_common_and_uncommon(tdm_results, tfidf_results, lsi_results):
    tdm_doc_ids = {doc_id for _, doc_id, _ in tdm_results}
    tfidf_doc_ids = {doc_id for _, doc_id, _ in tfidf_results}
    lsi_doc_ids = {doc_id for _, doc_id, _ in lsi_results}

    # Find common documents (intersection of all three sets)
    common_docs = tdm_doc_ids & tfidf_doc_ids & lsi_doc_ids

    # Find uncommon documents (union minus intersection)
    all_docs = tdm_doc_ids | tfidf_doc_ids | lsi_doc_ids
    uncommon_docs = all_docs - common_docs

    return list(common_docs), list(uncommon_docs)


# Function to handle search
def on_search():
    query = query_entry.get()

    # Search TDM
    tdm_results = search_query(query, tdm_df_tdm, vectorizer_tdm)
    update_results_text(tdm_result_text, "TDM", tdm_results)

    # Search TF-IDF
    tfidf_results = search_query(query, tfidf_df_tfidf, vectorizer_tfidf)
    update_results_text(tfidf_result_text, "TF-IDF", tfidf_results)

    # Search LSI
    lsi_results = search_lsi_query(query, vectorizer_tfidf, lsi_documents, T_matrix, S_matrix)
    update_results_text(lsi_result_text, "LSI", lsi_results)

    # Identify common and uncommon documents
    common_docs, uncommon_docs = identify_common_and_uncommon(tdm_results, tfidf_results, lsi_results)

    # Print common and uncommon documents to the console
    print("\nCommon Documents (among TDM, TF-IDF, and LSI):")
    if common_docs:
        for doc_id in sorted(common_docs):
            print(f"Document {doc_id}")
    else:
        print("No common documents found.")

    print("\nUncommon Documents (among TDM, TF-IDF, and LSI):")
    if uncommon_docs:
        for doc_id in sorted(uncommon_docs):
            print(f"Document {doc_id}")
    else:
        print("No uncommon documents found.")


# Update results in the text box
def update_results_text(text_widget, method_name, results):
    text_widget.config(state=tk.NORMAL)
    text_widget.delete(1.0, tk.END)
    if not results:
        text_widget.insert(tk.END, f"No results found for {method_name}.\n")
    else:
        text_widget.insert(tk.END, f"Top 10 Results ({method_name}):\n")
        for rank, doc_id, weight in results:
            text_widget.insert(tk.END, f"{rank}. Document {doc_id}: Total Weight {weight:.4f}\n")
    text_widget.config(state=tk.DISABLED)


# Load dataset
file_path = 'bbc_data.csv'
df = load_and_preprocess_dataset(file_path).head(100)

# Generate TDM and TF-IDF matrices
tdm_df_tdm, vectorizer_tdm = create_tdm_matrix(df)
tfidf_df_tfidf, vectorizer_tfidf = create_tfidf_matrix(df)

# Perform LSI
lsi_documents, T_matrix, S_matrix = perform_lsi(tfidf_df_tfidf.values, num_topics=10)

# Build UI with Tkinter
root = tk.Tk()
root.title("Triple Search Engine: TDM, TF-IDF, and LSI")

# Query Input
query_label = tk.Label(root, text="Enter your query:")
query_label.pack()

query_entry = tk.Entry(root, width=50)
query_entry.pack()

search_button = tk.Button(root, text="Search", command=on_search)
search_button.pack()

# Results Frames
frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

# Left Side: TDM Results
tdm_frame = tk.Frame(frame)
tdm_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

tdm_label = tk.Label(tdm_frame, text="TDM Search Results")
tdm_label.pack()

tdm_result_text = scrolledtext.ScrolledText(tdm_frame, wrap=tk.WORD, width=40, height=20, state=tk.DISABLED)
tdm_result_text.pack()

# Middle Side: TF-IDF Results
tfidf_frame = tk.Frame(frame)
tfidf_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

tfidf_label = tk.Label(tfidf_frame, text="TF-IDF Search Results")
tfidf_label.pack()

tfidf_result_text = scrolledtext.ScrolledText(tfidf_frame, wrap=tk.WORD, width=40, height=20, state=tk.DISABLED)
tfidf_result_text.pack()

# Right Side: LSI Results
lsi_frame = tk.Frame(frame)
lsi_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

lsi_label = tk.Label(lsi_frame, text="LSI Search Results")
lsi_label.pack()

lsi_result_text = scrolledtext.ScrolledText(lsi_frame, wrap=tk.WORD, width=40, height=20, state=tk.DISABLED)
lsi_result_text.pack()

# Run the Tkinter main loop
root.mainloop()
