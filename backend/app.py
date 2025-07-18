import pandas as pd
import numpy as np
import re
import os
import joblib
import random # For generating non-relevant pairs

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, average_precision_score

from sentence_transformers import SentenceTransformer

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Configuration ---
MODEL_ARTIFACTS_DIR = 'model_artifacts'
TFIDF_VECTORIZER_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'tfidf_vectorizer.pkl')
SVD_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'svd_model.pkl')
FCNN_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'hybrid_fcnn_model.h5')

# CISI Dataset file paths (assuming they are in the same directory as app.py)
CISI_ALL_PATH = 'CISI.ALL'
CISI_QRY_PATH = 'CISI.QRY'
CISI_REL_PATH = 'CISI.REL'

# Global variables to store loaded models and components
tfidf_vectorizer = None
svd_model = None
bert_model = None
fcnn_model = None
documents_for_search_df = None # Will store the actual CISI documents for searching

# --- CISI Data Parsing Functions ---
def parse_cisi_all(file_path):
    """Parses CISI.ALL file into a DataFrame."""
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        doc_id = None
        doc_content = []
        for line in f:
            line = line.strip()
            if line.startswith('.I'):
                if doc_id is not None:
                    documents.append({'doc_id': int(doc_id), 'content': ' '.join(doc_content).strip()})
                doc_id = line.split(' ')[1]
                doc_content = []
            elif line.startswith('.T') or line.startswith('.A') or line.startswith('.B') or line.startswith('.W'):
                # Start collecting content
                pass
            elif line.startswith('.'): # Another metadata tag, end of content for previous field
                pass
            else:
                doc_content.append(line)
        if doc_id is not None: # Add the last document
            documents.append({'doc_id': int(doc_id), 'content': ' '.join(doc_content).strip()})
    return pd.DataFrame(documents)

def parse_cisi_qry(file_path):
    """Parses CISI.QRY file into a DataFrame."""
    queries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        query_id = None
        query_text = []
        for line in f:
            line = line.strip()
            if line.startswith('.I'):
                if query_id is not None:
                    queries.append({'query_id': int(query_id), 'query_text': ' '.join(query_text).strip()})
                query_id = line.split(' ')[1]
                query_text = []
            elif line.startswith('.W'): # .W contains the query text
                pass
            elif line.startswith('.'): # Another metadata tag
                pass
            else:
                query_text.append(line)
        if query_id is not None: # Add the last query
            queries.append({'query_id': int(query_id), 'query_text': ' '.join(query_text).strip()})
    return pd.DataFrame(queries)

def parse_cisi_rel(file_path):
    """Parses CISI.REL file into a DataFrame."""
    relevance_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                query_id = int(parts[0])
                doc_id = int(parts[1])
                relevance_data.append({'query_id': query_id, 'doc_id': doc_id, 'relevant': 1})
    return pd.DataFrame(relevance_data)

def load_cisi_data():
    """Loads and parses all CISI data files."""
    print(f"Loading CISI documents from {CISI_ALL_PATH}...")
    df_docs = parse_cisi_all(CISI_ALL_PATH)
    print(f"Loaded {len(df_docs)} documents.")

    print(f"Loading CISI queries from {CISI_QRY_PATH}...")
    df_queries = parse_cisi_qry(CISI_QRY_PATH)
    print(f"Loaded {len(df_queries)} queries.")

    print(f"Loading CISI relevance data from {CISI_REL_PATH}...")
    df_relevance = parse_cisi_rel(CISI_REL_PATH)
    print(f"Loaded {len(df_relevance)} relevant pairs.")

    return df_docs, df_queries, df_relevance

def generate_qa_pairs(df_docs, df_queries, df_relevance):
    """
    Generates balanced query-document pairs (relevant and non-relevant).
    The journal mentions 3114 relevant and 3114 non-relevant pairs.
    """
    # Create a set of all possible (query_id, doc_id) pairs for quick lookup
    all_possible_pairs = set()
    for q_id in df_queries['query_id'].unique():
        for d_id in df_docs['doc_id'].unique():
            all_possible_pairs.add((q_id, d_id))

    # Get relevant pairs from df_relevance
    relevant_pairs = set(tuple(row) for row in df_relevance[['query_id', 'doc_id']].values)

    # Identify non-relevant pairs
    non_relevant_pairs = list(all_possible_pairs - relevant_pairs)
    
    # Balance the non-relevant pairs to match the number of relevant pairs
    # If there are more non-relevant pairs than relevant, sample them
    if len(non_relevant_pairs) > len(relevant_pairs):
        non_relevant_pairs_sampled = random.sample(non_relevant_pairs, len(relevant_pairs))
    else:
        non_relevant_pairs_sampled = non_relevant_pairs # Use all if fewer or equal

    print(f"Number of relevant pairs: {len(relevant_pairs)}")
    print(f"Number of sampled non-relevant pairs: {len(non_relevant_pairs_sampled)}")

    # Combine relevant and sampled non-relevant pairs
    final_pairs = []
    for q_id, d_id in relevant_pairs:
        final_pairs.append({'query_id': q_id, 'doc_id': d_id, 'relevant': 1})
    for q_id, d_id in non_relevant_pairs_sampled:
        final_pairs.append({'query_id': q_id, 'doc_id': d_id, 'relevant': 0})

    df_final_pairs = pd.DataFrame(final_pairs)

    # Merge with actual query text and document content
    df_final_pairs = df_final_pairs.merge(df_queries, on='query_id', how='left')
    df_final_pairs = df_final_pairs.merge(df_docs, on='doc_id', how='left')

    # Add 'title' column to match frontend expectation (using doc_id as title if not available)
    if 'title' not in df_final_pairs.columns:
         df_final_pairs['title'] = df_final_pairs['doc_id'].apply(lambda x: f"Document {x}")

    print(f"Total QA pairs generated: {len(df_final_pairs)}")
    return df_final_pairs


# --- Text Preprocessing Function ---
def preprocess_text(text):
    """
    Converts text to lowercase, removes punctuation,
    and converts multiple spaces/newlines to a single space.
    """
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Model Training Function ---
def train_and_save_model():
    global tfidf_vectorizer, svd_model, bert_model, fcnn_model, documents_for_search_df

    print("Loading CISI data for training...")
    df_docs_raw, df_queries_raw, df_relevance_raw = load_cisi_data()
    df_qa_pairs = generate_qa_pairs(df_docs_raw, df_queries_raw, df_relevance_raw)

    # Set documents_for_search_df for later use in /search endpoint
    # Ensure 'title' column is present for frontend
    documents_for_search_df = df_docs_raw.copy()
    if 'title' not in documents_for_search_df.columns:
        documents_for_search_df['title'] = documents_for_search_df['doc_id'].apply(lambda x: f"Document {x}")


    df_qa_pairs['query_text_processed'] = df_qa_pairs['query_text'].apply(preprocess_text)
    df_qa_pairs['doc_content_processed'] = df_qa_pairs['content'].apply(preprocess_text) # Use 'content' column from df_docs

    queries_text = df_qa_pairs['query_text_processed'].tolist()
    docs_text = df_qa_pairs['doc_content_processed'].tolist()

    # TF-IDF Vectorization
    print("Training TF-IDF Vectorizer...")
    all_texts_for_tfidf = queries_text + docs_text
    tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Revert to 5000 for real data
    tfidf_vectorizer.fit(all_texts_for_tfidf)
    X_q_tfidf = tfidf_vectorizer.transform(queries_text).toarray()
    X_d_tfidf = tfidf_vectorizer.transform(docs_text).toarray()
    X_tfidf_combined_raw = np.concatenate([X_q_tfidf, X_d_tfidf], axis=1)
    print(f"Shape of raw combined TF-IDF vectors: {X_tfidf_combined_raw.shape}")


    # Dimensionality Reduction TF-IDF
    print("Training TruncatedSVD for TF-IDF dimension reduction...")
    svd_model = TruncatedSVD(n_components=768, random_state=42) # Revert to 768 for real data
    X_tfidf_reduced = svd_model.fit_transform(X_tfidf_combined_raw)
    print(f"Shape of reduced TF-IDF vectors: {X_tfidf_reduced.shape}")


    # BERT Embedding
    print("Loading and using SentenceTransformer (BERT)...")
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    X_q_bert = bert_model.encode(queries_text, convert_to_tensor=False, show_progress_bar=False)
    X_d_bert = bert_model.encode(docs_text, convert_to_tensor=False, show_progress_bar=False)
    X_bert_combined = np.concatenate([X_q_bert, X_d_bert], axis=1)
    print(f"Shape of combined BERT embeddings: {X_bert_combined.shape}")

    # Feature Combination (Fusion)
    print("Combining BERT and TF-IDF features...")
    bert_weight = 0.9
    tfidf_weight = 0.1
    X_combined = (bert_weight * X_bert_combined) + (tfidf_weight * X_tfidf_reduced)
    print(f"Shape of final combined features (model input): {X_combined.shape}")

    # Dataset Split
    y = df_qa_pairs['relevant'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model Architecture Development (FCNN)
    print("Building and training FCNN model...")
    fcnn_model = Sequential([
        Input(shape=(X_combined.shape[1],)),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    fcnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = fcnn_model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=0)

    # Model Evaluation
    y_pred_proba = fcnn_model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    metrics = {
        'train_accuracy': float(history.history['accuracy'][-1]),
        'test_accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'map': float(average_precision_score(y_test, y_pred_proba))
    }

    # Saving Model and Components
    os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(tfidf_vectorizer, TFIDF_VECTORIZER_PATH)
    joblib.dump(svd_model, SVD_MODEL_PATH)
    fcnn_model.save(FCNN_MODEL_PATH)
    print(f"Model and components saved to directory: {MODEL_ARTIFACTS_DIR}")
    return metrics

# --- Function to Load Models ---
def load_models():
    global tfidf_vectorizer, svd_model, bert_model, fcnn_model, documents_for_search_df
    try:
        tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        svd_model = joblib.load(SVD_MODEL_PATH)
        fcnn_model = load_model(FCNN_MODEL_PATH)
        bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load actual CISI documents for search
        df_docs_raw, _, _ = load_cisi_data()
        documents_for_search_df = df_docs_raw.copy()
        # Add 'title' column if not present (CISI.ALL doesn't have explicit titles, so use content snippet or doc_id)
        if 'title' not in documents_for_search_df.columns:
            documents_for_search_df['title'] = documents_for_search_df['doc_id'].apply(lambda x: f"Document {x}")


        print("Models and components loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Models not trained or not found. Please run /train_model first.")
        return False

# --- Relevance Prediction Function ---
def predict_relevance(query_text, doc_content):
    if not all([tfidf_vectorizer, svd_model, bert_model, fcnn_model]):
        print("Models not loaded. Attempting to load...")
        if not load_models():
            return 0.0

    processed_query = preprocess_text(query_text)
    processed_doc = preprocess_text(doc_content)

    # TF-IDF
    X_q_tfidf_new = tfidf_vectorizer.transform([processed_query]).toarray()
    X_d_tfidf_new = tfidf_vectorizer.transform([processed_doc]).toarray()
    X_tfidf_combined_new = np.concatenate([X_q_tfidf_new, X_d_tfidf_new], axis=1)
    X_tfidf_reduced_new = svd_model.transform(X_tfidf_combined_new)

    # BERT
    X_q_bert_new = bert_model.encode([processed_query], convert_to_tensor=False)
    X_d_bert_new = bert_model.encode([processed_doc], convert_to_tensor=False)
    X_bert_combined_new = np.concatenate([X_q_bert_new, X_d_bert_new], axis=1)

    # Feature Combination
    bert_weight = 0.9
    tfidf_weight = 0.1
    X_combined_new = (bert_weight * X_bert_combined_new) + (tfidf_weight * X_tfidf_reduced_new)

    # Prediction
    prediction_proba = fcnn_model.predict(X_combined_new)[0][0]
    return prediction_proba

# --- API Endpoints ---

@app.route('/train_model', methods=['GET'])
def train_model_endpoint():
    """
    Endpoint to train the model and save artifacts.
    This should be run ONCE before using the /search endpoint.
    """
    print("Starting model training...")
    try:
        metrics = train_and_save_model()
        return jsonify({"status": "Model trained and saved successfully", "metrics": metrics})
    except FileNotFoundError as e:
        return jsonify({"error": f"CISI data file not found: {e}. Please ensure CISI.ALL, CISI.QRY, CISI.REL are in the backend directory."}), 500
    except Exception as e:
        return jsonify({"error": f"An error occurred during training: {e}"}), 500


@app.route('/search', methods=['POST'])
def search_endpoint():
    """
    Endpoint to perform document search based on the given query.
    """
    data = request.json
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    if documents_for_search_df is None:
        if not load_models():
            return jsonify({"error": "Models not loaded or found. Please train the model first."}), 500

    search_results = []
    # Iterate through all available documents and predict relevance
    for idx, doc_row in documents_for_search_df.iterrows():
        relevance_score = predict_relevance(query, doc_row['content'])
        search_results.append({
            'id': doc_row['doc_id'],
            'title': doc_row['title'], # Use the 'title' column
            'content': doc_row['content'],
            'relevanceScore': float(relevance_score)
        })

    # Sort by relevance score (highest first)
    search_results.sort(key=lambda x: x['relevanceScore'], reverse=True)

    return jsonify(search_results)

# --- Main Execution ---
if __name__ == '__main__':
    # Try to load models at startup. If unsuccessful, instruct to train.
    if not load_models():
        print("\n----------------------------------------------------------------------------------")
        print("ATTENTION: Models not trained or not found.")
        print("Please ensure CISI.ALL, CISI.QRY, CISI.REL are in the 'backend' directory.")
        print("Then, open your browser and visit http://127.0.0.1:5000/train_model")
        print("to train the model before using the search feature.")
        print("----------------------------------------------------------------------------------")
    app.run(debug=True)
