from datetime import datetime
import ast
from rapidfuzz import fuzz, process
import psycopg2
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from flask import Flask, render_template, request, jsonify
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB  # Import Naive Bayes for classification

# Initialize the Sentence Transformer model for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Database Connection
conn = psycopg2.connect(
    host="localhost",
    port="5432",
    dbname="mydatabase",
    user="postgres",
    password="y@pradeep"
)
cursor = conn.cursor()

# Load pre-trained T5 model and tokenizer for summarization
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")


# Helper Functions
def normalize_embedding(embedding):
    """Normalize the embedding to unit length."""
    norm = np.linalg.norm(embedding)
    return (embedding / norm).tolist() if norm > 0 else embedding


def cosine_similarity(vec1, vec2):
    """Compute the cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)


def insert_question_answer(question, answer):
    """Insert question, answer, and question embedding into PostgreSQL."""
    try:
        question_embedding = normalize_embedding(embedder.encode(question))
        query = """
            INSERT INTO qa_table (question, question_embedding, answer)
            VALUES (%s, %s::vector, %s);
        """
        cursor.execute(query, (question, question_embedding, answer))
        # update_question_frequency(question)
        conn.commit()
    except Exception as e:
        print(f"Error inserting question-answer pair: {e}")


def load_questions_answers_from_files(question_file, answer_file):
    """Load questions and answers from text files and insert them into the database."""
    try:
        with open(question_file, 'r', encoding='utf-8') as qf, open(answer_file, 'r', encoding='utf-8') as af:
            questions = qf.readlines()
            answers = af.readlines()

            if len(questions) != len(answers):
                print("Error: Questions and Answers files do not have the same number of entries.")
                return

            for question, answer in zip(questions, answers):
                question = question.strip()
                answer = answer.strip()
                if question and answer:  # Only insert if both are non-empty
                    insert_question_answer(question, answer)
            print(f"Successfully inserted {len(questions)} question-answer pairs into the database.")
    except Exception as e:
        print(f"Error loading questions and answers from files: {e}")


def retrieve_answer(query, top_k=1, relevance_threshold=0.5):
    """Retrieve the most similar question-answer pair or classify as irrelevant."""
    try:
        # Normalize query embedding
        query_embedding = normalize_embedding(embedder.encode(query))

        # Fetch questions, embeddings, and answers from the database
        cursor.execute("SELECT question, question_embedding, answer FROM qa_table")
        results = cursor.fetchall()

        # Extract questions for fuzzy matching
        questions = [row[0] for row in results]
        
        # Perform fuzzy matching to find the closest question
        best_match = process.extractOne(query, questions, scorer=fuzz.ratio)
        
        if best_match and best_match[1] > 75:  # Threshold for fuzzy match (adjust as needed)
            # Use the best match question
            matched_question = best_match[0]
            for question, embedding, answer in results:
                if question == matched_question:
                    similarity = cosine_similarity(query_embedding, np.array(ast.literal_eval(embedding)))
                    
                    # **Update frequency for the matched question**
                    update_question_frequency(matched_question)

                    return [(similarity, question, answer)]  # Return matched result

        # If no good fuzzy match is found, fall back to semantic similarity
        similarities = []
        for question, embedding, answer in results:
            # Convert the string embedding back to a numerical array
            if isinstance(embedding, str):
                embedding = np.array(ast.literal_eval(embedding))  # Parse string to array
            elif isinstance(embedding, list):
                embedding = np.array(embedding)  # Handle if embedding is already in list form

            similarity = cosine_similarity(query_embedding, embedding)
            similarities.append((similarity, question, answer))

        # Sort by similarity
        similarities.sort(reverse=True, key=lambda x: x[0])

        # Check relevance threshold
        if not similarities or similarities[0][0] < relevance_threshold:
            return [{"question": None, "answer": "Please ask a question related to Machine Learning."}]

        # **Update frequency for the top result (most similar question)**
        update_question_frequency(similarities[0][1])

        # Return top-k results
        return similarities[:top_k]
    except Exception as e:
        print(f"Error retrieving answer: {e}")
        return [{"question": None, "answer": "Please ask a question related to Machine Learning."}]


def summarize_text_with_answer(answer, query):
    """Summarize the full answer with context."""
    try:
        input_text = f"query: {query.strip()} context: {answer.strip()}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(model.device)
        outputs = model.generate(
            inputs.input_ids, max_length=200, min_length=50, length_penalty=1.0, num_beams=5
        )
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary if summary.endswith('.') else summary + '.'
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return "Error in summarization."


def update_question_frequency(question):
    """Update the frequency of a question."""
    try:
        # Check if the question already exists in the question_frequency table
        cursor.execute("SELECT count FROM question_frequency WHERE question = %s", (question,))
        result = cursor.fetchone()

        if result:
            # Increment the count if the question exists
            cursor.execute("UPDATE question_frequency SET count = count + 1 WHERE question = %s", (question,))
        else:
            # Insert a new record if the question does not exist
            cursor.execute("INSERT INTO question_frequency (question, count) VALUES (%s, 1)", (question,))
        
        # Commit the transaction
        conn.commit()
    except Exception as e:
        print(f"Error updating question frequency: {e}")



def classify_feedback_types():
    """Classify feedback types using Naive Bayes."""
    try:
        # Join feedback_table and qa_table to fetch embeddings and feedback types
        cursor.execute("""
            SELECT q.question_embedding, f.feedback_type
            FROM qa_table q
            JOIN feedback_table f ON q.question = f.question;
        """)
        data = cursor.fetchall()

        if not data:
            print("No feedback data available for classification.")
            return

        # Extract embeddings and feedback types
        embeddings = [np.array(ast.literal_eval(row[0])) for row in data]  # Parse string embeddings
        feedback_types = [row[1] for row in data]

        # Encode feedback types to numeric labels
        encoder = LabelEncoder()
        labels = encoder.fit_transform(feedback_types)

        # Train a Naive Bayes classifier
        model = GaussianNB()
        model.fit(embeddings, labels)

        print("Naive Bayes model trained successfully for feedback classification.")
    except Exception as e:
        print(f"Error classifying feedback types: {e}")



def cluster_existing_data(num_clusters=10):
    """Cluster the existing data using K-means and update cluster labels."""
    try:
        # Fetch all embeddings
        cursor.execute("SELECT id, question_embedding FROM qa_table")
        data = cursor.fetchall()
        if not data:
            print("No data to cluster.")
            return
        
        ids = [row[0] for row in data]
        embeddings = np.array([ast.literal_eval(row[1]) for row in data])

        # Train K-means
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Update cluster labels in the database
        for idx, cluster_label in zip(ids, cluster_labels):
            cursor.execute("""
                UPDATE qa_table
                SET cluster_label = %s
                WHERE id = %s;
            """, (int(cluster_label), idx))  # Convert numpy.int32 to regular Python int
        conn.commit()
        print("Cluster labels updated successfully.")
    except Exception as e:
        print(f"Error clustering data: {e}")

def check_and_create_tables():
    """Check if the necessary tables exist, and create them if not."""
    try:
        cursor.execute("SELECT to_regclass('public.qa_table');")
        if cursor.fetchone()[0] is None:
            cursor.execute("""
                CREATE TABLE qa_table (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    question_embedding VECTOR NOT NULL,
                    answer TEXT NOT NULL,
                    cluster_label INTEGER
                );
            """)
        
        cursor.execute("SELECT to_regclass('public.question_frequency');")
        if cursor.fetchone()[0] is None:
            cursor.execute("""
                CREATE TABLE question_frequency (
                    id SERIAL PRIMARY KEY,
                    question TEXT UNIQUE NOT NULL,
                    count INTEGER NOT NULL DEFAULT 1
                );
            """)

        
        cursor.execute("SELECT to_regclass('public.feedback_table');")
        if cursor.fetchone()[0] is None:
            cursor.execute("""
                CREATE TABLE feedback_table (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,  -- Positive or Negative
                    feedback_comment TEXT,        -- Optional comment
                    helpfulness_rating INTEGER,   -- Rating between 1-5
                    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
                );
            """)
        conn.commit()
    except Exception as e:
        print(f"Error checking/creating tables: {e}")


def apply_linear_regression():
    """Apply Linear Regression on feedback data."""
    try:
        # Fetch relevant feedback data
        cursor.execute("SELECT q.question_embedding, f.helpfulness_rating, f.feedback_type FROM qa_table q JOIN feedback_table f ON q.question = f.question")
        data = cursor.fetchall()

        if not data:
            print("No feedback data to train model.")
            return

        # Extract features and target variable
        embeddings = [np.array(ast.literal_eval(row[0])) for row in data]  # Question embeddings
        helpfulness_ratings = [row[1] for row in data]  # Target variable (Helpfulness rating)
        feedback_types = [row[2] for row in data]  # Feedback type (e.g., Positive or Negative)

        # Use LabelEncoder to convert categorical feedback_type to numeric values
        encoder = LabelEncoder()
        feedback_type_encoded = encoder.fit_transform(feedback_types)

        # Combine question embeddings and feedback type as features
        features = np.column_stack((embeddings, feedback_type_encoded))

        # Standardize the embeddings and feedback features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Train a linear regression model
        model = LinearRegression()
        model.fit(features_scaled, helpfulness_ratings)

        print("Linear regression model trained successfully.")
    except Exception as e:
        print(f"Error applying Linear Regression: {e}")


# Flask App
app = Flask(__name__)

@app.route('/')
def index():
    """Render the main page with top questions."""
    cursor.execute("SELECT question FROM question_frequency ORDER BY count DESC LIMIT 8")
    top_questions = [row[0] for row in cursor.fetchall()]
    return render_template('index.html', top_questions=top_questions)


@app.route('/ask', methods=['POST'])
def ask():
    """Handle user queries."""
    user_input = request.form.get('user_input', '')

    # Retrieve answers or classify as irrelevant
    similar_docs = retrieve_answer(user_input, relevance_threshold=0.5)

    # Prepare summaries or default response
    summaries = []
    for doc in similar_docs:
        if isinstance(doc, dict):  # If the response is a dictionary (irrelevant question)
            summaries.append({
                'question': "Irrelevant Question",
                'summary': doc['answer']
            })
        else:  # If the response is a tuple (relevant question)
            summaries.append({
                'question': doc[1],
                'summary': summarize_text_with_answer(doc[2], user_input)
            })

    return jsonify(summaries)


@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    """Handle feedback submission from the user."""
    try:
        # Get the data from the form
        question = request.form['question']
        answer = request.form['answer']
        feedback_type = request.form['feedback_type']
        feedback_comment = request.form.get('feedback_comment', '')  # Optional comment
        helpfulness_rating = int(request.form['helpfulness_rating'])  # Rating from 1-5

        # Get the current timestamp for feedback submission
        timestamp = datetime.now()

        # Insert feedback into the feedback_table (including timestamp and rating)
        query = """
            INSERT INTO feedback_table (question, answer, feedback_type, feedback_comment, helpfulness_rating, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s);
        """
        cursor.execute(query, (question, answer, feedback_type, feedback_comment, helpfulness_rating, timestamp))
        conn.commit()

        return jsonify({"success": True, "message": "Feedback submitted successfully."})
    except Exception as e:
        print(f"Error submitting feedback: {e}")
        return jsonify({"success": False, "message": "Failed to submit feedback."})


if __name__ == "__main__":
    check_and_create_tables()  # Ensure tables exist
    question_file = "Questions.txt"
    answer_file = "Answer.txt"
    load_questions_answers_from_files(question_file, answer_file)
    cluster_existing_data()  # Apply clustering
    classify_feedback_types()  # Classify feedback using Naive Bayes
    apply_linear_regression()  # Train regression model
    app.run(debug=True)
