import csv
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import numpy as np
import logging  # Added import for logging
from flask import Flask, render_template, request, jsonify
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define your existing preprocess_text and keyword_matching functions here

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return tokens

def keyword_matching(user_query):
    # Preprocess user query
    user_tokens = preprocess_text(user_query)

    # Replace this with your actual CSV file path
    csv_file_path = 'Dataset.csv'

    # Open the CSV file
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        # Read the header line to get column names
        column_names = next(csv_reader)

        # Find the index of the TAGS column
        tags_column_index = column_names.index("TAGS")

        # Process each row in the CSV file
        matching_rows = []
        for row in csv_reader:
            # Tokenize and preprocess the TAGS column in the CSV
            csv_tokens = preprocess_text(row[tags_column_index])

            # Check for keyword match
            if any(token in csv_tokens for token in user_tokens):
                matching_rows.append(dict(zip(column_names, row)))

    return matching_rows

# Add a dictionary to store feedback
feedback_data = {}

# Add a dictionary for Q-values
q_values = {}

# Initialize Q-values for each project
def initialize_q_values(projects):
    for project_id in projects:
        q_values[project_id] = 0.0

# Exploration rate (epsilon) for epsilon-greedy policy
epsilon = 0.2

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/search', methods=['POST'])
def search():
    user_query = request.form.get('query')
    matching_rows = keyword_matching(user_query)
    initialize_q_values([result['PROJECT ID'] for result in matching_rows])
    return render_template('search_results.html', matching_rows=matching_rows)

@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_value = request.form.get('feedback')
    project_id = request.form.get('project_id')

    # Store the feedback in the dictionary
    if project_id in feedback_data:
        feedback_data[project_id].append(feedback_value)
    else:
        feedback_data[project_id] = [feedback_value]

    # Update Q-values using a basic Q-learning approach
    update_q_values(project_id, feedback_value)

    return jsonify({'success': True})

def update_q_values(project_id, feedback_value):
    # Simple Q-learning update rule
    reward = 1.0 if feedback_value == 'thumbs_up' else -1.0
    learning_rate = 0.1
    discount_factor = 0.9

    # Update Q-value
    current_q_value = q_values[project_id]
    max_q_value = max(q_values.values())
    updated_q_value = (1 - learning_rate) * current_q_value + learning_rate * (reward + discount_factor * max_q_value)

    q_values[project_id] = updated_q_value

    # Log Q-value updates
    logger.debug(f"Updated Q-value for Project ID {project_id}: {current_q_value} -> {updated_q_value}")

if __name__ == '__main__':
    app.run(debug=True)
