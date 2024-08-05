from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

def load_documents():
    documents = []
    for filename in os.listdir('documents'):
        if filename.endswith('.txt'):
            with open(os.path.join('documents', filename), 'r') as file:
                documents.append(file.read())
    return documents

def check_plagiarism(input_text, documents):
    documents.append(input_text)
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    input_vector = cosine_matrix[-1][:-1]
    return input_vector

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['input_text']
        documents = load_documents()
        similarities = check_plagiarism(input_text, documents)
        return render_template('index.html', similarities=similarities)
    return render_template('index.html', similarities=None)

if __name__ == '__main__':
    app.run(debug=True)
