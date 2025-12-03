import os
from json import JSONEncoder

import time

import httpagentparser  # for getting the user agent as json
from flask import Flask, render_template, session
from flask import request

from project_progress.part_1.data_preparation import ProcessedDocument
from project_progress.part_2.indexing import create_index_tfidf, search_tfidf

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine
from myapp.generation.rag import RAGGenerator
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env


# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default
# end lines ***for using method to_json in objects ***


# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = os.getenv("SECRET_KEY")
# open browser dev tool to see the cookies
app.session_cookie_name = os.getenv("SESSION_COOKIE_NAME")
# instantiate our search engine
search_engine = SearchEngine()
# instantiate our in memory persistence
analytics_data = AnalyticsData()
# instantiate RAG generator
rag_generator = RAGGenerator()

# load documents corpus into memory.
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
file_path = path + "/" + os.getenv("DATA_FILE_PATH")
corpus = load_corpus(file_path)
# Log first element of corpus to verify it loaded correctly:
print("\nCorpus is loaded... \n First element:\n", list(corpus.values())[0])

# Log first element of corpus onvert it into a ProcessedDocument
processed_doc = ProcessedDocument.from_document(list(corpus.values())[0])
processed_doc.process_fields()
print("\nFirst processed element:\n", processed_doc)

# Example documents
print("Processing documents...")
start_time = time.time()

processed_corpus = [None] * len(corpus)

for i in range(len(corpus)):
    processed_corpus[i] = ProcessedDocument.from_document(list(corpus.values())[i])
    processed_corpus[i].process_fields()

process_time = time.time() - start_time
print(f"Documents processed in {process_time:.4f} seconds.\n")


# Time the index creation
print("Building inverted index and computing TF-IDF...")
start_time = time.time()

index, tf, df, idf, title_index = create_index_tfidf(processed_corpus)

index_time = time.time() - start_time
print(f"Index built in {index_time:.4f} seconds.\n")


# Define queries
query1 = "ARBO cotton track pants for men"
query2 = "Multicolor track pants combo ECKO"
query3 = "Black solid women track pants"
query4 = "Elastic waist cotton blend track pants"
query5 = "Self design multicolor track pants"

queries = [query1, query2, query3, query4, query5]


# Run and time searches
for i, query in enumerate(queries, start=1):
    print(f"\nQuery {i}: {query}")
    start_time = time.time()
    ranked_docs = search_tfidf(query, index, tf, idf)
    search_time = time.time() - start_time
    print(f"Search completed in {search_time:.4f} seconds.")

    if not ranked_docs:
        print("No matching documents found.")
    else:
        print("Ranked results:")
        for rank, doc_id in enumerate(ranked_docs, start=1):
            title = title_index.get(doc_id, "Unknown Title")
            print(f"{rank}. {title} (doc_pid={doc_id})")



# Home URL "/"
@app.route('/')
def home():
    print("starting home url /...")

    # flask server creates a session by persisting a cookie in the user's browser.
    # the 'session' object keeps data between multiple requests. Example:
    session['some_var'] = "Some value that is kept in session"

    user_agent = request.headers.get('User-Agent')
    print("Raw user browser:", user_agent)

    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    print("Remote IP: {} - JSON user browser {}".format(user_ip, agent))
    print(session)
    return render_template('index.html', page_title="Welcome")


# In web_app.py

@app.route('/search', methods=['POST'])
def search_form_post():
    search_query = request.form['search-query']
    analytics_data.save_query_terms(search_query)

    # 1. Run Search
    start_time = time.time()
    # Call your search logic (Part 2/3)
    ranked_pids = search_tfidf(search_query, index, tf, idf) 
    end_time = time.time()

    # 2. Get full objects
    results = []
    for pid in ranked_pids:
        if pid in corpus:
            results.append(corpus[pid])

    # 3. Generate RAG (Requirement 6)
    rag_response = rag_generator.generate_response(search_query, results, top_N=5)

    # 4. Render Template
    return render_template(
        'results.html',
        results_list=results,
        page_title="Search Results",
        found_counter=len(results),
        rag_response=rag_response,
        search_time=f"{end_time - start_time:.4f}"
    )


@app.route('/doc_details', methods=['GET'])
def doc_details():
    """
    Requirement 7: Document Details Page
    """
    clicked_doc_id = request.args.get("pid")
    
    # 1. Track the click (Analytics)
    if clicked_doc_id:
        if clicked_doc_id in analytics_data.fact_clicks:
            analytics_data.fact_clicks[clicked_doc_id] += 1
        else:
            analytics_data.fact_clicks[clicked_doc_id] = 1

    # 2. Retrieve the specific document
    # 'corpus' is a dictionary where Key=PID, Value=Document OBJECT
    doc = corpus.get(clicked_doc_id)
    
    if not doc:
        return render_template('doc_details.html', doc=None, page_title="Not Found")

    # --- THE FIX IS HERE ---
    # We use 'doc.title' (dot notation), NOT 'doc.get("title")'
    print(f"Found document: {doc.title}") 

    return render_template('doc_details.html', doc=doc, page_title=doc.title)

@app.route('/stats', methods=['GET'])
def stats():
    """
    Show simple statistics example. ### Replace with yourdashboard ###
    :return:
    """

    docs = []
    for doc_id in analytics_data.fact_clicks:
        row: Document = corpus[doc_id]
        count = analytics_data.fact_clicks[doc_id]
        doc = StatsDocument(pid=row.pid, title=row.title, description=row.description, url=row.url, count=count)
        docs.append(doc)
    
    # simulate sort by ranking
    docs.sort(key=lambda doc: doc.count, reverse=True)
    return render_template('stats.html', clicks_data=docs)


@app.route('/dashboard', methods=['GET'])
def dashboard():
    visited_docs = []
    for doc_id in analytics_data.fact_clicks.keys():
        d: Document = corpus[doc_id]
        doc = ClickedDoc(doc_id, d.description, analytics_data.fact_clicks[doc_id])
        visited_docs.append(doc)

    # simulate sort by ranking
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)

    for doc in visited_docs: print(doc)
    return render_template('dashboard.html', visited_docs=visited_docs)


# New route added for generating an examples of basic Altair plot (used for dashboard)
@app.route('/plot_number_of_views', methods=['GET'])
def plot_number_of_views():
    return analytics_data.plot_number_of_views()


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG"), use_reloader=False)