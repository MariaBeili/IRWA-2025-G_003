import os
from json import JSONEncoder
import time
import httpagentparser  # for getting the user agent as json
from flask import Flask, render_template, session, request
from project_progress.part_1.data_preparation import ProcessedDocument
from myapp.analytics.analytics_data import AnalyticsData
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine
from myapp.generation.rag import RAGGenerator
from dotenv import load_dotenv

load_dotenv() 

# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default

# instantiate the Flask application
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
app.session_cookie_name = os.getenv("SESSION_COOKIE_NAME")

# instantiate engines and data
search_engine = SearchEngine()
analytics_data = AnalyticsData()
rag_generator = RAGGenerator()

# load documents corpus
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
file_path = path + "/" + os.getenv("DATA_FILE_PATH")
corpus = load_corpus(file_path)

# Initialize Search Engine
# search_engine = SearchEngine(corpus=corpus) # Uncomment if building index from scratch

@app.route('/')
def home():
    # Generate a simple session ID for analytics if not present
    if 'uid' not in session:
        import uuid
        session['uid'] = str(uuid.uuid4())
    
    user_agent = request.headers.get('User-Agent')
    agent = httpagentparser.detect(user_agent)
    print(f"Session: {session.get('uid')} - Browser: {agent}")
    return render_template('index.html', page_title="Welcome")


@app.route('/search', methods=['POST'])
def search_form_post():
    search_query = request.form['search-query']
    selected_method = request.form.get('ranking-method')
    
    # --- ANALYTICS TRACKING START ---
    user_agent_str = request.headers.get('User-Agent')
    agent_data = httpagentparser.detect(user_agent_str)
    user_ip = request.remote_addr
    
    # Save the query with full context (Browser, OS, IP, Session)
    analytics_data.save_query_event(
        query=search_query,
        session_id=session.get('uid', 'anonymous'), 
        user_agent=agent_data,
        ip=user_ip
    )
    # --- ANALYTICS TRACKING END ---

    # 1. Run Search
    start_time = time.time()
    ranked_pids = search_engine.search(query=search_query, corpus=corpus, method=selected_method, topN=20) 
    end_time = time.time()

    # 2. Get full objects
    results = []
    for pid in ranked_pids:
        if pid in corpus:
            results.append(corpus[pid])

    # 3. Generate RAG
    rag_response = rag_generator.generate_response(search_query, results, top_N=5)

    return render_template(
        'results.html',
        results_list=results,
        page_title="Search Results",
        found_counter=len(results),
        rag_response=rag_response,
        search_time=f"{end_time - start_time:.4f}",
        # Pass the query so we can track it if the user clicks a doc
        last_query=search_query 
    )


@app.route('/doc_details', methods=['GET'])
def doc_details():
    clicked_doc_id = request.args.get("pid")
    related_query = request.args.get("query", "direct_link") 
    
    # 1. Track the click (Analytics)
    if clicked_doc_id:
        analytics_data.save_click_event(clicked_doc_id, related_query)

    # 2. Retrieve the specific document
    doc = corpus.get(clicked_doc_id)
    
    if not doc:
        return render_template('doc_details.html', doc=None, page_title="Not Found")

    return render_template('doc_details.html', doc=doc, page_title=doc.title)


@app.route('/dashboard', methods=['GET'])
def dashboard():
    """
    Analytics Dashboard displaying Vega-Lite charts
    """
    browser_chart = analytics_data.plot_browser_distribution()
    query_chart = analytics_data.plot_top_queries()
    time_chart = analytics_data.plot_clicks_over_time()

    return render_template(
        'dashboard.html',
        page_title="Analytics Dashboard",
        browser_chart=browser_chart,
        query_chart=query_chart,
        time_chart=time_chart
    )

if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG"), use_reloader=False)