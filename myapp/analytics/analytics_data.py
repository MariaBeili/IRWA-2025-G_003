import pandas as pd
import altair as alt
from datetime import datetime

class AnalyticsData:
    """
    An in-memory persistence object using a Star Schema-like structure.
    """

    def __init__(self):
        # FACT TABLE: Stores Search Query Events
        self.fact_queries = []

        # FACT TABLE: Stores Click Events
        self.fact_clicks = []

        # FACT TABLE: Stores Dwell Time (Time Spent on Page)
        self.fact_dwell_times = []

    def save_query_event(self, query: str, session_id: str, user_agent: dict, ip: str, ranking_method: str):
        event = {
            "timestamp": datetime.now(),
            "query": query,
            "session_id": session_id,
            "browser": user_agent.get('browser', {}).get('name', 'Unknown'),
            "os": user_agent.get('os', {}).get('name', 'Unknown'),
            "ip_address": ip,
            "ranking_method": ranking_method
        }
        self.fact_queries.append(event)
        print(f"Logged Query: {event}")

    def save_click_event(self, doc_id: str, query: str):
        event = {
            "timestamp": datetime.now(),
            "doc_id": doc_id,
            "related_query": query
        }
        self.fact_clicks.append(event)
        print(f"Logged Click: {event}")

    def save_dwell_time_event(self, doc_id: str, time_spent: float, session_id: str):
        event = {
            "timestamp": datetime.now(),
            "doc_id": doc_id,
            "time_spent": time_spent,
            "session_id": session_id
        }
        self.fact_dwell_times.append(event)
        print(f"Logged Dwell Time: {event}")

    ### VISUALIZATIONS FOR DASHBOARD ###

    def plot_browser_distribution(self):
        if not self.fact_queries:
            return None
        
        df = pd.DataFrame(self.fact_queries)
        
        chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
            theta=alt.Theta("count()", stack=True),
            color=alt.Color("browser", legend=alt.Legend(title="Browser")),
            tooltip=["browser", "count()"]
        ).properties(title="User Browser Distribution")
        
        return chart.to_json()

    def plot_top_queries(self):
        if not self.fact_queries:
            return None

        df = pd.DataFrame(self.fact_queries)
        
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('count()', title='Frequency'),
            y=alt.Y('query', sort='-x', title='Search Terms'),
            tooltip=['query', 'count()']
        ).properties(title="Top Search Queries")
        
        return chart.to_json()

    def plot_clicks_over_time(self):
        if not self.fact_clicks:
            return None

        df = pd.DataFrame(self.fact_clicks)
        
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X('timestamp', timeUnit='hoursminutes', title='Time'),
            y=alt.Y('count()', title='Number of Clicks'),
            tooltip=['timestamp', 'count()']
        ).properties(title="Clicks Over Time")

        return chart.to_json()

    def plot_ranking_method_usage(self):
        if not self.fact_queries:
            return None

        df = pd.DataFrame(self.fact_queries)

        if 'ranking_method' not in df.columns:
            return None

        chart = alt.Chart(df).mark_bar(color='teal').encode(
            x=alt.X('ranking_method', title='Ranking Method'),
            y=alt.Y('count()', title='Usage Count'),
            tooltip=['ranking_method', 'count()']
        ).properties(title="Ranking Method Usage")

        return chart.to_json()

    def plot_top_clicked_items(self):
        if not self.fact_clicks:
            return None

        df = pd.DataFrame(self.fact_clicks)

        chart = alt.Chart(df).mark_bar(color='orange').encode(
            x=alt.X('count()', title='Total Clicks'),
            y=alt.Y('doc_id', sort='-x', title='Product ID (PID)'),
            tooltip=['doc_id', 'count()']
        ).properties(title="Top Clicked Products")

        return chart.to_json()

    def plot_dwell_time_distribution(self):
        if not self.fact_dwell_times:
            return None

        df = pd.DataFrame(self.fact_dwell_times)

        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('time_spent', bin=alt.Bin(maxbins=20), title='Time Spent (Seconds)'),
            y=alt.Y('count()', title='Number of Views'),
            tooltip=['time_spent', 'count()']
        ).properties(title="Dwell Time Distribution")

        return chart.to_json()