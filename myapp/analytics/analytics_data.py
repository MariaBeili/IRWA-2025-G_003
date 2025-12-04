import pandas as pd
import altair as alt
from datetime import datetime

class AnalyticsData:
    """
    An in-memory persistence object using a Star Schema-like structure.
    """

    # FACT TABLE: Stores Search Query Events
    # Schema: {timestamp, query, session_id, browser, os, ip_address}
    fact_queries = []

    # FACT TABLE: Stores Click Events
    # Schema: {timestamp, doc_id, related_query}
    fact_clicks = []

    def save_query_event(self, query: str, session_id: str, user_agent: dict, ip: str):
        """
        Logs a search query event with context.
        """
        event = {
            "timestamp": datetime.now(),
            "query": query,
            "session_id": session_id,
            "browser": user_agent.get('browser', {}).get('name', 'Unknown'),
            "os": user_agent.get('os', {}).get('name', 'Unknown'),
            "ip_address": ip
        }
        self.fact_queries.append(event)
        # Print for debugging
        print(f"Logged Query: {event}")

    def save_click_event(self, doc_id: str, query: str):
        """
        Logs a document click event.
        """
        event = {
            "timestamp": datetime.now(),
            "doc_id": doc_id,
            "related_query": query  # Links the click back to the search
        }
        self.fact_clicks.append(event)
        print(f"Logged Click: {event}")

    ### VISUALIZATIONS FOR DASHBOARD ###

    def plot_browser_distribution(self):
        """
        Donut chart showing which browsers users are using.
        """
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
        """
        Bar chart of the most frequent search terms.
        """
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
        """
        Line chart showing clicks per hour/minute.
        """
        if not self.fact_clicks:
            return None

        df = pd.DataFrame(self.fact_clicks)
        
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X('timestamp', timeUnit='hoursminutes', title='Time'),
            y=alt.Y('count()', title='Number of Clicks'),
            tooltip=['timestamp', 'count()']
        ).properties(title="Clicks Over Time")

        return chart.to_json()