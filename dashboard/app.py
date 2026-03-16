"""
Pre-Delinquency Intervention Engine - Dashboard
Enterprise-grade Plotly Dash dashboard with 5 views:
1. Portfolio Risk Heatmap
2. Trending Customers Panel
3. Intervention Tracker
4. Customer Deep-Dive View
5. Model Health Monitor
"""
import os
import sys
import json
import logging
from datetime import datetime, timedelta

import dash
from dash import html, dcc, dash_table, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import PostgresConfig, DashboardConfig

logger = logging.getLogger(__name__)

# Database connection
engine = create_engine(PostgresConfig.get_url())

# ─────────────────────────────────────────────
# Dash App
# ─────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="PDI Engine Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# ─────────────────────────────────────────────
# Styling Constants
# ─────────────────────────────────────────────
COLORS = {
    "bg": "#0a0a1a",
    "card_bg": "#111127",
    "card_border": "#1e1e3f",
    "text": "#e0e0ff",
    "text_muted": "#8888aa",
    "critical": "#ff4444",
    "watch": "#ffaa00",
    "stable": "#00cc88",
    "accent": "#6366f1",
    "accent_light": "#818cf8",
}

CARD_STYLE = {
    "backgroundColor": COLORS["card_bg"],
    "border": f"1px solid {COLORS['card_border']}",
    "borderRadius": "12px",
    "padding": "20px",
    "marginBottom": "15px",
}

TIER_COLORS = {
    "critical": COLORS["critical"],
    "watch": COLORS["watch"],
    "stable": COLORS["stable"],
}


# ─────────────────────────────────────────────
# Data Loading Functions
# ─────────────────────────────────────────────
def load_risk_distribution():
    """Load risk tier distribution."""
    try:
        df = pd.read_sql("""
            SELECT DISTINCT ON (customer_id) customer_id, risk_tier, risk_score, scored_at
            FROM risk_scores ORDER BY customer_id, scored_at DESC
        """, engine)
        return df
    except Exception:
        return pd.DataFrame(columns=["customer_id", "risk_tier", "risk_score", "scored_at"])


def load_customers_with_risk():
    """Load customers with their latest risk scores."""
    try:
        df = pd.read_sql("""
            SELECT c.customer_id, c.first_name, c.last_name, c.city, c.region,
                   c.income_bracket, c.credit_score as profile_credit_score,
                   r.risk_score, r.risk_tier, r.credit_score_mapped,
                   r.top_shap_features, r.scored_at
            FROM customers c
            LEFT JOIN LATERAL (
                SELECT * FROM risk_scores rs
                WHERE rs.customer_id = c.customer_id
                ORDER BY rs.scored_at DESC LIMIT 1
            ) r ON true
        """, engine)
        return df
    except Exception:
        return pd.DataFrame()


def load_trending_customers():
    """Load customers with fastest-deteriorating risk scores."""
    try:
        df = pd.read_sql("""
            WITH ranked AS (
                SELECT customer_id, risk_score, scored_at,
                       LAG(risk_score) OVER (PARTITION BY customer_id ORDER BY scored_at) as prev_score
                FROM risk_scores
                WHERE scored_at > NOW() - INTERVAL '14 days'
            )
            SELECT r.customer_id, c.first_name, c.last_name,
                   r.risk_score as current_score,
                   r.prev_score,
                   (r.risk_score - COALESCE(r.prev_score, r.risk_score)) as score_change,
                   r.scored_at
            FROM ranked r
            JOIN customers c ON c.customer_id = r.customer_id
            WHERE r.prev_score IS NOT NULL
            ORDER BY score_change DESC
            LIMIT 20
        """, engine)
        return df
    except Exception:
        return pd.DataFrame()


def load_interventions():
    """Load intervention history."""
    try:
        df = pd.read_sql("""
            SELECT i.*, c.first_name, c.last_name
            FROM interventions i
            JOIN customers c ON c.customer_id = i.customer_id
            ORDER BY i.sent_at DESC
            LIMIT 200
        """, engine)
        return df
    except Exception:
        return pd.DataFrame()


def load_customer_detail(customer_id: str):
    """Load detailed customer data for deep-dive."""
    try:
        customer = pd.read_sql(
            "SELECT * FROM customers WHERE customer_id = %s",
            engine, params=(customer_id,)
        )
        scores = pd.read_sql(
            "SELECT * FROM risk_scores WHERE customer_id = %s ORDER BY scored_at",
            engine, params=(customer_id,)
        )
        transactions = pd.read_sql(
            """SELECT * FROM transactions WHERE customer_id = %s
               ORDER BY timestamp DESC LIMIT 100""",
            engine, params=(customer_id,)
        )
        balances = pd.read_sql(
            """SELECT * FROM account_balances WHERE customer_id = %s
               ORDER BY timestamp""",
            engine, params=(customer_id,)
        )
        interventions = pd.read_sql(
            """SELECT * FROM interventions WHERE customer_id = %s
               ORDER BY sent_at DESC""",
            engine, params=(customer_id,)
        )
        return customer, scores, transactions, balances, interventions
    except Exception as e:
        logger.error(f"Error loading customer detail: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def load_model_health():
    """Load model performance metrics."""
    try:
        scores = pd.read_sql("""
            SELECT DATE(scored_at) as date,
                   AVG(risk_score) as avg_risk,
                   STDDEV(risk_score) as std_risk,
                   COUNT(*) as num_scores,
                   SUM(CASE WHEN risk_tier = 'critical' THEN 1 ELSE 0 END) as critical_count,
                   SUM(CASE WHEN risk_tier = 'watch' THEN 1 ELSE 0 END) as watch_count,
                   SUM(CASE WHEN risk_tier = 'stable' THEN 1 ELSE 0 END) as stable_count
            FROM risk_scores
            GROUP BY DATE(scored_at)
            ORDER BY date
        """, engine)
        drift = pd.read_sql("SELECT * FROM drift_logs ORDER BY detection_timestamp DESC LIMIT 20", engine)
        return scores, drift
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


# ─────────────────────────────────────────────
# Layout Components
# ─────────────────────────────────────────────
def make_stat_card(title, value, color=None, icon=None):
    """Create a stat card component."""
    return dbc.Card(
        dbc.CardBody([
            html.P(title, style={"color": COLORS["text_muted"], "fontSize": "13px", "marginBottom": "5px"}),
            html.H3(value, style={"color": color or COLORS["text"], "fontWeight": "bold", "margin": 0}),
        ]),
        style={**CARD_STYLE, "textAlign": "center"},
    )


# ─────────────────────────────────────────────
# Tab Layouts
# ─────────────────────────────────────────────
def portfolio_heatmap_layout():
    return html.Div([
        html.H4("📊 Portfolio Risk Heatmap", style={"color": COLORS["text"], "marginBottom": "20px"}),
        dbc.Row(id="portfolio-stats-row"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="risk-distribution-pie"), md=4),
            dbc.Col(dcc.Graph(id="risk-by-region"), md=4),
            dbc.Col(dcc.Graph(id="risk-by-income"), md=4),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="risk-heatmap"), md=12),
        ]),
    ])


def trending_customers_layout():
    return html.Div([
        html.H4("📈 Trending Customers", style={"color": COLORS["text"], "marginBottom": "20px"}),
        html.P("Customers with the fastest-deteriorating risk scores",
               style={"color": COLORS["text_muted"]}),
        html.Div(id="trending-table-container"),
    ])


def intervention_tracker_layout():
    return html.Div([
        html.H4("🎯 Intervention Tracker", style={"color": COLORS["text"], "marginBottom": "20px"}),
        dbc.Row([
            dbc.Col(dcc.Graph(id="intervention-by-type"), md=6),
            dbc.Col(dcc.Graph(id="intervention-by-channel"), md=6),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="intervention-outcomes"), md=6),
            dbc.Col(dcc.Graph(id="intervention-timeline"), md=6),
        ]),
        html.Div(id="intervention-table-container"),
    ])


def customer_deepdive_layout():
    return html.Div([
        html.H4("🔍 Customer Deep-Dive", style={"color": COLORS["text"], "marginBottom": "20px"}),
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id="customer-select",
                    placeholder="Select or search customer...",
                    style={"backgroundColor": COLORS["card_bg"], "color": "#000"},
                ),
            ], md=6),
        ]),
        html.Div(id="customer-detail-container", style={"marginTop": "20px"}),
    ])


def model_health_layout():
    return html.Div([
        html.H4("🏥 Model Health Monitor", style={"color": COLORS["text"], "marginBottom": "20px"}),
        dbc.Row(id="model-stats-row"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="risk-score-trend"), md=6),
            dbc.Col(dcc.Graph(id="tier-distribution-trend"), md=6),
        ]),
        html.H5("Drift Detection History", style={"color": COLORS["text"], "marginTop": "20px"}),
        html.Div(id="drift-table-container"),
    ])


# ─────────────────────────────────────────────
# Main Layout
# ─────────────────────────────────────────────
app.layout = html.Div([
    # Header
    html.Div([
        html.H2("⚡ Pre-Delinquency Intervention Engine",
                style={"color": COLORS["text"], "fontWeight": "bold", "marginBottom": "0"}),
        html.P("Real-time credit risk monitoring & proactive intervention",
               style={"color": COLORS["text_muted"], "fontSize": "14px"}),
    ], style={"padding": "20px 30px", "borderBottom": f"1px solid {COLORS['card_border']}"}),

    # Tabs
    dcc.Tabs(
        id="main-tabs",
        value="portfolio",
        children=[
            dcc.Tab(label="Portfolio Heatmap", value="portfolio",
                    style={"backgroundColor": COLORS["card_bg"], "color": COLORS["text_muted"]},
                    selected_style={"backgroundColor": COLORS["accent"], "color": "#fff"}),
            dcc.Tab(label="Trending Customers", value="trending",
                    style={"backgroundColor": COLORS["card_bg"], "color": COLORS["text_muted"]},
                    selected_style={"backgroundColor": COLORS["accent"], "color": "#fff"}),
            dcc.Tab(label="Intervention Tracker", value="interventions",
                    style={"backgroundColor": COLORS["card_bg"], "color": COLORS["text_muted"]},
                    selected_style={"backgroundColor": COLORS["accent"], "color": "#fff"}),
            dcc.Tab(label="Customer Deep-Dive", value="deepdive",
                    style={"backgroundColor": COLORS["card_bg"], "color": COLORS["text_muted"]},
                    selected_style={"backgroundColor": COLORS["accent"], "color": "#fff"}),
            dcc.Tab(label="Model Health", value="model_health",
                    style={"backgroundColor": COLORS["card_bg"], "color": COLORS["text_muted"]},
                    selected_style={"backgroundColor": COLORS["accent"], "color": "#fff"}),
        ],
        style={"padding": "10px 30px"},
    ),

    # Tab content
    html.Div(id="tab-content", style={"padding": "20px 30px"}),

    # Auto-refresh every 30 seconds
    dcc.Interval(id="refresh-interval", interval=30 * 1000, n_intervals=0),

], style={"backgroundColor": COLORS["bg"], "minHeight": "100vh", "fontFamily": "'Inter', sans-serif"})


# ─────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────
@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
)
def render_tab(tab):
    if tab == "portfolio":
        return portfolio_heatmap_layout()
    elif tab == "trending":
        return trending_customers_layout()
    elif tab == "interventions":
        return intervention_tracker_layout()
    elif tab == "deepdive":
        return customer_deepdive_layout()
    elif tab == "model_health":
        return model_health_layout()
    return html.Div("Select a tab")


# --- Portfolio Heatmap Callbacks ---
@app.callback(
    [Output("portfolio-stats-row", "children"),
     Output("risk-distribution-pie", "figure"),
     Output("risk-by-region", "figure"),
     Output("risk-by-income", "figure"),
     Output("risk-heatmap", "figure")],
    [Input("refresh-interval", "n_intervals"),
     Input("main-tabs", "value")],
)
def update_portfolio(n, tab):
    if tab != "portfolio":
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    df = load_customers_with_risk()
    if df.empty:
        empty = go.Figure().update_layout(template="plotly_dark",
                                          paper_bgcolor=COLORS["bg"],
                                          plot_bgcolor=COLORS["bg"])
        return [], empty, empty, empty, empty

    # Stats
    total = len(df)
    critical = len(df[df["risk_tier"] == "critical"])
    watch = len(df[df["risk_tier"] == "watch"])
    stable = len(df[df["risk_tier"] == "stable"])
    no_score = total - critical - watch - stable

    stats = dbc.Row([
        dbc.Col(make_stat_card("Total Customers", f"{total:,}"), md=3),
        dbc.Col(make_stat_card("Critical", f"{critical:,}", COLORS["critical"]), md=3),
        dbc.Col(make_stat_card("Watch", f"{watch:,}", COLORS["watch"]), md=3),
        dbc.Col(make_stat_card("Stable", f"{stable + no_score:,}", COLORS["stable"]), md=3),
    ])

    # Pie chart
    scored = df[df["risk_tier"].notna()]
    if not scored.empty:
        tier_counts = scored["risk_tier"].value_counts()
        pie = px.pie(
            values=tier_counts.values, names=tier_counts.index,
            color=tier_counts.index,
            color_discrete_map=TIER_COLORS,
            title="Risk Tier Distribution",
        )
    else:
        pie = go.Figure()

    pie.update_layout(template="plotly_dark", paper_bgcolor=COLORS["bg"],
                      plot_bgcolor=COLORS["bg"], font_color=COLORS["text"])

    # By region
    if "region" in df.columns and scored.any().any():
        region_df = scored.groupby(["region", "risk_tier"]).size().reset_index(name="count")
        by_region = px.bar(
            region_df, x="region", y="count", color="risk_tier",
            color_discrete_map=TIER_COLORS, title="Risk by Region",
            barmode="stack",
        )
    else:
        by_region = go.Figure()
    by_region.update_layout(template="plotly_dark", paper_bgcolor=COLORS["bg"],
                            plot_bgcolor=COLORS["bg"], font_color=COLORS["text"])

    # By income
    if "income_bracket" in df.columns and scored.any().any():
        income_df = scored.groupby(["income_bracket", "risk_tier"]).size().reset_index(name="count")
        by_income = px.bar(
            income_df, x="income_bracket", y="count", color="risk_tier",
            color_discrete_map=TIER_COLORS, title="Risk by Income Bracket",
            barmode="stack",
        )
    else:
        by_income = go.Figure()
    by_income.update_layout(template="plotly_dark", paper_bgcolor=COLORS["bg"],
                            plot_bgcolor=COLORS["bg"], font_color=COLORS["text"])

    # Heatmap: region vs income
    if "region" in df.columns and "income_bracket" in df.columns and scored.any().any():
        pivot = scored.pivot_table(
            values="risk_score", index="region", columns="income_bracket",
            aggfunc="mean", fill_value=0,
        )
        heatmap = px.imshow(
            pivot, color_continuous_scale="RdYlGn_r",
            title="Average Risk Score: Region × Income",
            labels={"color": "Avg Risk Score"},
        )
    else:
        heatmap = go.Figure()
    heatmap.update_layout(template="plotly_dark", paper_bgcolor=COLORS["bg"],
                          plot_bgcolor=COLORS["bg"], font_color=COLORS["text"])

    return stats.children, pie, by_region, by_income, heatmap


# --- Trending Customers Callback ---
@app.callback(
    Output("trending-table-container", "children"),
    [Input("refresh-interval", "n_intervals"),
     Input("main-tabs", "value")],
)
def update_trending(n, tab):
    if tab != "trending":
        return dash.no_update

    df = load_trending_customers()
    if df.empty:
        return html.P("No trending data available yet. Run scoring first.",
                      style={"color": COLORS["text_muted"]})

    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[
            {"name": "Customer", "id": "first_name"},
            {"name": "Current Score", "id": "current_score", "type": "numeric",
             "format": dash_table.Format.Format(precision=3)},
            {"name": "Previous Score", "id": "prev_score", "type": "numeric",
             "format": dash_table.Format.Format(precision=3)},
            {"name": "Change", "id": "score_change", "type": "numeric",
             "format": dash_table.Format.Format(precision=3)},
        ],
        style_header={
            "backgroundColor": COLORS["card_bg"],
            "color": COLORS["text"],
            "fontWeight": "bold",
            "borderBottom": f"2px solid {COLORS['accent']}",
        },
        style_data={
            "backgroundColor": COLORS["bg"],
            "color": COLORS["text"],
            "borderBottom": f"1px solid {COLORS['card_border']}",
        },
        style_data_conditional=[
            {"if": {"filter_query": "{score_change} > 0.1"},
             "color": COLORS["critical"], "fontWeight": "bold"},
            {"if": {"filter_query": "{score_change} > 0 AND {score_change} <= 0.1"},
             "color": COLORS["watch"]},
        ],
        page_size=20,
        sort_action="native",
    )


# --- Intervention Tracker Callbacks ---
@app.callback(
    [Output("intervention-by-type", "figure"),
     Output("intervention-by-channel", "figure"),
     Output("intervention-outcomes", "figure"),
     Output("intervention-timeline", "figure"),
     Output("intervention-table-container", "children")],
    [Input("refresh-interval", "n_intervals"),
     Input("main-tabs", "value")],
)
def update_interventions(n, tab):
    if tab != "interventions":
        return [dash.no_update] * 5

    df = load_interventions()
    empty = go.Figure().update_layout(template="plotly_dark",
                                      paper_bgcolor=COLORS["bg"],
                                      plot_bgcolor=COLORS["bg"])

    if df.empty:
        msg = html.P("No interventions recorded yet.", style={"color": COLORS["text_muted"]})
        return empty, empty, empty, empty, msg

    # By type
    type_counts = df["intervention_type"].value_counts()
    by_type = px.bar(x=type_counts.index, y=type_counts.values,
                     title="Interventions by Type",
                     color=type_counts.index,
                     color_discrete_sequence=px.colors.qualitative.Set2)
    by_type.update_layout(template="plotly_dark", paper_bgcolor=COLORS["bg"],
                          plot_bgcolor=COLORS["bg"], font_color=COLORS["text"],
                          showlegend=False)

    # By channel
    channel_counts = df["channel"].value_counts()
    by_channel = px.pie(values=channel_counts.values, names=channel_counts.index,
                        title="Outreach by Channel",
                        color_discrete_sequence=px.colors.qualitative.Pastel)
    by_channel.update_layout(template="plotly_dark", paper_bgcolor=COLORS["bg"],
                             plot_bgcolor=COLORS["bg"], font_color=COLORS["text"])

    # Outcomes
    if "outcome" in df.columns and df["outcome"].notna().any():
        outcome_counts = df["outcome"].value_counts()
        outcomes = px.bar(x=outcome_counts.index, y=outcome_counts.values,
                          title="Intervention Outcomes",
                          color=outcome_counts.index,
                          color_discrete_map={
                              "paid": COLORS["stable"],
                              "restructured": COLORS["watch"],
                              "defaulted": COLORS["critical"],
                              "no_response": COLORS["text_muted"],
                          })
    else:
        outcomes = go.Figure()
    outcomes.update_layout(template="plotly_dark", paper_bgcolor=COLORS["bg"],
                           plot_bgcolor=COLORS["bg"], font_color=COLORS["text"],
                           showlegend=False)

    # Timeline
    if "sent_at" in df.columns and df["sent_at"].notna().any():
        df["sent_date"] = pd.to_datetime(df["sent_at"]).dt.date
        daily = df.groupby("sent_date").size().reset_index(name="count")
        timeline = px.line(daily, x="sent_date", y="count",
                          title="Intervention Volume Over Time")
    else:
        timeline = go.Figure()
    timeline.update_layout(template="plotly_dark", paper_bgcolor=COLORS["bg"],
                            plot_bgcolor=COLORS["bg"], font_color=COLORS["text"])

    # Table
    table = dash_table.DataTable(
        data=df.head(50).to_dict("records"),
        columns=[
            {"name": "Customer", "id": "first_name"},
            {"name": "Type", "id": "intervention_type"},
            {"name": "Channel", "id": "channel"},
            {"name": "Risk Tier", "id": "risk_tier_at_trigger"},
            {"name": "Status", "id": "status"},
            {"name": "Outcome", "id": "outcome"},
        ],
        style_header={
            "backgroundColor": COLORS["card_bg"],
            "color": COLORS["text"],
            "fontWeight": "bold",
        },
        style_data={
            "backgroundColor": COLORS["bg"],
            "color": COLORS["text"],
        },
        page_size=15,
        sort_action="native",
    )

    return by_type, by_channel, outcomes, timeline, table


# --- Customer Deep-Dive Callbacks ---
@app.callback(
    Output("customer-select", "options"),
    Input("main-tabs", "value"),
)
def load_customer_dropdown(tab):
    if tab != "deepdive":
        return dash.no_update
    try:
        df = pd.read_sql("SELECT customer_id, first_name, last_name FROM customers LIMIT 200", engine)
        return [{"label": f"{r['first_name']} {r['last_name']} ({r['customer_id'][:15]}...)",
                 "value": r["customer_id"]} for _, r in df.iterrows()]
    except Exception:
        return []


@app.callback(
    Output("customer-detail-container", "children"),
    Input("customer-select", "value"),
)
def update_customer_detail(customer_id):
    if not customer_id:
        return html.P("Select a customer to view details",
                      style={"color": COLORS["text_muted"]})

    customer, scores, txns, balances, interventions = load_customer_detail(customer_id)

    if customer.empty:
        return html.P("Customer not found", style={"color": COLORS["critical"]})

    c = customer.iloc[0]

    # Profile card
    profile = dbc.Card(dbc.CardBody([
        html.H5(f"{c.get('first_name', '')} {c.get('last_name', '')}",
                style={"color": COLORS["text"]}),
        html.P(f"ID: {customer_id}", style={"color": COLORS["text_muted"], "fontSize": "12px"}),
        dbc.Row([
            dbc.Col(html.Div([
                html.Small("Age", style={"color": COLORS["text_muted"]}),
                html.P(str(c.get("age", "N/A")), style={"color": COLORS["text"]}),
            ])),
            dbc.Col(html.Div([
                html.Small("City", style={"color": COLORS["text_muted"]}),
                html.P(str(c.get("city", "N/A")), style={"color": COLORS["text"]}),
            ])),
            dbc.Col(html.Div([
                html.Small("Income", style={"color": COLORS["text_muted"]}),
                html.P(f"Rs.{c.get('monthly_salary', 0):,.0f}/mo", style={"color": COLORS["text"]}),
            ])),
            dbc.Col(html.Div([
                html.Small("Credit Score", style={"color": COLORS["text_muted"]}),
                html.P(str(c.get("credit_score", "N/A")), style={"color": COLORS["text"]}),
            ])),
            dbc.Col(html.Div([
                html.Small("Tenure", style={"color": COLORS["text_muted"]}),
                html.P(f"{c.get('tenure_months', 0)} months", style={"color": COLORS["text"]}),
            ])),
        ]),
    ]), style=CARD_STYLE)

    # Risk score timeline
    if not scores.empty:
        scores["scored_at"] = pd.to_datetime(scores["scored_at"])
        risk_timeline = px.line(scores, x="scored_at", y="risk_score",
                                title="Risk Score History")
        risk_timeline.add_hline(y=0.7, line_dash="dash", line_color=COLORS["critical"],
                                annotation_text="Critical")
        risk_timeline.add_hline(y=0.5, line_dash="dash", line_color=COLORS["watch"],
                                annotation_text="Watch")
        risk_timeline.update_layout(template="plotly_dark", paper_bgcolor=COLORS["bg"],
                                    plot_bgcolor=COLORS["bg"], font_color=COLORS["text"])
    else:
        risk_timeline = go.Figure()
        risk_timeline.update_layout(template="plotly_dark", paper_bgcolor=COLORS["bg"],
                                    title="No risk scores yet")

    # Balance trend
    if not balances.empty:
        balances["timestamp"] = pd.to_datetime(balances["timestamp"])
        balance_fig = make_subplots(specs=[[{"secondary_y": True}]])
        balance_fig.add_trace(
            go.Scatter(x=balances["timestamp"], y=balances["balance"],
                      name="Account Balance", line=dict(color=COLORS["accent"])),
            secondary_y=False,
        )
        balance_fig.add_trace(
            go.Scatter(x=balances["timestamp"], y=balances["savings_balance"],
                      name="Savings", line=dict(color=COLORS["stable"])),
            secondary_y=True,
        )
        balance_fig.update_layout(title="Balance History", template="plotly_dark",
                                  paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
                                  font_color=COLORS["text"])
    else:
        balance_fig = go.Figure()

    # Transaction breakdown
    if not txns.empty:
        txn_by_cat = txns.groupby("merchant_category")["amount"].sum().sort_values(ascending=False).head(10)
        txn_fig = px.bar(x=txn_by_cat.index, y=txn_by_cat.values,
                        title="Top Spending Categories (Last 100 txns)",
                        color=txn_by_cat.index,
                        color_discrete_sequence=px.colors.qualitative.Set3)
        txn_fig.update_layout(template="plotly_dark", paper_bgcolor=COLORS["bg"],
                              plot_bgcolor=COLORS["bg"], font_color=COLORS["text"],
                              showlegend=False)
    else:
        txn_fig = go.Figure()

    return html.Div([
        profile,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=risk_timeline), md=6),
            dbc.Col(dcc.Graph(figure=balance_fig), md=6),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=txn_fig), md=12),
        ]),
    ])


# --- Model Health Callbacks ---
@app.callback(
    [Output("model-stats-row", "children"),
     Output("risk-score-trend", "figure"),
     Output("tier-distribution-trend", "figure"),
     Output("drift-table-container", "children")],
    [Input("refresh-interval", "n_intervals"),
     Input("main-tabs", "value")],
)
def update_model_health(n, tab):
    if tab != "model_health":
        return [dash.no_update] * 4

    scores_df, drift_df = load_model_health()
    empty = go.Figure().update_layout(template="plotly_dark",
                                      paper_bgcolor=COLORS["bg"],
                                      plot_bgcolor=COLORS["bg"])

    if scores_df.empty:
        msg = html.P("No model metrics available yet.", style={"color": COLORS["text_muted"]})
        return [], empty, empty, msg

    # Stats
    latest = scores_df.iloc[-1] if not scores_df.empty else {}
    total_scores = int(scores_df["num_scores"].sum()) if "num_scores" in scores_df.columns else 0

    stats = [
        dbc.Col(make_stat_card("Total Predictions", f"{total_scores:,}"), md=3),
        dbc.Col(make_stat_card("Avg Risk Score",
                              f"{latest.get('avg_risk', 0):.3f}" if not scores_df.empty else "N/A"), md=3),
        dbc.Col(make_stat_card("Score Std Dev",
                              f"{latest.get('std_risk', 0):.3f}" if not scores_df.empty else "N/A"), md=3),
        dbc.Col(make_stat_card("Drift Alerts",
                              f"{len(drift_df[drift_df.get('action_taken', '') == 'retrain_triggered'])}"
                              if not drift_df.empty else "0", COLORS["watch"]), md=3),
    ]

    # Risk score trend
    scores_df["date"] = pd.to_datetime(scores_df["date"])
    trend = go.Figure()
    trend.add_trace(go.Scatter(x=scores_df["date"], y=scores_df["avg_risk"],
                               name="Avg Risk", line=dict(color=COLORS["accent"])))
    trend.add_hline(y=0.7, line_dash="dash", line_color=COLORS["critical"])
    trend.add_hline(y=0.5, line_dash="dash", line_color=COLORS["watch"])
    trend.update_layout(title="Average Risk Score Trend", template="plotly_dark",
                        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
                        font_color=COLORS["text"])

    # Tier distribution over time
    tier_fig = go.Figure()
    for tier, color in TIER_COLORS.items():
        col = f"{tier}_count"
        if col in scores_df.columns:
            tier_fig.add_trace(go.Bar(x=scores_df["date"], y=scores_df[col],
                                     name=tier.capitalize(),
                                     marker_color=color))
    tier_fig.update_layout(title="Risk Tier Distribution Over Time",
                           barmode="stack", template="plotly_dark",
                           paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
                           font_color=COLORS["text"])

    # Drift table
    if not drift_df.empty:
        drift_table = dash_table.DataTable(
            data=drift_df.to_dict("records"),
            columns=[
                {"name": "Timestamp", "id": "detection_timestamp"},
                {"name": "Drift Score", "id": "drift_score"},
                {"name": "Action", "id": "action_taken"},
            ],
            style_header={"backgroundColor": COLORS["card_bg"], "color": COLORS["text"]},
            style_data={"backgroundColor": COLORS["bg"], "color": COLORS["text"]},
            page_size=10,
        )
    else:
        drift_table = html.P("No drift detection runs yet.", style={"color": COLORS["text_muted"]})

    return stats, trend, tier_fig, drift_table


# ─────────────────────────────────────────────
# Run Server
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host=DashboardConfig.HOST, port=DashboardConfig.PORT, debug=DashboardConfig.DEBUG)

