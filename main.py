from __future__ import annotations
import streamlit as st
import contextlib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; must be set before pyplot is imported (utility_functions imports it)

import matplotlib.pyplot as plt
import pandas as pd

from utility_functions import (
    calculate_duplicate_value,
    create_corr_matrix,
    create_word_cloud,
    load_amazon_sales,
    load_keys,
    merge_category_column,
    modify_and_calculate_skipped_value,
)

st.set_page_config(page_title="Amazon sales stats", page_icon="🛒", layout="wide")

load_keys(Path(__file__).with_name("keys.json"))  # sets the project / table / separator used by the queries


# ------------------------------------------------------------------ data (cached)
@st.cache_data(ttl=3600, show_spinner="Loading sales data from BigQuery...")
def load_data() -> tuple[pd.DataFrame, float, float]:
    """Query BigQuery once an hour; every click and dropdown change reuses the cached result."""
    duplicated_revenue = float(calculate_duplicate_value())
    sales_df, skipped_revenue = modify_and_calculate_skipped_value(load_amazon_sales())
    return sales_df, duplicated_revenue, float(skipped_revenue)


@st.cache_data(show_spinner=False)
def revenue_report(df: pd.DataFrame) -> str:
    """merge_category_column() prints its revenue table, so capture that text to show it in the page."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        merge_category_column(df)
    return printed.getvalue()


# ------------------------------------------------------------------ plot helpers (same shape on purpose)
def show_figure(fig) -> None:
    """Display a matplotlib Figure and release it; Streamlit reruns the script on every interaction."""
    try:
        st.pyplot(fig)  # fills the container width by default
    finally:
        plt.close(fig)  # always release the figure, even if st.pyplot raises; any error still propagates


def show_corr_matrix(df: pd.DataFrame) -> None:
    """Run create_corr_matrix on df and display the heatmap in the Streamlit page."""
    if len(df) <= 2:
        st.info("At least three products are needed to compute a correlation matrix.")
        return
    try:
        fig = create_corr_matrix(df)
    except KeyError as exc:  # a required column is missing from df
        st.error(f"Cannot build the correlation matrix: {exc.args[0]}")
        return
    except ValueError:  # none of the columns are numeric, so there is nothing to correlate
        st.warning("There are no numeric columns to correlate for this selection.")
        return
    st.pyplot(fig)
    plt.close(fig)  # free the figure; Streamlit reruns the script on every interaction
 


def show_word_cloud(df: pd.DataFrame, column: str, word: str = "") -> None:
    """Run create_word_cloud on df[column] and display the result in the Streamlit page."""
    if column not in df.columns:
        st.error(f"Column '{column}' was not found in the dataframe.")
        return
    try:
        fig = create_word_cloud(df, column, word)
    except ValueError:  # WordCloud raises this when no words are left to plot
        st.warning("There are no words left to plot for this selection.")
        return
    show_figure(fig)


# ------------------------------------------------------------------ the two views
def show_overall_stats(df: pd.DataFrame, duplicated_revenue: float, skipped_revenue: float) -> None:
    stats = pd.DataFrame(
        {
            "Metric": ["TOTAL_DUPLICATED_REVENUE", "SKIPPED_NULL_REVENUE"],
            "Value": [duplicated_revenue, skipped_revenue],
        }
    ).set_index("Metric")
    st.subheader("Skipped value summary")
    st.table(stats.style.format({"Value": "{:,.2f}"}))

    st.subheader("Sales report by department tier")
    st.code(revenue_report(df), language=None)


def show_item_details(df: pd.DataFrame, choice: str | None) -> None:
    if choice is None:
        st.info("Please choose a category or product from the dropdown above.")
        return

    selection_df = df[(df["specific_item"] == choice) | (df["category_group"] == choice)]
    st.subheader(f"Details for {choice}")
    st.caption(f"{len(selection_df):,} products in this selection")

    revenue_tab, cloud_tab, corr_tab = st.tabs(["Revenue breakdown", "Word clouds", "Correlation matrix"])
    with revenue_tab:
        st.code(revenue_report(selection_df), language=None)
    with cloud_tab:
        product_col, review_col = st.columns(2)
        with product_col:
            st.markdown("##### Product")
            show_word_cloud(selection_df, "product", word=choice)  # `word` is added to the stop words
        with review_col:
            st.markdown("##### Review")
            show_word_cloud(selection_df, "review", word=choice)
    with corr_tab:
        show_corr_matrix(selection_df)


# ------------------------------------------------------------------ page
st.title("Visualizing Amazon Sales!")
st.markdown(
    """
    Welcome! In this app, you can do two things:
    1. Click **View overall stats** to view the overall sales report by each tiered department.
    You will also see TOTAL_DUPLICATED_REVENUE and SKIPPED_NULL_REVENUE, which are repeated and incomplete rows not included in the final calculation

    2. Select from the dropdown menu, then click **View specific item** to view the revenue breakdown and analytic graphs by overarching department or product name
    """
)

amazon_sales_df, TOTAL_DUPLICATED_REVENUE, SKIPPED_NULL_REVENUE = load_data()
SELECTION_LIST = set(amazon_sales_df["specific_item"]) | set(amazon_sales_df["category_group"])

choice = st.selectbox(
    "Choose one category or product to view in detail",
    options=sorted(SELECTION_LIST),  # sets are unordered, so sort for a stable order
    index=None,  # nothing preselected
    placeholder="Start typing to search...",
)


def set_view(name: str) -> None:
    st.session_state["view"] = name


view = st.session_state.get("view")  # which button was pressed last; survives reruns such as changing the dropdown

overall_col, item_col, _spacer = st.columns([1, 1, 2])  # both buttons on the same row
with overall_col:
    st.button(
        "View overall stats 🎈",
        key="btn_overall",
        on_click=set_view,
        args=("overall",),
        type="primary" if view == "overall" else "secondary",
    )
with item_col:
    st.button(
        "View specific item",
        key="btn_item",
        on_click=set_view,
        args=("item",),
        type="primary" if view == "item" else "secondary",
    )

if view == "overall":
    show_overall_stats(amazon_sales_df, TOTAL_DUPLICATED_REVENUE, SKIPPED_NULL_REVENUE)
elif view == "item":
    show_item_details(amazon_sales_df, choice)
