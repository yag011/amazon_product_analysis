from __future__ import annotations
import json
import pandas as pd
from google.cloud import bigquery
from functools import reduce
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import sys
import re
import cleantext
from cleantext import clean
import numpy as np
import textwrap
import warnings
import seaborn as sns
#import bayesian_average as ba
from collections import Counter, deque
from dataclasses import dataclass, field
from itertools import groupby
import spacy
from wordcloud import WordCloud, STOPWORDS
from spacy.lang.en.stop_words import STOP_WORDS
stop_words = STOP_WORDS
pd.set_option('display.max_colwidth', None)

# Settings read from keys.json by load_keys(); the helpers below read them when they are called.
PROJECT_ID: str | None = None
TABLE_ID: str | None = None
DUPLICATE_QUERY: str | None = None
DATA_QUERY: str | None = None
SEPARATOR = "|"
_CHILDREN_WIDTH = 44
_NUMBER = re.compile(r"[+-]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)")

DUPLICATE_QUERY = """
WITH flagged_products AS (
  SELECT
    * EXCEPT (
    -- 1. Columns to drop outright
    product_link, img_link, review_id, user_name, user_id,
    -- 3. Source columns replaced by the concatenated versions below
    product_name, about_product, review_title, review_content,
    -- actual price
    actual_price
  )
  -- 4. Convert rating to FLOAT (unparseable values become NULL instead of failing)
  REPLACE (SAFE_CAST(rating AS FLOAT64) AS rating),
  -- 2. Concatenate text columns (NULL-safe: a missing part doesn't wipe out the result)
  ARRAY_TO_STRING([product_name, about_product], ' ') AS product,
  ARRAY_TO_STRING([review_title, review_content], ' ') AS review,
  COUNT(*) OVER(PARTITION BY product_id) AS product_count
  FROM
    `{table_id}`
)
SELECT
  * EXCEPT(product_count)
FROM
  flagged_products
WHERE
  product_count > 1;
"""

DATA_QUERY = """
SELECT
  * EXCEPT (
    -- 1. Columns to drop outright
    product_link, img_link, review_id, user_name, user_id,
    -- 3. Source columns replaced by the concatenated versions below
    product_name, about_product, review_title, review_content,
     -- actual price
    actual_price
  )
  -- 4. Convert rating to FLOAT (unparseable values become NULL instead of failing)
  REPLACE (SAFE_CAST(rating AS FLOAT64) AS rating),
  -- 2. Concatenate text columns (NULL-safe: a missing part doesn't wipe out the result)
  ARRAY_TO_STRING([product_name, about_product], ' ') AS product,
  ARRAY_TO_STRING([review_title, review_content], ' ') AS review
FROM `{table_id}`
"""

CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

INVARIANT = {"series", "species", "news", "jeans", "shorts", "pants", "scissors", "clothes"}
IRREGULAR = {"men": "man", "women": "woman", "children": "child", "feet": "foot",
             "teeth": "tooth", "mice": "mouse", "people": "person"}
IE_WORDS = {"movies", "cookies", "hoodies", "selfies", "smoothies", "beanies",
            "brownies", "zombies", "goodies", "rookies"}


def load_keys(path):
    """Read keys.json, store the settings in this module, and return them as a tuple."""
    global PROJECT_ID, TABLE_ID, SEPARATOR, _CHILDREN_WIDTH, DUPLICATE_QUERY, DATA_QUERY
    with open(path, "r") as file:
        data = json.load(file)
    PROJECT_ID = data["PROJECT_ID"]
    TABLE_ID = f"{PROJECT_ID}.{data['TABLE_ID']}"
    SEPARATOR = data["SEPARATOR"]
    _CHILDREN_WIDTH = data["_CHILDREN_WIDTH"]
    DUPLICATE_QUERY = DUPLICATE_QUERY.format(table_id=TABLE_ID)
    DATA_QUERY = DATA_QUERY.format(table_id=TABLE_ID)
    return PROJECT_ID, TABLE_ID, SEPARATOR, _NUMBER, _CHILDREN_WIDTH

def duplicate_query(project_id: str | None = None) -> pd.DataFrame:
    """Run the duplicate identification query and return the result as a DataFrame."""
    client = bigquery.Client(project=project_id or PROJECT_ID)
    return client.query(DUPLICATE_QUERY).to_dataframe()

def calculate_duplicate_value() -> float:
    duplicate_data = duplicate_query()
    duplicate_data['estimated_revenue'] = duplicate_data['discounted_price'] * duplicate_data['rating_count']
    TOTAL_DUPLICATED_REVENUE = duplicate_data['estimated_revenue'].sum()
    duplicate_data = duplicate_data.drop_duplicates(subset=['product_id'], keep='last')
    NEW_DUPLICATED_REVENUE = duplicate_data['estimated_revenue'].sum()
    TOTAL_DUPLICATED_REVENUE = TOTAL_DUPLICATED_REVENUE - NEW_DUPLICATED_REVENUE  
    return TOTAL_DUPLICATED_REVENUE

def load_amazon_sales(project_id: str | None = None) -> pd.DataFrame:
    """Run the cleaning query and return the result as a DataFrame."""
    client = bigquery.Client(project=project_id or PROJECT_ID)
    return client.query(DATA_QUERY).to_dataframe()

def clean_column(series):
    return (
        series.fillna("")
        .astype(str)
        .apply(lambda x: cleantext.clean(
            x,
            no_line_breaks=True,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=True,
            no_numbers=True,
            no_digits=True,
            no_currency_symbols=True,
            no_punct=True,
            no_emoji=True,
            replace_with_url="",
            replace_with_email="",
            replace_with_phone_number="",
            replace_with_number="",
            replace_with_digit="",
            replace_with_currency_symbol="",
            replace_with_punct="",
        ))
        .apply(lambda x: "".join(c for c in x if c.isalpha() or c.isspace()))
        .apply(lambda x: " ".join(w for w in x.split() if w not in stop_words and len(w) >= 3))
    )

def modify_and_calculate_skipped_value(amazon_sales_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Clean the sales frame and return (cleaned_df, revenue skipped because of null values)."""
    amazon_sales_df = amazon_sales_df.drop_duplicates(subset=['product_id'], keep='last')

    amazon_sales_df['estimated_revenue'] = amazon_sales_df['discounted_price'] * amazon_sales_df['rating_count']
    amazon_sales_df["specific_item"] = amazon_sales_df["category"].str.split("|").str[-1]
    amazon_sales_df["category_group"] = amazon_sales_df["category"].str.split("|").str[0]
    null_values = amazon_sales_df[amazon_sales_df.isna().any(axis=1)]
    amazon_sales_df = add_bayesian_rating_fast(amazon_sales_df)
    SKIPPED_NULL_REVENUE = null_values['estimated_revenue'].sum()
    amazon_sales_df = amazon_sales_df.dropna()
    for col in ["product", "review"]:
        amazon_sales_df[col] = clean_column(amazon_sales_df[col])
    return amazon_sales_df, SKIPPED_NULL_REVENUE

def create_corr_matrix(df: pd.DataFrame, method: str = "pearson") -> Figure:
    """Correlation heatmap of every numeric column in `df`, returned as a matplotlib Figure.
 
    Shows the full matrix (both triangles and the diagonal) with every value annotated.
    `method` is "pearson", "spearman" or "kendall".
    Raises ValueError if `df` has fewer than three numeric columns.
    """
    df = df[['discounted_price', 'discount_percentage', 'bayesian_rating', 'rating_count', 'estimated_revenue']]
    corr = df.corr(method='pearson', numeric_only=True)
    n = len(corr)
    labels = [str(col).replace("_", " ").title() for col in corr.columns]
 
    side = max(6, 0.8 * n + 2)  # grow the figure with the number of columns
    fig, ax = plt.subplots(figsize=(side, side * 0.85))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        annot_kws={"size": max(6, 12 - n // 2)},  # smaller numbers when there are many columns
        cmap="coolwarm",
        vmin=-1, vmax=1, center=0,  # red = positive, blue = negative
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": f"{method.title()} correlation"},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
 
    # Undefined correlations (NaN, e.g. a column with a single repeated value) are left blank by
    # seaborn, so label them to show nothing is missing.
    for row, col in zip(*np.where(corr.isna().to_numpy())):
        ax.text(col + 0.5, row + 0.5, "n/a", ha="center", va="center", color="gray")
 
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    return fig
 

def list_to_nested_dict(items: list[str]) -> dict:
    """['A', 'B', 'C'] -> {'A': {'B': {'C': {}}}}  (each item sits under the one before it)."""
    nested: dict = {}
    for key in reversed(items):
        nested = {key: nested}
    return nested

def _effective_children(node, path: tuple, ancestors: tuple, skip) -> list[tuple[tuple, object, tuple]]:
    """
    Children of `node` as (path, child, ancestor ids), in order. A child whose label
    `skip` accepts is left out and its own children take its place, so the layers
    below simply close up. Paths always stay the full original paths.
    """
    out = []
    pending = [(k, c, path, ancestors) for k, c in node.items()][::-1] if _is_category(node) else []
    while pending:
        key, child, parent_path, parent_ids = pending.pop()
        child_path = parent_path + (key,)
        if _is_category(child) and id(child) in parent_ids:
            raise ValueError(f"circular reference at {child_path}")
        child_ids = parent_ids + (id(child),)
        if skip is not None and skip(key):
            if _is_category(child):
                pending.extend([(k, c, child_path, child_ids) for k, c in child.items()][::-1])
        else:
            out.append((child_path, child, child_ids))
    return out

def _is_category(node) -> bool:
    """True for a non-empty dict-like node (a dict, or e.g. a pandas Series); anything else is an item."""
    return hasattr(node, "items") and len(node) > 0

def _is_numeric_label(key) -> bool:
    """True when a node's name is just a number, e.g. 2024, -3, 1,299.50 (but not "2024-01" or "$5")."""
    return _NUMBER.fullmatch(str(key).strip()) is not None

def bfs_layers(tree, skip=None) -> list[list[tuple[tuple, tuple | None]]]:
    """
    Breadth-first search of a nested dict, one layer at a time.

    Returns a list of layers. layers[0] is the root [((), None)]; layers[d] holds
    every node d levels below it as (path, parent_path), in BFS order, so the
    children of one parent are always next to each other.

    skip : optional function of a node's label; nodes it accepts are left out and
           their children move up into their place (used to hide numeric nodes).

    Raises ValueError on a circular reference instead of looping forever.
    """
    layers: list[list[tuple[tuple, tuple | None]]] = []
    queue = deque([((), None, tree, (id(tree),))])  # (path, parent_path, node, ids of its ancestors)
    while queue:
        layers.append([(path, parent) for path, parent, _node, _anc in queue])  # the whole queue = one layer
        for _ in range(len(queue)):  # ...so consume exactly that many nodes
            path, _parent, node, ancestors = queue.popleft()
            for child_path, child, child_ids in _effective_children(node, path, ancestors, skip):
                queue.append((child_path, path, child, child_ids))
    return layers

def _leaf_paths(tree) -> list[tuple]:
    """Paths of all leaves in depth-first, left-to-right order (the order `revenues` is matched in)."""
    paths = []
    stack = [((key,), child) for key, child in tree.items()][::-1]
    while stack:
        path, node = stack.pop()
        if _is_category(node):
            stack.extend([(path + (key,), child) for key, child in node.items()][::-1])
        else:
            paths.append(path)
    return paths

def add_bayesian_rating_fast(df, min_entries=3, out_col='bayesian_rating'):
    g = df.groupby('specific_item')
    rv = df['rating'] * df['rating_count']

    C = rv.groupby(df['specific_item']).transform('sum') / g['rating_count'].transform('sum')
    m = g['rating_count'].transform('median')
    n = g['rating_count'].transform('size')           # rows per item

    score = (df['rating_count'] * df['rating'] + m * C) / (df['rating_count'] + m)

    #df = df.copy()
    df[out_col] = score.where(n >= min_entries, df['rating'])  # raw rating for small groups
    return df

def _wrap_names(names: list[str], width: int) -> list[str]:
    """Join names with ", ", starting a new line before a name would pass `width` (names are never split)."""
    lines: list[str] = []
    current = ""
    for i, item in enumerate(names):
        piece = item + ("," if i < len(names) - 1 else "")
        if current and len(current) + 1 + len(piece) > width:
            lines.append(current)
            current = piece
        else:
            current = f"{current} {piece}" if current else piece
    if current:
        lines.append(current)
    return lines

def format_revenue_layers(tree: dict, totals: dict[tuple, float], hide_numeric: bool = True) -> str:
    """
    All BFS layers of `tree` in one table, top layer first (original BFS order).

    One row per name per layer: nodes that share a name inside the same layer are
    consolidated into a single row with their combined revenue. Each row lists its
    children by name, and a child name is listed only ONCE per layer. If the same
    name sits under several parents, it is listed under the parent where it earns
    the most revenue (ties: the parent shown first); the other parents say
    "↳ shares children with <that parent>". A node with no children is flagged
    "■ unique item". Inside a layer the unique items come first, ranked from highest
    to lowest revenue, followed by the other nodes in BFS order. TOTAL's children
    are not repeated: they are simply all of layer 1.

    hide_numeric : leave nodes whose name is a plain number (2024, 1,299.50, ...) out
                   of the table. Their revenue still counts in every parent's total,
                   their own children move up a layer, and a node whose only children
                   were numeric shows as a unique item. Pass False to show them.
    """
    layers = bfs_layers(tree, skip=_is_numeric_label if hide_numeric else None)

    kids_of: dict = {}  # path -> paths of its children
    for layer in layers:
        for path, parent in layer:
            if parent is not None:
                kids_of.setdefault(parent, []).append(path)

    grand_total = 0.0
    for key in tree:
        grand_total += totals.get((key,), 0.0)

    def revenue(path: tuple) -> float:
        return totals.get(path, 0.0) if path else grand_total

    def name(path: tuple) -> str:
        return str(path[-1]) if path else "TOTAL"

    # Step 1: consolidate each layer by name and decide the order the rows will be shown in.
    merged_layers: list[tuple[dict, list[str]]] = []  # (name -> entry, display order of the names)
    for depth, layer in enumerate(layers):
        merged: dict = {}
        for path, _ in layer:
            entry = merged.setdefault(name(path), {"revenue": 0.0, "kids": {}})
            entry["revenue"] += revenue(path)
            for kid in kids_of.get(path, []):  # child name -> revenue it earns under this parent name
                entry["kids"][name(kid)] = entry["kids"].get(name(kid), 0.0) + revenue(kid)
        for entry in merged.values():
            entry["unique"] = depth > 0 and not entry["kids"]
        uniques = [n for n, entry in merged.items() if entry["unique"]]
        uniques.sort(key=lambda n: merged[n]["revenue"], reverse=True)  # ties keep BFS order
        others = [n for n, entry in merged.items() if not entry["unique"]]  # original BFS order
        merged_layers.append((merged, uniques + others))

    # Step 2: give every child name to exactly one parent per layer.
    listed: list[dict] = []  # per layer: parent name -> child names it lists (in the order of the layer below)
    shares: list[dict] = []  # per layer: parent name -> names of the parents it shares children with
    for depth, (merged, order) in enumerate(merged_layers):
        listed.append({n: [] for n in order})
        shares.append({n: [] for n in order})
        if depth + 1 == len(merged_layers):
            continue
        rank = {n: i for i, n in enumerate(order)}
        owners_of: dict = {}  # child name -> parents that have it, in display order
        for parent_name in order:
            for child in merged[parent_name]["kids"]:
                owners_of.setdefault(child, []).append(parent_name)
        for child in merged_layers[depth + 1][1]:
            owners = owners_of[child]
            primary = max(owners, key=lambda n: (merged[n]["kids"][child], -rank[n]))
            listed[depth][primary].append(child)
            for other in owners:
                if other != primary and primary not in shares[depth][other]:
                    shares[depth][other].append(primary)

    # Step 3: build the rows.
    rows: list[tuple[str, str, float, list[str]]] = []  # (layer number, node, revenue, children lines)
    for depth, (merged, order) in enumerate(merged_layers):
        for i, node in enumerate(order):
            entry = merged[node]
            if depth == 0:  # TOTAL: its children are the whole of layer 1, so listing them would be redundant
                child_lines: list[str] = []
            elif entry["unique"]:
                child_lines = ["■ unique item"]
            else:
                child_lines = _wrap_names(listed[depth][node], _CHILDREN_WIDTH)
                partners = shares[depth][node]
                if partners:
                    child_lines += _wrap_names(["↳ shares children with " + partners[0]] + partners[1:], _CHILDREN_WIDTH)
            rows.append((str(depth) if i == 0 else "", "● TOTAL" if depth == 0 else node, entry["revenue"], child_lines))

    gutter_w = max(len("Layer"), len(str(len(layers) - 1)))
    left_w = max(len(left) for _, left, _, _ in rows)
    amount_w = max(len("Revenue"), max(len(f"{value:,.2f}") for _, _, value, _ in rows))
    children_w = max(len("■ unique item"), max((len(text) for *_, texts in rows for text in texts), default=0))
    body_w = left_w + 4 + amount_w + 2 + children_w
    title = "Cumulative revenue by BFS layer   (■ unique item = node has no children)"

    lines = [
        title,
        "═" * max(len(title), gutter_w + 3 + body_w),
        f"{'Layer'.rjust(gutter_w)} │ {'Node'.ljust(left_w + 4)}{'Revenue'.rjust(amount_w)}  Children",
        "─" * (gutter_w + 1) + "┼" + "─" * (body_w + 1),
    ]
    for layer_no, left, value, child_lines in rows:
        dots = "·" * (left_w - len(left) + 2)
        first = child_lines[0] if child_lines else ""
        lines.append(f"{layer_no.rjust(gutter_w)} │ {left} {dots} {f'{value:,.2f}'.rjust(amount_w)}  {first}".rstrip())
        for extra in child_lines[1:]:  # continuation of a long children list
            lines.append(f"{' ' * gutter_w} │ {' ' * (left_w + 4 + amount_w)}  {extra}".rstrip())
    return "\n".join(lines)

def merge_into(
    target: dict,
    source: dict,
    revenues: list[float] | np.ndarray,
    totals: dict[tuple, float],
    show: bool = True,
    hide_numeric: bool = True,
) -> dict:
    """
    Merge `source` into `target` (in place) and return `target`, adding up
    revenue for every item and category along the way.

    The merge is a breadth-first search: `source` is explored layer by layer
    and each node is created in `target` (or found, if it already exists), then
    revenue is rolled up from the deepest layer back towards the top.

    revenues : one revenue per leaf (item) of `source`, in the order the leaves
               appear (depth-first, left to right, following the dict's
               insertion order). Can be a list, tuple, NumPy array or pandas
               Series (pass the Series itself, or `s.to_numpy()` / `s.tolist()`;
               note that `s.values` has no parentheses).
    totals   : running revenue per node, keyed by its full path, e.g.
               ("Electronics", "Phones", "iPhone"). Updated in place. A node's
               total is its own revenue plus everything beneath it, so a
               category holds the cumulative revenue of all of its items. When
               the same path shows up again in a later merge, revenues are summed.
    show     : print the cumulative revenue of every node in `target`, layer by
               layer, once the merge is done (same-named nodes in a layer are
               consolidated into one row, children are listed by name, and the
               unique items are ranked from highest to lowest revenue inside each
               layer; each child name is listed once per layer). `totals` itself
               stays per path. Labels repeated under one parent in `source` (e.g.
               a pandas Series with repeated index labels) are one node: their
               revenues add up. Use show=False when
               merging in a loop, then call
               print(format_revenue_layers(target, totals)) once at the end.
    hide_numeric : keep nodes whose name is a plain number (2024, 1,299.50, ...) out
               of the printed table (their revenue still counts in the totals).
    """
    revenue_list = np.asarray(revenues, dtype=np.float64).ravel().tolist()  # plain Python floats
    #n_leaves = _count_leaves(source)  # also rejects circular references, before anything is touched
    #if len(revenue_list) != n_leaves:  # check first, so a bad call never half-merges
    #    raise ValueError(f"`source` has {n_leaves} leaves but {len(revenue_list)} revenues were given")
    item_revenue: dict = {}
    for path, r in zip(_leaf_paths(source), revenue_list):  # repeated labels under one parent add up instead of overwriting
        item_revenue[path] = item_revenue.get(path, 0.0) + r

    # Pass 1 (top-down, layer by layer): mirror every source node into target.
    layers = bfs_layers(source)
    nodes = {(): target}  # path -> the dict that represents that node inside `target`
    subtotal = {}  # path -> cumulative revenue of that node coming from this source
    for layer in layers:
        for path, parent in layer:
            if parent is not None:
                nodes[path] = nodes[parent].setdefault(path[-1], {})
            subtotal[path] = item_revenue.get(path, 0.0)

    # Pass 2 (bottom-up): push each layer's revenue into its parents, deepest layer first.
    pushed: set = set()  # a label repeated under one parent is one node: push it up only once
    for layer in reversed(layers):
        for path, parent in layer:
            if parent is not None and path not in pushed:
                pushed.add(path)
                subtotal[parent] += subtotal[path]

    for path, value in subtotal.items():
        if path:  # skip the synthetic root
            totals[path] = totals.get(path, 0.0) + value

    if show:
        print(format_revenue_layers(target, totals, hide_numeric))
    return target

def merge_category_column(
    series: pd.DataFrame, sep: str | None = None, drop_empty: bool = True) -> dict | list:
    """Turn a Series of 'A|B|C' strings into a single merged nested dictionary.

    Set drop_empty=False to keep the {} leaves, which is the shape that
    load_tree_to_neo4j in category_graph.py expects.
    """
    sep = SEPARATOR
    category_lists = series['category'].drop_duplicates().str.split(sep)   # 1. string -> list
    revenue = series.groupby("category", sort=False)["estimated_revenue"].sum().tolist()  # sort=False keeps first-appearance order, matching category_lists
    nested_dicts = category_lists.apply(list_to_nested_dict)            # 2. list -> nested dict
    merged = merge_into({}, nested_dicts, revenue, {})                       # 3. merge into one
    return merged        # 4. drop empty values

def create_word_cloud(df: pd.DataFrame, column: str, word: str = "") -> Figure:
    """Word cloud of the most frequent words in df[column], returned as a matplotlib Figure.

    The words in `word` (e.g. the selected category) are added to the stop words so
    they don't dominate the cloud. Raises ValueError when no words are left to plot.
    The caller shows (and closes) the Figure.
    """
    text = " ".join(df[column].dropna().astype(str))
    added_subwords = set(split_subwords(word))
    custom_stops = STOPWORDS.union({"product", "item", "one", "get", "good", "bad"}).union(stop_words).union(added_subwords)
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="viridis",
        stopwords=custom_stops,
        max_words=25,
        collocations=False,  # avoids repeated bigrams like "great product"
        random_state=42,     # same layout on every Streamlit rerun
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig

def load_drop_list() -> set: 
    items_set = set(amazon_sales_df['specific_item'])
    groups_set = set(list(amazon_sales_df["category_group"]))
    return items_set.union(groups_set)

def split_subwords(text: str, singular: bool = True) -> list[str]:
    """Split on non-alphanumeric symbols, then on capitalization; optionally singularize."""
    text = re.sub(r"['’]s\b", "", text)                            # drop possessives: Men's -> Men
    chunks = re.split(r"[^A-Za-z0-9]+", text)
    words = [w for chunk in chunks for w in CAMEL.findall(chunk)]
    return [singularize(w) for w in words] if singular else words

def singularize(word: str) -> str:
    """Cheap rule-based singularizer; keeps the original capitalization."""
    low = word.lower()
    if low in IRREGULAR:
        return IRREGULAR[low].capitalize() if word[0].isupper() else IRREGULAR[low]
    if word.isupper() or len(word) <= 3 or low in INVARIANT:      # acronyms / short words untouched
        return word
    if low in IE_WORDS:
        return word[:-1]                                          # Movies -> Movie
    if low.endswith("ies"):
        return word[:-3] + "y"                                    # Batteries -> Battery
    if low.endswith(("sses", "shes", "ches", "xes")):
        return word[:-2]                                          # Watches -> Watch, Boxes -> Box
    if low.endswith(("ss", "us", "is")):
        return word                                               # Glass, Bus, Analysis
    if low.endswith("s"):
        return word[:-1]                                          # Pens -> Pen
    return word