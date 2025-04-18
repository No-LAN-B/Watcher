# src/features/graph_features.py

import os
import pandas as pd
import networkx as nx

def build_correlation_graph(returns_df: pd.DataFrame, threshold: float = 0.7) -> nx.Graph:
    """
    Build an undirected graph where each node is a ticker,
    and an edge exists between two tickers if the absolute
    correlation of their returns exceeds `threshold`.
    """
    # 1) Compute the correlation matrix among tickers
    corr = returns_df.corr()

    # 2) Initialize an empty graph
    G = nx.Graph()

    # 3) Add each ticker as a node
    for ticker in corr.columns:
        G.add_node(ticker)

    # 4) For each pair of tickers, add an edge if |corr| >= threshold
    for i, t1 in enumerate(corr.columns):
        for j, t2 in enumerate(corr.columns):
            if j <= i:
                continue
            weight = corr.loc[t1, t2]
            if abs(weight) >= threshold:
                G.add_edge(t1, t2, weight=weight)

    return G

def compute_graph_features(G: nx.Graph) -> pd.DataFrame:
    """
    Given a graph G, compute:
      - degree centrality for each node
      - community assignment via greedy modularity
    Returns a DataFrame indexed by ticker.
    """
    # 1) Degree centrality: how many connections each node has
    deg_cent = nx.degree_centrality(G)

    # 2) Community detection
    communities = list(nx.community.greedy_modularity_communities(G))
    # assign each node to a community ID (0,1,...)
    node_comm = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            node_comm[node] = cid

    # 3) Build a DataFrame of these features
    df = pd.DataFrame({
        "degree_centrality": deg_cent,
        "community_id": node_comm
    })
    df.index.name = "Ticker"
    return df

def main():
    # A) Load all processed return series into one DataFrame
    folder = "data/processed"
    returns = {}
    for fname in os.listdir(folder):
        if not fname.endswith("_features.csv"):
            continue
        ticker = fname.split("_")[0]
        df = pd.read_csv(
            os.path.join(folder, fname),
            index_col="Date",
            parse_dates=True
        )
        returns[ticker] = df["return"]

    returns_df = pd.DataFrame(returns).dropna(how="any")

    # B) Build the correlation graph
    G = build_correlation_graph(returns_df, threshold=0.7)

    # C) Compute graph metrics and save to CSV
    graph_feats = compute_graph_features(G)
    os.makedirs(folder, exist_ok=True)
    graph_feats.to_csv(os.path.join(folder, "graph_features.csv"))
    print("Saved graph_features.csv with columns:", list(graph_feats.columns))

if __name__ == "__main__":
    main()
