#!/usr/bin/env python
# coding: utf-8

# # Final working

# In[ ]:


import os
import re
import time
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from tqdm import tqdm
from itertools import combinations

class SciNexusUniversalAuditor:
    def __init__(self, email, ollama_model="llama3"):
        self.email = email
        self.ollama_model = ollama_model
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.ollama_url = "http://localhost:11434/api/generate"

    def _ask_ollama_json(self, prompt):
        """Forces Ollama into JSON mode for robust expansion."""
        payload = {"model": self.ollama_model, "prompt": prompt, "format": "json", "stream": False}
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            raw = response.json().get("response", "")
            data = json.loads(raw)
            return data.get("terms", []) if isinstance(data, dict) else data
        except: return []

    def get_semantic_expansion(self, topic):
        print(f"--- Analyzing '{topic}' Domain Architecture ---")
        prompt = f"""
        Act as a Research Lead. Identify 6 technical gene acronyms or sub-topics for '{topic}'.
        Return JSON object: {{"terms": ["TERM1", "TERM2", ...]}}
        """
        terms = self._ask_ollama_json(prompt)
        valid = [str(t).strip().upper() for t in terms if len(str(t)) > 1]
        # Universal Fallback for KRAS if LLM fails
        if not valid and "KRAS" in topic.upper(): return ["RAF1", "MEK", "ERK", "MAPK", "SOS1"]
        return valid if valid else [topic.upper()]

    def run_discovery(self, topic, location="Ireland", years=15, top_n=10):
        # 1. Expand Query
        sub_topics = self.get_semantic_expansion(topic)
        search_query = f"({topic} OR " + " OR ".join(sub_topics) + ")"
        date_cutoff = (datetime.now() - timedelta(days=years * 365)).strftime("%Y/%m/%d")
        
        # 2. Global Toggle
        loc_filter = "" if location.lower() == "global" else f" AND ({location}[Affiliation])"
        final_query = f"({search_query}){loc_filter} AND ({date_cutoff}[PDAT] : 2026[PDAT])"
        print(f"Universal Query: {final_query}")

        # 3. PubMed ESearch
        resp = requests.get(f"{self.base_url}esearch.fcgi", params={"db": "pubmed", "term": final_query, "retmax": 500, "email": self.email})
        ids = [id_node.text for id_node in ET.fromstring(resp.content).findall(".//Id")]
        if not ids: return None

        # 4. Batch Mining with Progress Bar
        all_records = []
        network_edges = []
        chunk_size = 50 
        
        print(f"\nMining Expert Metadata for {location}...")
        for i in tqdm(range(0, len(ids), chunk_size), desc="PubMed Harvesting", bar_format="{l_bar}{bar:30}{r_bar}"):
            batch = ids[i : i + chunk_size]
            try:
                fetch_resp = requests.get(f"{self.base_url}efetch.fcgi", params={"db": "pubmed", "id": ",".join(batch), "retmode": "xml", "email": self.email})
                if not fetch_resp.content: continue
                root = ET.fromstring(fetch_resp.content)
                
                for article in root.findall(".//PubmedArticle"):
                    title = (article.find(".//ArticleTitle").text or "No Title").strip()
                    authors = article.findall(".//Author")
                    paper_authors = []
                    
                    for auth in authors:
                        last = auth.find("LastName")
                        fore = auth.find("ForeName")
                        if last is not None:
                            name = f"{last.text.upper()}, {fore.text[0].upper() if fore is not None else ''}"
                            # Affiliation scope check
                            is_local = True if location.lower() == "global" else any(location.upper() in (aff.text or "").upper() for aff in auth.findall(".//Affiliation"))
                            
                            if is_local:
                                all_records.append({"Researcher": name, "Title": title})
                                paper_authors.append(name)
                    
                    if len(paper_authors) > 1:
                        network_edges.extend(list(combinations(sorted(paper_authors), 2)))
                time.sleep(0.35) 
            except Exception: continue

        # 5. Process Results
        df = pd.DataFrame(all_records)
        if df.empty: return None
        top_df = df['Researcher'].value_counts().nlargest(top_n).reset_index()
        top_df.columns = ['Researcher', 'Papers']
        top_names = set(top_df['Researcher'].tolist())

        # 6. EXPORTS & PLOTS
        # A. CSVs (Audit & Cytoscape)
        df[df['Researcher'].isin(top_names)].to_csv(f"audit_{topic}_{location}.csv", index=False)
        edge_df = pd.DataFrame(network_edges, columns=['Source', 'Target']).groupby(['Source', 'Target']).size().reset_index(name='Weight')
        edge_df.to_csv(f"network_cytoscape_{location}.csv", index=False)

        # B. Interactive Visuals (Plotly)
        self._create_interactive_network(edge_df, f"global_network_{location}.html", f"Global {location} Ecosystem")
        
        # C. Elite Static Plot (Matplotlib)
        self._create_static_elite_plot(edge_df, top_names, topic, location)

        print(f"\nCompleted! Files generated:\n1. audit_{topic}_{location}.csv\n2. network_cytoscape_{location}.csv\n3. global_network_{location}.html\n4. top_10_elite_map.png")
        return top_df

    def _create_interactive_network(self, edge_df, filename, title):
        G = nx.from_pandas_edgelist(edge_df, 'Source', 'Target', ['Weight'])
        pos = nx.spring_layout(G, k=0.4)
        
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])

        node_x, node_y, node_text, node_size = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]; node_x.append(x); node_y.append(y); node_text.append(node)
            node_size.append(10 + G.degree(node) * 2)

        fig = go.Figure(data=[
            go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.7, color='#888'), hoverinfo='none', mode='lines'),
            go.Scatter(x=node_x, y=node_y, mode='markers', text=node_text, hoverinfo='text',
                       marker=dict(showscale=True, colorscale='Reds', size=node_size, color=[G.degree(n) for n in G.nodes()]))
        ], layout=go.Layout(title=title, template="plotly_white", showlegend=False))
        fig.write_html(filename)

    def _create_static_elite_plot(self, edge_df, top_names, topic, location):
        elite_edges = edge_df[edge_df['Source'].isin(top_names) & edge_df['Target'].isin(top_names)]
        G = nx.from_pandas_edgelist(elite_edges, 'Source', 'Target', ['Weight'])
        for name in top_names: 
            if name not in G: G.add_node(name)
            
        plt.figure(figsize=(12, 10), dpi=300)
        pos = nx.kamada_kawai_layout(G)
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', width=2)
        nx.draw_networkx_nodes(G, pos, node_size=[500 + (G.degree(n)*400) for n in G.nodes()], 
                               node_color=range(len(G.nodes())), cmap=plt.cm.Reds, edgecolors='black')
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
        plt.title(f"Elite Leadership Map: {topic} ({location})", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.savefig("top_10_elite_map.png", bbox_inches='tight')
        plt.close()

# --- RUN ---
engine = SciNexusUniversalAuditor(email="youremail@example.com") # EMAIL IS REQUOIRED FOR PUBMED LOGIN
engine.run_discovery(topic="KRAS", location="Ireland")

