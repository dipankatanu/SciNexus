#!/usr/bin/env python
# coding: utf-8

# In[18]:


from Bio import Entrez
import pandas as pd
import plotly.express as px
import networkx as nx
from pyvis.network import Network
from datetime import datetime, timedelta
import re
import time
import os

class SciNexusEngine:
    def __init__(self, email):
        Entrez.email = email
        self.output_dir = "outputs"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def run_pipeline(self, 
                     broad_term="KRAS", 
                     location="Ireland", 
                     years=10, 
                     max_recs=1000, 
                     strict_mode=True, 
                     semantic_cluster=None):
        """
        Main pipeline execution.
        :param broad_term: The main topic (e.g., 'Autophagy' or 'KRAS')
        :param location: Filter by country (e.g., 'Ireland'). Use "" for Global.
        :param years: Timeframe for research.
        :param max_recs: Maximum papers to scan.
        :param strict_mode: Boolean. If True, applies semantic scoring.
        :param semantic_cluster: List of specific technical keywords.
        """
        
        # 1. INITIAL SEARCH
        date_cutoff = (datetime.now() - timedelta(days=years * 365)).strftime("%Y/%m/%d")
        loc_query = f" AND ({location}[Affiliation])" if location else ""
        query = f"({broad_term}){loc_query} AND ({date_cutoff}[PDAT] : 2026[PDAT])"
        
        print(f"\n--- Initiating SciNexus Scan ---")
        print(f"Topic: {broad_term} | Scope: {'Global' if not location else location} | Mode: {'Strict' if strict_mode else 'Broad'}")
        
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_recs, usehistory="y")
        id_list = Entrez.read(handle).get("IdList", [])
        handle.close()
        
        if not id_list:
            print("No records found.")
            return

        # 2. PROCESSING
        results, edges = [], []
        # Fallback cluster if none provided
        if semantic_cluster is None:
            semantic_cluster = [broad_term]
        
        combined_pattern = re.compile("|".join(semantic_cluster), re.IGNORECASE)
        
        print(f"Found {len(id_list)} papers. Analyzing metadata...")
        
        batch_size = 250
        for start in range(0, len(id_list), batch_size):
            batch = id_list[start:start+batch_size]
            try:
                handle = Entrez.efetch(db="pubmed", id=",".join(batch), retmode="xml")
                records = Entrez.read(handle)
                handle.close()
                
                for art in records.get('PubmedArticle', []):
                    article = art['MedlineCitation']['Article']
                    title = article['ArticleTitle']
                    
                    # Semantic Scoring Logic
                    if strict_mode:
                        abstract_data = article.get('Abstract', {}).get('AbstractText', [''])
                        abstract = " ".join([str(t) for t in abstract_data])
                        is_match = 1 if combined_pattern.search(title + " " + abstract) else 0
                    else:
                        is_match = 1 # Every paper is a match in Broad Mode

                    authors = article.get('AuthorList', [])
                    paper_team = []
                    
                    for author in authors:
                        last_name = author.get('LastName')
                        fore_name = author.get('ForeName', author.get('Initials', ""))
                        
                        if last_name:
                            name = f"{last_name}, {fore_name}".strip()
                        elif 'CollectiveName' in author:
                            name = author['CollectiveName']
                        else: continue

                        affils = author.get('AffiliationInfo', [])
                        if affils:
                            affil_text = affils[0]['Affiliation']
                            if not location or re.search(location, affil_text, re.IGNORECASE):
                                paper_team.append(name)
                                results.append({
                                    "Researcher": name, 
                                    "Institution": affil_text.split(',')[0],
                                    "Expert_Score": is_match
                                })
                    
                    if len(paper_team) > 1:
                        for i in range(len(paper_team)):
                            for j in range(i + 1, len(paper_team)):
                                edges.append((paper_team[i], paper_team[j]))
                
                print(f"Batch {start//batch_size + 1} complete...")
            except Exception as e:
                print(f"Batch failed: {e}")
                continue

        df = pd.DataFrame(results)
        self._generate_visuals(df, edges, broad_term, location, strict_mode)

    def _generate_visuals(self, df, edges, topic, loc, mode):
        """Internal method to generate Leaderboard and Advanced Nexus Map."""
        
        # A. LEADERBOARD
        summary = df.groupby(['Researcher', 'Institution']).agg({'Expert_Score':'sum', 'Researcher':'count'}).rename(columns={'Researcher':'Total_Papers'}).reset_index()
        summary = summary[summary['Total_Papers'] > 1].sort_values('Expert_Score', ascending=False)
        
        fig = px.bar(summary.head(15), x=['Expert_Score', 'Total_Papers'], y='Researcher', 
                     barmode='group', title=f"SciNexus Leaderboard: {topic} ({'Strict' if mode else 'Broad'})",
                     color_discrete_map={'Expert_Score': '#e74c3c', 'Total_Papers': '#34495e'})
        fig.write_html(f"{self.output_dir}/leaderboard.html")
        
        # B. INTERACTIVE NEXUS MAP
        net = Network(height="850px", width="100%", bgcolor="#ffffff", font_color="black")
        G = nx.Graph()
        G.add_edges_from(edges)
        
        if G.nodes:
            communities = nx.community.louvain_communities(G)
            community_map = {node: i for i, comm in enumerate(communities) for node in comm}
            scores = df.groupby('Researcher')['Expert_Score'].sum().to_dict()

            for node in G.nodes:
                s = scores.get(node, 0)
                cid = community_map.get(node, 0)
                colors = ["#3498db", "#e74c3c", "#2ecc71", "#f1c40f", "#9b59b6", "#34495e"]
                net.add_node(node, label=node, size=15+(s*4), color=colors[cid % len(colors)], title=f"Score: {s}")

            for src, tgt in G.edges:
                net.add_edge(src, tgt, color="#bdc3c7", alpha=0.3)

            net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)
            net.show_buttons(filter_=['physics'])
            net.save_graph(f"{self.output_dir}/SciNexus_Interactive_Map.html")
        
        df.to_csv(f"{self.output_dir}/scinexus_raw_data.csv", index=False)
        print(f"\nSuccess! Check the /{self.output_dir} folder.")

# =============================================================================
# CHOOSE YOUR EXECUTION ARGUMENTS HERE
# =============================================================================

engine = SciNexusEngine(email="ENTER_YOUR_EMAIL_HERE") # REQUIRED TO FETCH DATA FROM NCBS

# EXAMPLE 1: Broad Search (Autophagy - No Specific Cluster Needed)
# engine.run_pipeline(broad_term="Autophagy", location="India", years=5, strict_mode=False)

# EXAMPLE 2: Strict Search (KRAS Case Study with Specific Cluster)
kras_keywords = [r"KRAS", r"RAS signaling", r"feedback loops", r"mathematical model", r"systems biology"]
engine.run_pipeline(
    broad_term="KRAS", 
    location="Ireland", 
    years=10, 
    strict_mode=True, 
    semantic_cluster=kras_keywords
)


# In[ ]:




