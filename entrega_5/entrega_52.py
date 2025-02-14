import networkx as nx
import json
import community as community_louvain  

###################################Leitura de arquivo#####################################
with open("posts_data.json", "r") as file:
    dataset = json.load(file)

###################################Grafo de autores e tópicos########################################
key_words = [ 'canada', 'facebook', 'instagram','eua', 'usa','Elon Musk', 'joebiden','Kamala Harris',
              'Lindsey Graham','greenland','island','acquire','purchase','buy','military','control',
              'markzuckerberg','zuckerberg','trump','Obama','corrupt','biased', 'lies','racist','fake']

author_topic_map = {}

for entry in dataset:
    author = entry['comments_auth']
    words_tokens_comments = entry['comments'].split()
    related_keywords = [word.lower() for word in words_tokens_comments if word.lower() in key_words]
    
    if related_keywords:
        if author not in author_topic_map:
            author_topic_map[author] = set()
        author_topic_map[author].update(related_keywords)

G = nx.Graph()

for author in author_topic_map.keys():
    G.add_node(author)

authors = list(author_topic_map.keys())

for i in range(len(authors)):
    for j in range(i + 1, len(authors)):
        common_topics = author_topic_map[authors[i]].intersection(author_topic_map[authors[j]])
        if common_topics:  # Se há interseção de tópicos, cria conexão
            G.add_edge(authors[i], authors[j], weight=len(common_topics))
            
# Detecção das comunidades no grafo
partition = community_louvain.best_partition(G)

# Salvando o grafo em um arquivo .graph
nx.write_graphml(G, "grafo_autores.graphml")
