import nltk, json, re, gensim
import fasttext.util
import pandas as pd
import networkx as nx
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models import Word2Vec

####################### Pré-processamento ##########################
# fasttext.util.download_model('en', if_exists='ignore')  # English
# ft = fasttext.load_model('cc.en.300.bin')

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('vader_lexicon')

analyzer = SentimentIntensityAnalyzer()

with open("posts_data.json", "r") as file:
    dataset = json.load(file)

stop_words = set(stopwords.words('english'))

key_words = [ 'canada', 'facebook', 'instagram','eua', 'usa','Elon Musk', 'joebiden','Kamala Harris',
              'Lindsey Graham','greenland','island','acquire','purchase','buy','military','control',
              'markzuckerberg','zuckerberg','trump','Obama',"corrupt","biased", "lies","racist","fake"
            ]

negative_words = [
    "corrupt", "unfair", "biased", "lies", "dishonest", "scam", "fraud","hate", "terrible", "awful",
    "disgusting", "nonsense", "rigged","manipulated", "fake", "wrong", "unacceptable", "shameful", 
    "problematic","disgrace", "toxic", "incompetent", "hypocrite", "crooked", "failing","horrible",
    "evil","selfish", "ruined", "destroyed", "overrated", "controversial", "dangerous", "threat",
    "misleading", "cheated","broken", "backward", "divisive", "outdated", "pathetic", "ignorant", 
    "arrogant", "oppressive", "exploitative", "despicable", "racist","crazy",'fat','ugly','stupid',
    'dumb','idiot','fool','foolish','dull','dullard'
]

author_topic_map = {}

filtred_sentences = []

for entry in dataset:
    author = entry['comments_auth']

    words_tokens_comments = word_tokenize(entry['comments'])
    
    filtered_words = [
        w.lower() for w in words_tokens_comments 
        if w.lower() not in stop_words and re.match(r'^[a-zA-Z]+$', w)
    ]
    
    related_keywords = [word for word in filtered_words if word in key_words]

    if filtered_words: 
        filtred_sentences.append(filtered_words)
    
    if related_keywords:
        if author not in author_topic_map:
            author_topic_map[author] = set()
        author_topic_map[author].update(related_keywords)

print(author_topic_map)
#################################### Word2Vec #####################################################
all_similar_words = []  # Lista para armazenar todas as palavras similares

for i, filtred_list in enumerate(filtred_sentences):
    model = Word2Vec([filtred_list], vector_size=100, window=10, min_count=1, workers=4, sg=1)  # Treina um modelo para cada conjunto de frases
    valid_key_words = [word for word in key_words if word in model.wv]
    valid_negative_words = [word for word in negative_words if word in model.wv]

    if valid_key_words:
        try:
            similar_words = model.wv.most_similar(positive=valid_key_words, negative=valid_negative_words)
            relation = [(item[0], round(item[1], 2)) for item in similar_words]
            all_similar_words.extend(relation)  # Adiciona os resultados à lista geral

            print(f"\n🔹 Lista {i + 1} - Palavras mais associadas:")
            for word, score in relation:
                print(f"{word}: {score}")

        except KeyError:
            print(f"\n⚠️ Lista {i + 1}: Não há palavras suficientes para calcular similaridade.")
            continue  # Caso não tenha palavras suficientes, ignora e segue para a próxima lista

# Remove duplicatas e ordena pela pontuação de similaridade média
from collections import defaultdict

word_scores = defaultdict(list)
for word, score in all_similar_words:
    word_scores[word].append(score)

# Calcula a média da pontuação de similaridade para cada palavra
aggregated_scores = [(word, round(sum(scores) / len(scores), 2)) for word, scores in word_scores.items()]
aggregated_scores.sort(key=lambda x: x[1], reverse=True)  # Ordena do mais similar para o menos similar

# Exibe os resultados finais agregados
print("\n🔹 Palavras mais associadas ao conjunto completo:")

for word, score in aggregated_scores:
    print(f"{word}: {score}")

    related_sentences = [entry['comments'] for entry in dataset if word in entry['comments']]
    combined_text = " ".join(related_sentences)
    sentiment_score = analyzer.polarity_scores(combined_text)['compound']
    sentiment = "Positivo" if sentiment_score > 0 else "Negativo" if sentiment_score < 0 else "Neutro"

    print(f"Palavra: {word} - Score: {sentiment_score} ({sentiment})")

###############################################
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

# Criando um dicionário a partir dos comentários filtrados
dictionary = Dictionary(filtred_sentences)
corpus = [dictionary.doc2bow(text) for text in filtred_sentences]

# Treinando o modelo LDA para encontrar 5 tópicos
lda = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# Mostrando os tópicos identificados
topics = lda.print_topics(num_words=5)
for topic in topics:
    print(topic)

#################################################################
G = nx.Graph()

# Adiciona nós (autores)
for author in author_topic_map.keys():
    G.add_node(author)

# Adiciona arestas se os autores compartilham temas similares
authors = list(author_topic_map.keys())

for i in range(len(authors)):
    for j in range(i + 1, len(authors)):
        common_topics = author_topic_map[authors[i]].intersection(author_topic_map[authors[j]])
        if common_topics:  # Se há interseção de tópicos, cria conexão
            G.add_edge(authors[i], authors[j], weight=len(common_topics))

# Desenhando o grafo (opcional)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
plt.show()