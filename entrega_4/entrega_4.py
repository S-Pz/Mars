import nltk, json, re, gensim
import fasttext.util
import pandas as pd

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
              'markzuckerberg','zuckerberg','trump','Obama'
            ]

negative_words = [
    "corrupt", "unfair", "biased", "lies", "dishonest", "scam", "fraud","hate", "terrible", "awful",
    "disgusting", "nonsense", "rigged","manipulated", "fake", "wrong", "unacceptable", "shameful", 
    "problematic","disgrace", "toxic", "incompetent", "hypocrite", "crooked", "failing","horrible",
    "evil","selfish", "ruined", "destroyed", "overrated", "controversial", "dangerous", "threat",
    "misleading", "cheated","broken", "backward", "divisive", "outdated", "pathetic", "ignorant", 
    "arrogant", "oppressive", "exploitative", "despicable", "racist","crazy",'fat','ugly','stupid',
    'dumb','idiot','fool','foolish','dull','dullard',
]

filtred_sentences = []

for entry in dataset:

    words_tokens_comments = word_tokenize(entry['comments'])
    
    filtered_words = [
        w.lower() for w in words_tokens_comments 
        if w.lower() not in stop_words and re.match(r'^[a-zA-Z]+$', w)
    ]
    if filtered_words: 
        filtred_sentences.append(filtered_words)

print(filtred_sentences)
#################################### Word2Vec #####################################################
model = Word2Vec(filtred_sentences, vector_size=100, window=10, min_count=1, workers=4, sg=1)

valid_key_words = [word for word in key_words if word in model.wv]
valid_negative_words = [word for word in negative_words if word in model.wv]

if valid_key_words:
    similar_words = model.wv.most_similar(positive=valid_key_words, negative=valid_negative_words)
    relation = [(item[0], round(item[1], 2)) for item in similar_words]
    print (relation)
    #colocar um para relacionar com todos tipo o item 0 com o 1,2,3,4,5,6,7,8,9 e assim por diante
    print("\n🔹 Palavras mais associadas às Key Words:")
    for word, score in relation:
        print(f"{word}: {score}")

    for word, score in relation:
        related_sentences = [entry['comments'] for entry in dataset if word in entry['comments']]
        combined_text = " ".join(related_sentences)
        sentiment_score = analyzer.polarity_scores(combined_text)['compound']
        sentiment = "Positivo" if sentiment_score > 0 else "Negativo" if sentiment_score < 0 else "Neutro"
        
        print(f"Palavra: {word} - Score: {sentiment_score} ({sentiment})")