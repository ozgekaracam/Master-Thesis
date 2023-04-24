# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
#Import pandas
import pandas as pd
import time
import re
import string
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import wordnet
from collections import Counter
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from d3graph import d3graph

def visualize_graph(apriori_rules):
    # Initialize
    d3 = d3graph()
    d3.graph(apriori_rules['antecedents'], apriori_rules['consequents'], weight=apriori_rules['lift'])
    d3.set_edge_properties(directed=True, scaler='minmax')
    # d3.set_edge_properties() , edge_distance = 0
    #d3.set_node_properties(color=adjmat.columns.values)
    # Show
    d3.show()
# Generate rules
def generate_rules(frequent_itemsets):
    apriori_rules = association_rules(frequent_itemsets, metric='lift', min_threshold=20)
    apriori_rules.sort_values('confidence', ascending=False, inplace=True)
    apriori_rules.antecedents = apriori_rules.antecedents.apply(lambda x: next(iter(x)))
    apriori_rules.consequents = apriori_rules.consequents.apply(lambda x: next(iter(x)))
    return apriori_rules
# One-hot encoder
def encode_onehot(corpus_list):
    te = TransactionEncoder()
    te_ary = te.fit(corpus_list).transform(corpus_list)
    corpus_df = pd.DataFrame(te_ary, columns=te.columns_)
    return corpus_df
# Apriori algorithm
def apriori_algo(corpus_list):
    start_time = time.time()
    frequent_itemsets = apriori(encode_onehot(corpus_list), min_support = 0.003, use_colnames=True)
    print("---Runtime: %s seconds ---" % (time.time() - start_time))
    #API: apriori(df, min_support=0.5, use_colnames=False, max_len=None, verbose=0, low_memory=False)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    print("the number of frequent itemsets generated:", len(frequent_itemsets))
    return generate_rules(frequent_itemsets)
# Create corpus list
def create_corpus(dataset):
    df = dataset.query("aacat1 not in ['Noise', 'none']")
    # df['aacat1'].unique()
    corpus_list = df['clean_content'].tolist()
    return corpus_list
#Instantiate Stemmer
stemmer = PorterStemmer()
def word_stemmer(text):
    stem_text = [stemmer.stem(i) for i in text]
    return stem_text
def get_part_of_speech(word):
    probable_part_of_speech = wordnet.synsets(word)
    pos_counts = Counter()
    pos_counts["n"] = len([item for item in probable_part_of_speech if item.pos() == "n"])
    pos_counts["v"] = len([item for item in probable_part_of_speech if item.pos() == "v"])
    pos_counts["a"] = len([item for item in probable_part_of_speech if item.pos() == "a"])
    pos_counts["r"] = len([item for item in probable_part_of_speech if item.pos() == "r"])

    most_likely_part_of_speech = pos_counts.most_common(1)[0][0]
    return most_likely_part_of_speech

# Instantiate lemmatizer
lemmatizer = WordNetLemmatizer()
def word_lemmatizer(text):
    lem_text = [lemmatizer.lemmatize(i, get_part_of_speech(i)) for i in text]
    return lem_text

def removeNumber(text):
    return' '.join(re.sub(r'[0-9]',' ', text).split())
def deEmojify(text):
    return text.encode('ascii', 'ignore').decode('ascii')
def remove_stopwords(text):
    stpwrd = nltk.corpus.stopwords.words('english')
    text = text.split(" ")
    words = [w for w in text if w not in stpwrd]
    return ' '.join(words)
def removePunctuation(text):
    no_punc = "".join([c for c in text if c not in string.punctuation])
    return no_punc
def removeLink(text):
    no_link = ' '.join(re.sub("(w+://S+)", " ", text).split())
    return no_link
def preprocess(content): #res -> clean_content
    clean_content = content.lower()
    # removeLinks
    clean_content = removeLink(clean_content)
    # remove stop words
    clean_content = remove_stopwords(clean_content)
    # removePunc
    clean_content = removePunctuation(clean_content)
    # removeEmojis
    clean_content = deEmojify(clean_content)
    # removeNumber
    clean_content = removeNumber(clean_content)
    # tokenizer
    tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
    clean_content = tokenizer.tokenize(clean_content)
    # lemmatizer
    clean_content = word_lemmatizer(clean_content)
    # stemmer
    #clean_content = word_stemmer(clean_content)
    return clean_content
def prepare_data(dataset):
#prepare_data(df, stemmer='lan', spellcheck=False):
    start_time = time.time()
    dataset['clean_content'] = [preprocess(x) for x in dataset['content']]
    #if spellcheck: df.to_csv("/Users/neel/Desktop/bigsample_spellchecked.csv")
    print("--- %s seconds ---" % (time.time() - start_time))
    return dataset
def get_data(file):
    data = pd.read_csv(file)
    print(data.shape)
    return data
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file = "final_annotations.csv"
    # prepare the data
    print("preprocessing data...")
    dataset = prepare_data(get_data(file))
    print("done preprocessing data...")
    print("dataset length: ", len(dataset))
    apriori_rules = apriori_algo(create_corpus(dataset))
    visualize_graph(apriori_rules)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
