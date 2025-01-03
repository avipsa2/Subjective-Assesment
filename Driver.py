# Importing libraries
import subprocess
subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_trf'])
import pandas as pd
import re
import nltk
import math
import pickle
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.corpus import wordnet
import yake
import FinalScoringDNN
import generategraphs
from keybert import KeyBERT
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk import ne_chunk
import numpy as np
import torch
import spacy
from numpy.linalg import norm
from nltk.tokenize import sent_tokenize
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from sentence_transformers import SentenceTransformer
from happytransformer import HappyTextToText, TTSettings
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import sklearn.metrics
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow.keras import initializers
import matplotlib.pyplot as plt

# Importing dataset to the workplace
# Importing stsb-en benchmarked dataset also to compare keyword scoring models
dataset = pd.read_excel("../data/FinalDatasetQA.xlsx")
keywordTestData = pd.read_csv("../data/stsb-en-data.csv", names = ['sentence1', 'sentence2', 'keyword_similarity_score'])

# Data Preprocessing
# Here we define our methods to clean the user answer and model answer
# Cleaning Contractions
R_patterns = [
    (r'won\'t', 'will not'),
    (r'can\'t', 'can not'),
    (r'(\w+)\'m', '\g<1> am'),
    (r'(\w+)\'ll', '\g<1> will'),
    (r'(\w+)\'d like to', '\g<1> would like to'),
    (r'(\w+)n\'t', '\g<1> not'),
    (r'(\w+)\'ve', '\g<1> have'),
    (r'(\w+)\'s', '\g<1> is'),
    (r'(\w+)\'re', '\g<1> are')
]

class REReplacer(object):
    def __init__(self, patterns = R_patterns):
        self.patterns = [(re.compile(regex), replace) for (regex, replace) in R_patterns]

    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        
        return s
    
# Lowering Text
def text_lower(text):
    return text.lower()

# Sentence Correction
def corrector(text):
    return text.replace('. ', '.')

# Cleaning Punctuations
def replace_punct(text):
    pattern = "[^\w\d\+\*\-\\\=\s]"
    replace = " "

    return re.sub(pattern, replace, text)

# Removing additional white spaces
def remove_extra(text):
    return " ".join(text.strip().split())

# Pre-processing the user and model answers
rep_word = REReplacer() # object to remove contractions

dataset['Model Answer'] = dataset['Model Answer'].apply(text_lower)
dataset['User Answer'] = dataset['User Answer'].apply(text_lower)
dataset['Model Answer'] = dataset['Model Answer'].apply(rep_word.replace)
dataset['User Answer'] = dataset['User Answer'].apply(rep_word.replace)

# The below preprocessing is only for ner and keyword scoring units, hence withdrawing the inputs for similarity scoring unit
mod_ans = dataset.iloc[:, 2].values # List of Model Answers
usr_ans = dataset.iloc[:, 3].values # List of User Answers
dataset['Model Answer'] = dataset['Model Answer'].apply(text_lower)
dataset['User Answer'] = dataset['User Answer'].apply(text_lower)
dataset['Model Answer'] = dataset['Model Answer'].apply(corrector)
dataset['User Answer'] = dataset['User Answer'].apply(corrector)
dataset['Model Answer'] = dataset['Model Answer'].apply(replace_punct)
dataset['User Answer'] = dataset['User Answer'].apply(replace_punct)
dataset['Model Answer'] = dataset['Model Answer'].apply(remove_extra)
dataset['User Answer'] = dataset['User Answer'].apply(remove_extra)

def normalize(x):
    return x/5

# KeyWord Scoring Module
# Adding the YAKE model to include keyword scoring
def yake_keywordExtractor(text):
    language = "en"
    max_ngram_size = 1
    deduplication_threshold = 0.9
    deduplication_algo = "seqm"
    windowSize = 2
    numOfKeywords = 101

    # Finetuning the yake model to perform the intended operations
    # Generates the extractor as per hypertuned parameters
    custom_kw_extractor = yake.KeywordExtractor(lan = language, 
                                            n = max_ngram_size, 
                                            dedupLim = deduplication_threshold, 
                                            dedupFunc = deduplication_algo, 
                                            windowsSize = windowSize, 
                                            top = numOfKeywords, 
                                            features = None)

    return custom_kw_extractor.extract_keywords(text)

# Loading the KeyBERT models for comparisions
def keybert_keywordExtractor(text):
    kw_model = KeyBERT()
    return kw_model.extract_keywords(text)

# Adding Keyword Scoring Unit
def keyword_scoring(keywords_text1, keywords_text2):
    match = 0
    total = 0
    synonym_dict = [] # List to include synonym matching

    for token in keywords_text1:
        total += token[1] # Total number of keywords in model answer
    if total == 0:
        return 10 # No keywords in model answer
    # Generating Synonyms for second text
    for var in keywords_text2:
        syn = wordnet.synsets(var[0]) # Synonym Dictionary
        syn_words = [x.lemma_names() for x in syn]
        syn_words = [x for elem in syn_words for x in elem]
        syn_words.append(var[0])
        syn_words = list(set(syn_words))
        temp = []
        wt = word_tokenize(var[0])
        pos = pos_tag(wt)[0][1]
        for i in range(0, len(syn_words)):
            checker_wt = word_tokenize(syn_words[i])
            checker_pos = pos_tag(wt)[0][1]
            if pos == checker_pos:
                temp.append(syn_words[i])
        synonym_dict = synonym_dict + temp # Enhancing the synonym dict for direct match compatibility 
    
    # Calculating the total number of matching keywords
    for token in keywords_text1:
        syn = wordnet.synsets(token[0])
        syn_words = [x.lemma_names() for x in syn]
        syn_words = [x for elem in syn_words for x in elem]
        syn_words.append(token[0])
        syn_words = list(set(syn_words))
        if len(set(syn_words).intersection(set(synonym_dict))) != 0:
            match += token[1]
    # Keyword score is number of matching keywords over total number of model keywords (normalized to 10)
    return match * 10 / total

keywordTestData['keyword_similarity_score'] = keywordTestData['keyword_similarity_score'].apply(normalize)
keywordTestData['sentence1'] = keywordTestData['sentence1'].apply(text_lower)
keywordTestData['sentence2'] = keywordTestData['sentence2'].apply(text_lower)
keywordTestData['sentence1'] = keywordTestData['sentence1'].apply(rep_word.replace)
keywordTestData['sentence2'] = keywordTestData['sentence2'].apply(rep_word.replace)
keywordTestData['sentence1'] = keywordTestData['sentence1'].apply(replace_punct)
keywordTestData['sentence2'] = keywordTestData['sentence2'].apply(replace_punct)
keywordTestData['sentence1'] = keywordTestData['sentence1'].apply(remove_extra)
keywordTestData['sentence2'] = keywordTestData['sentence2'].apply(remove_extra)

keywordTestData['keybert_keys_sentence1'] = keywordTestData['sentence1'].apply(keybert_keywordExtractor)
keywordTestData['keybert_keys_sentence2'] = keywordTestData['sentence2'].apply(keybert_keywordExtractor)
keywordTestData['yake_keys_sentence1'] = keywordTestData['sentence1'].apply(yake_keywordExtractor)
keywordTestData['yake_keys_sentence2'] = keywordTestData['sentence2'].apply(yake_keywordExtractor)

keywordTestData['keybert_score'] = keywordTestData[['keybert_keys_sentence1',
                                              'keybert_keys_sentence2']].apply(lambda x:keyword_scoring(x.keybert_keys_sentence1, 
                                                              x.keybert_keys_sentence2), axis=1)
keywordTestData['yake_score'] = keywordTestData[['yake_keys_sentence1',
                                              'yake_keys_sentence2']].apply(lambda x:keyword_scoring(x.yake_keys_sentence1, 
                                                                                                     x.yake_keys_sentence2), axis=1)
rmse_keybert = math.sqrt(sklearn.metrics.mean_squared_error(keywordTestData['keyword_similarity_score'], (keywordTestData['keybert_score']/10)))  
print("The error using keybert values", rmse_keybert) 

rmse_yake = math.sqrt(sklearn.metrics.mean_squared_error(keywordTestData['keyword_similarity_score'], (keywordTestData['yake_score']/10)))   
print("The error using yake values", rmse_yake)      

x = np.arange(1, 21)
y1 = keywordTestData['yake_score'][:20]  
y2 = keywordTestData['keybert_score'][:20]  
y3 = keywordTestData['keyword_similarity_score'][:20]   

plt.stackplot(
    x, 
    y1/10, 
    y2/10, 
    y3,
    labels=['Yake Score', 'KeyBert Score', 'Actual Score']    
)

plt.legend(loc="upper left")
plt.xlabel("Distribution of Indexes")
plt.ylabel("Stacked Scores")
plt.title("Keyword Scores and affection with Actual Score")

plt.show()
plt.savefig('../results/keywordModule.png',dpi=300)       
plt.clf()                                                                                         
                                                                                                     
# Similarity Scoring Module
# Adding Similarity Scoring Unit
# Define Cosine Similarity
def cos_sim(sent1_emb, sent2_emb):
    cos = np.dot(sent1_emb, sent2_emb)/(norm(sent1_emb)*norm(sent2_emb))
    return cos

# Similarity Scoring
def similarity_scoring(text1, text2, model):
    mod_sent = sent_tokenize(text1) # Individual sentences in model answer
    usr_sent = sent_tokenize(text2) # Individual sentences in user answer
    mod_emb = [] # Incorporating Model sentences embeddings
    usr_emb = [] # Incorporating User sentences embeddings
    for sent in mod_sent:
        sent_emb = model.encode(sent)
        mod_emb.append(sent_emb)
    for sent in usr_sent:
        sent_emb = model.encode(sent)
        usr_emb.append(sent_emb)
    n = len(mod_sent)
    m = len(usr_sent)
    match = 0
    for i in range (0, n):
        for j in range (0, m):
            if cos_sim(mod_emb[i], usr_emb[j]) >= 0.75:
                # Defining cosine threshold at 0.75
                match += 1
                break
    return match / n

# Adding Summarizer
# creating a t5 summarizer model
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu') # Can replace to gpu for faster processing

def summarizer(text):
    preprocessed_text = text.strip().replace('\n', '')
    t5_input_text = 'summarize: ' + preprocessed_text
    tokenized_text = tokenizer.encode(t5_input_text, return_tensors='pt', max_length=512).to(device)
    summary_ids = model.generate(tokenized_text, min_length=30, max_length=120)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Adding Grammar Corrector
def grammar_corrector(text):
    happy_tt = HappyTextToText("T5", "prithivida/grammar_error_correcter_v1") # Loading the corrector library
    text = "gec: " + text # Necessary pre-processing
    settings = TTSettings(do_sample=True, top_k=10, temperature=0.5, min_length=1, max_length=100)
    result = happy_tt.generate_text(text, args=settings)
    return result.text

simcse = SentenceTransformer('princeton-nlp/sup-simcse-roberta-large') # SimCSE Model
sbert = SentenceTransformer('stsb-mpnet-base-v2')
hf = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

num_of_inputs = len(mod_ans) # Total number of Answers
sim_score = [0] * num_of_inputs # List to keep respective similarity score
sbert_score = [0] * num_of_inputs
hf_score = [0] * num_of_inputs

# Similarity matrix in use for the calculation
for i in range(0, num_of_inputs):
    curr_mod = mod_ans[i]
    curr_usr = usr_ans[i]
    cur_usr = grammar_corrector(curr_usr) # grammar correction on user answer
    curr_usr += summarizer(curr_usr) # adding summary of user answer to user answer
    sim_val = similarity_scoring(curr_mod, curr_usr, simcse)
    sim_score[i] = sim_val * 10
    sbert_val = similarity_scoring(curr_mod, curr_usr, sbert)
    sbert_score[i] = sbert_val * 10
    hf_val = similarity_scoring(curr_mod, curr_usr, hf)
    hf_score[i] = hf_val * 10

# Converting list to numpy array
sim_score = np.array(sim_score)
sbert_score = np.array(sbert_score)
hf_score = np.array(hf_score)

print("Similarity/Semantic scores are generated Successfully!!!")


        

# Addition of the fields to calculate the yake scores
dataset['yake_keys_sentence1'] = dataset['Model Answer'].apply(yake_keywordExtractor)
dataset['yake_keys_sentence2'] = dataset['User Answer'].apply(yake_keywordExtractor)

# Addition of the fields to calculate the keybert scores
dataset['keybert_keys_sentence1'] = dataset['Model Answer'].apply(keybert_keywordExtractor)
dataset['keybert_keys_sentence2'] = dataset['User Answer'].apply(keybert_keywordExtractor)

# Adding the Keyword Score in dataset
dataset['yake_score'] = dataset[['yake_keys_sentence1', 'yake_keys_sentence2']].apply(lambda x : keyword_scoring(x.yake_keys_sentence1,
                                                                                        x.yake_keys_sentence2),
                                                                                        axis = 1)

dataset['keybert_score'] = dataset[['keybert_keys_sentence1', 'keybert_keys_sentence2']].apply(lambda x : keyword_scoring(x.keybert_keys_sentence1,
                                                                                        x.keybert_keys_sentence2),
                                                                                        axis = 1)

print("Keyword scores are generated Successfully!!!")
# Dropping the additional, not functional fields
dataset.drop(['yake_keys_sentence1', 'yake_keys_sentence2'], axis = 1, inplace = True)
dataset.drop(['keybert_keys_sentence1', 'keybert_keys_sentence2'], axis = 1, inplace = True)

# Adding the Similarity Score in dataset
dataset['simcse_score'] = sim_score.tolist()
dataset['sbert_score'] = sbert_score.tolist()
dataset['hf_score'] = hf_score.tolist()

# Named-Entity Recognition Module
# Adding entities extraction unit

def ner_scoring(text1, text2):
    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
    model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")
    nlp = pipeline('ner', 
                   model=model, 
                   tokenizer=tokenizer, 
                   aggregation_strategy="simple") # Initializing the NLP pipeline
    mod = nlp(text1) # pre-processing model answer
    usr = nlp(text2) # pre-processing user answer
    mod_arr = []
    usr_arr = []
    for i in range(0, len(mod)):
        if(mod[i]['score'] >= 0.9):
            mod_arr.append(mod[i]['word'].lower())
    if len(mod_arr) == 0:
        return 10 # return full marks if no entities in model answer
    for i in range(0, len(usr)):
        usr_arr.append(usr[i]['word'].lower())
    mod_arr = set(mod_arr)
    usr_arr = set(usr_arr)
    return len(mod_arr.intersection(usr_arr)) * 10 / len(mod_arr)


nlp = spacy.load("en_core_web_trf")
n = dataset.shape[0]

def spacy_name(text):
    doc = nlp(text)
    named_entities = []
    for entity in doc.ents:
        named_entities.append((entity.text, entity.label_))
    return named_entities

def nltk_name(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    final_pos1 =[]
    for i in pos_tags:
        if(i[1]=='NNP' or i[1]=='NNS' or i[1]=='NN'):
            final_pos1.append(i)   
    named_entities = ne_chunk(final_pos1)
    leaves = named_entities.leaves()
    return leaves

def compute_score_nltk(a, b):
    usr_arr = []
    mod_arr = []
    for i in range(0, len(a)):
        usr_arr.append(a[i][0].lower())
    for i in range(0, len(b)):
        mod_arr.append(b[i][0].lower())
    entity_usr = list(set(usr_arr))
    entity_mod = list(set(mod_arr))
    n = len(entity_usr)
    m = len(entity_mod)
    count = 0
    if(n == 0):
        return 10
    for i in range(m):
        if(entity_mod[i] in entity_usr):
            count += 1
    count = count / n * 10
    return count

def compute_score_spacy(a, b):
    usr_arr = []
    mod_arr = []
    for i in range(0, len(a)):
        if(a[i][1] != 'CARDINAL'):
            usr_arr.append(a[i][0].lower())
    for i in range(0, len(b)):
        if(b[i][1] != 'CARDINAL'):
            mod_arr.append(b[i][0].lower())
    a = list(set(usr_arr))
    b = list(set(mod_arr))
    n = len(b)
    m = len(a)
    count = 0 
    if(n == 0):
        return 10
    for i in range(n):
        if(b[i] in a):
            count += 1

    count = count / n * 10
    return count

def regex(text):
    text2 = ''
    for i in text:
        if ((ord(i) >= 65 and ord(i) <= 90) or (ord(i) >= 97 and ord(i) <= 122) or ord(i)==32):
            text2 += i
    text2 = " ".join(text2.strip().split())
    text2 = text2.lower()
    return text2

# Computing and adding the NER Score in dataset
cam_score = [0] * num_of_inputs
nltk_score = [0] * num_of_inputs
spacy_score = [0] * num_of_inputs

for i in range(dataset.shape[0]):
    mod = dataset['Model Answer'][i] # Individual model answers
    usr = dataset['User Answer'][i] # Individual user answers
    cam_score[i] = ner_scoring(mod, usr)
    mod1 = regex(mod)
    usr1 = regex(usr)
    mod_ent_nltk = nltk_name(mod1)
    usr_ent_nltk = nltk_name(usr1)
    nltk_score[i] = compute_score_nltk(usr_ent_nltk, mod_ent_nltk)
    mod_ent_spcy = spacy_name(mod)
    usr_ent_spcy = spacy_name(usr)
    spacy_score[i] = compute_score_spacy(usr_ent_spcy, mod_ent_spcy)
    
cam_score = np.array(cam_score) # List to numpy array conversion
nltk_score = np.array(nltk_score)
spacy_score = np.array(spacy_score)
dataset['cam_score'] = cam_score.tolist()
dataset['nltk_score'] = nltk_score.tolist()
dataset['spacy_score'] = spacy_score.tolist()

print("NER scores are generated Successfully!!!")

# Saving the dataset in form of a csv file to be in neural network
dataset.to_csv('../results/FinalScoringDataset.csv', index=False)

# Calling final DNN generator
FinalScoringDNN.generate_DNNmodel()

# Graph generator
generategraphs.generate_graphical_inference()