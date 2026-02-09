'''
Project done in the framework of the Statistical Counsulting course for the SWDE (La Société Walonne Des Eaux) 
during the academic year 2023-2024.
Consist of :
    - cleaning of the text
    - Stemming
    - Tokenization
    - Clustering using Hierarchical Clustering
'''

#%% Imports
#--- Package
import seaborn as sns
import pandas as pd
import random
import numpy as np
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
import nltk
import re
import scipy.cluster.hierarchy as shc
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans, AgglomerativeClustering 
import matplotlib.pyplot as plt
from copy import deepcopy
from prince import MCA
import altair as alt
from nltk.stem import PorterStemmer
import spacy
'''
to run to install the french stemmer

import spacy.cli
spacy.cli.download("fr_core_news_sm")
import spacy

# Load the French model
nlp = spacy.load("fr_core_news_sm")

'''
#%%--- Data and functions
crash = pd.read_excel("C:/Users/User/Documents/Data Science/LSTAT2380 Statistical Consulting/Project_SWDE/Data/Crash4.xlsx")
gloss = pd.read_excel("C:/Users/User/Documents/Data Science/LSTAT2380 Statistical Consulting/Project_SWDE/Data/Glossaire.xlsx")
comm_belge = pd.read_excel("C:/Users/User/Documents/Data Science/LSTAT2380 Statistical Consulting/Project_SWDE/Data/comm_belge2.xlsx")
cities = pd.read_csv('C:/Users/User/Documents/Data Science/LSTAT2380 Statistical Consulting/Project_SWDE/Data/worldcities.csv')
cities2 = pd.read_excel('C:/Users/User/Documents/Data Science/LSTAT2380 Statistical Consulting/Project_SWDE/Data/sectoren.xlsx')
nltk.download('stopwords')
# Download and load the French model from spacy
spacy.cli.download("fr_core_news_sm")
nlp = spacy.load('fr_core_news_sm')
                 
cities = cities[cities['country'] == 'Belgium']
cities2 = set(cities2['TX_DESCR_FR'])
cities_to_rem = list(cities['city'])

# From old cluster 5, who had all the city names
to_supp = ['petit', 'vieux', 'viesville', 'biesmerée', 'vezin','new', 'vert', 'thibessart', 'izier', 'pont', 'couillet', 
                 'bray', 'hauts', 'nord', 'hydro', 'samrée', 'farnieres', 'bringuette', 'baty', 'yves', 'waillimont', 'offaing', 'wibrin', 'mortehan', 'ville', 'hourt', 
                 'houvegnez', 'borlon', 'dairomont', 'forges', 'ver', 'champagne', 'beaulieusart', 'ellignies', 'ennal', 'chanxhe', 'anseremme', 'barsy',                  
                 'orroir', 'payenne', 'trixheuf', 'our', 'remience', 'capt', 'luzery', 'lez', 'renard', 'trivières', 'autelhaut', 'gileppe', 'montzen', 'gossoin'
                 , 'floriheid', 'compl', 'provedroux', 'sarts', 'taintignies', 'péronnes', 'guerlange', 'ansuelle', 'aout', 'antheit', 'denis', 'marc',                 
                 'mandelavaux', 'pommeroeul', 'bouge', 'braine', 'croix', 'plantation', 'aspèches', 'tcheresse', 'anne', 'lisogne', 'petitvoir', 'evelette', 'tete', 'goe',
                  'beole', 'champsia', 'evrard', 'massul', 'combes', 'tamizon', 'solre', 'ferrieres', 'barrage', 'pannerie',                  
                 'fontenoille', 'barry', 'marbaix', 'niersant', 'ladrerie', 'dupuis', 'waha', 'horward', 'borlez', 'noduwez'                 
                 , 'lincé', 'arimont', 'rouvreux', 'warnoumont', 'velaine', 'arbrefontaine', 'nouv', 'hautregard', 'enclenchement', 
                 'chapelle', 'bellevue', 'faubourg', 'dep', 'espinette', 'baneux', 'meuse', 'biesme', 'comte', 'defoy', 'coo', 'vissoule', 'anc', 'hargimont', 'sauveniere'                 
                 , 'christophe', 'bêche', 'wépion', 'diabolo', 'materne', 'justice', 'waffe', 'crupet', 'gozée', 'charreau', 'lesse', 'blaregnies',                 
                 'hockai', 'fire', 'onoz', 'hordenne', 'gertrude', 'gilly', 'robertville', 'wiheries', 'wespin', 'niveze', 'horrues', 'forchies', 'claminforge',                  
                 'erquennes', 'lince', 'rhisnes','schaltin', 'naninne', 'brimez', 'cedrogne', 'jaumaux', 'gomezée', 'bronromme', 'waret', 'alouette'                 
                 , 'salm', 'stambruge', 'piersay', 'follie', 'belle', 'hirtzenberg', 'blocqmont', 'débimetre', 'sorinnes', 'longuefeu', 'heze', 'gofe', 'moha', 'spy',                 
                 'erbisoeul', 'chalet', 'bonlez', 'fontaine', 'biester', 'fourbechies', 'bagatelle', 'lustin', 'hestre', 'famenne',                  
                 'bonneville', 'milmort', 'jacques', 'gomzée', 'roy', 'homby', 'rouges', 'laguespré', 'tribomont', 'loyers', 'forzee', 'campinaire', 'rochettes',                  
                 'werbomont', 'preaix', 'jevigne', 'peissant', 'aieg', 'meslin', 'oignies', 'diables', 'houtain', 'medard', 'tiege', 'géry',                 
                  'xhoris', 'andrimont', 'boussi', 'wardin', 'morialmé', 'dampremy', 'quesval',                 
                 'brye', 'marchienne', 'gottignies', 'lesves', 'eneilles', 'néchin', 'fize', 'evrehailles', 'lambusart', 'gouy', 'quevaucamps',                  
                 'maur', 'navinne', 'carnières', 'couthuin', 'lesve', 'slins', 'saintes', 'nvelle', 'pry', 'werimont',                  
                 'wimboru', 'jacquet','flaches', 'gomezee', 'sainlez', 'joncs', 'donat', 'bayemont', 'bovenistier', 'haillot', 'wihéries',             
                 'fontenaille', 'erpent', 'fromiée', 'heron', 'heppignies', 'bise', 'feuillien', 'focroule', 'charmes', 'dorinne', 'peruwelz',                 
                 'binches', 'wasmes', 'fosteau', 'leers', 'citadelle', 'monceau', 'ophain', 'bonnet', 'ransart', 'laid', 'patard', 'piéton', 'conc', 'chêne', 
                 'ronveau', 'champion', 'gay', 'lodelinsart',"proximus"]

#-------- USED
#--- Function to clean text with stemming
def clean_text2(text):
    #--- LIST abbreviations
    abrev = list(gloss["Abréviation"])
    #--- THINGS TO REMOVE
    stop_words = set(stopwords.words('french'))
    #--- remove abbreviation from stop words
    stop_words.difference_update(['d', 's', 'ce'])
    rem = [".", ",", ":", ";", "!", "?", '-', '\ ', "'", '"', "(", ")", "[", "]", "-", "on", "by",
           "mobile", "to", "set", "user", "started", "completed", "received", "created", "from", 'for', 'hold', 'operation', 'status', 'reason']
    time_related = ["janvier", "février", "fevrier", "mars", "avril", "mai", "juin", "juillet", "août", "aout", "septembre", "octobre", "novembre", "décembre", "decembre",
                    "mois", "semaine", "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

    # add to the remove list all the words to remove
    commune_name = list(comm_belge['name'])
    commune_name = [word.lower() for word in commune_name]
    c_to_rem = [word.lower() for word in cities_to_rem]
    rem.extend(commune_name)
    rem.extend(time_related)
    rem.extend(c_to_rem)
    rem.extend(to_supp)
    rem.extend(cities2)

    #--- Outputed List with cleaned sentences
    final = []
    sent_count = 0
    for sent in text:
        doc = nlp(sent)
        words = []
        pattern = r'^[A-Za-z]+\d+$' # maybe could suppress this
        # pattern = r'([A-Za-z]+|\d+)' # modified version to handle kind "SEP_SP"
        for token in doc:
            w = token.text
            #--- Check for stop_words
            if w.lower() not in stop_words:
                if re.match(pattern, w):
                    letters = re.findall("[A-Za-z]+", w)[0]
                    digits = re.findall('[0-9]+', w)[0]

                    #--- Check for abbreviation
                    if letters in abrev:
                        index = abrev.index(letters)
                        letters = gloss.loc[index, "Libellé"]

                    words.append(str(letters))
                    words.append(str(digits))

                #--- Check for abbreviation again
                elif w in abrev:
                    index = abrev.index(w)
                    words.append(gloss.loc[index, "Libellé"])

                #--- Check for removable character and if word is alphanumeric
                elif w.lower() not in rem and w.isalnum():
                    # Apply lemmatization
                    words.append(token.lemma_.lower())

        #--- Add the sentence to the final list 
        if words and any(char.isalpha() for char in ' '.join(words)): # non empty
            final.append(' '.join(words))
        else:
            print(f'Problem with sentence n {sent_count}', "\n", '-'*40)
        sent_count += 1

    return final

#--- tokenize a text
def do_token(text):
    stop_words = set(stopwords.words('french'))
    tok_data = []
    sent_count = 0
    for sent in text:
        temp = []
        for word in word_tokenize(sent):
            if not re.search(r'\d',word) and word.lower() not in stop_words: #no digits and no stop_words (le, la, les,au) allowed  
                temp.append(word.lower())
        if temp: #non empty
            tok_data.append(temp)
        else:
            print(f'Problem with sentence n {sent_count}')
        sent_count += 1
        
    return  tok_data

#--- getting the optimal number of clusters
def opt_clust(DS):
    dist = []
    K_range = range(1,15)
    for k in K_range :
        kmeans = KMeans(init = 'k-means++',
                    n_clusters = k,
                    n_init = 20,
                    max_iter = 300,
                    random_state=4243)
        kmeans.fit(DS)
        dist.append(kmeans.inertia_)
    #plotting
    plt.plot(K_range, dist, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()

def cluster_words(emb, num_clusters):
    '''
    made by chat GPT
    '''
    # Get the embedded word vectors
    word_vectors = emb.wv.vectors
    # Normalize the word vectors
    word_vectors_normalized = word_vectors / np.linalg.norm(word_vectors)
    # Perform K-means clustering
    kmeans = KMeans(num_clusters, random_state=42, init='k-means++', n_init=20, max_iter=300)
    kmeans.fit(word_vectors_normalized)
    # Get cluster centers and labels
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    # Analyze the resulting clusters
    clusters = {}
    for i in range(num_clusters):
        cluster_words = [emb.wv.index_to_key[j] for j in range(len(cluster_labels)) if cluster_labels[j] == i]
        clusters[f"Cluster {i + 1}"] = cluster_words
    return clusters,kmeans

#--- Function that returns
def KM_class_(embedding,kmeans):
    '''
    Function that returns the closest cluster per sentence
    
    embedding : representation in space of my words
    kmeans : KMeans() type of object
    
    return: a list with the class corresponding to each sentence
    '''
    sentence_clusters = []
    count_sent = 0
    # Iterate through each sentence in the tokenized variable
    for sentence in tokenized:
        #print(f'Iter for sentence numb {count_sent} \n ---------------------------------------------')
        # Initialize an empty array to store word vectors of the sentence
        sentence_word_vectors = []
        
        # Iterate through each word in the sentence
        for word in sentence:
            # Check if the word exists in the vocabulary of embedding
            #print(word)
            if word in embedding.wv:
                #print('YES')
                # Append the embedded word vector to the list
                sentence_word_vectors.append(embedding.wv[word])
            #print('-------')
        
        # Calculate the average word vector for the sentence
        if sentence_word_vectors:
            avg_word_vector = np.mean(sentence_word_vectors, axis=0)
        else:
            print('ELSE LOOP')
            # If no word vectors found for the sentence, use zeros
            avg_word_vector = np.zeros(embedding.vector_size)
        
        # Reshape the average word vector to match the input format expected by kmeans2.predict
        avg_word_vector = avg_word_vector.reshape(1, -1)
        
        # Predict the cluster label for the sentence using kmeans2
        cluster_label = kmeans.predict(avg_word_vector)[0]
        
        # Append the cluster label to the list
        sentence_clusters.append(cluster_label)
        
        count_sent += 1
        
    return sentence_clusters

#--- one hot encoding
def one_hot_encode(df):
    # Identify cat columns
    cat_columns = df.select_dtypes(include=['category']).columns

    # Apply one-hot encoding to string columns
    df_encoded = pd.get_dummies(df, columns=cat_columns, drop_first=False, dtype= float)

    return df_encoded

    
#%% creating an interisting subset
#rename for easier use
crash.rename(columns = {'Désignation - Crash - Long':'Des_long',
                        'TexteCodeProbl.':'Code_Probl',
                        'TexteGrpeCodes':'Groupe','Nbre Heures Travail':'Volume_trav',
                        'Date début plf':'DD_plf',
                        'Désignation - Crash':'Des',
                        'Date début réel':"DD_Real",
                        'Bilan période 2020 (m² annuel)':"Bilan_20",
                        "Bilan période 2021 (m² annuel)":'Bilan_21',
                        "ID Poste technique":"ID_PT",
                        "Type de travail":"Type_work",
                        "Lancement réel":"DD_start"}, inplace= True)

sub = deepcopy(crash[['Des_long',"Code_Probl","Groupe","Des"]])

sub['Groupe'] = sub['Groupe'].astype('category')
sub['Groupe'].value_counts().sum()
#suppress the 1 time occuring class
sub['Groupe'].value_counts()


(sub['Groupe'].value_counts().sum()/len(sub))*100 #% of labeled description
(sub['Des_long'].value_counts().sum()/len(sub))*100 # 99% of description
(sub['Des'].value_counts().sum()/len(sub))*100 # 99 of short description
#%%  WORD2VEC and KMEANS

#--- Tokenization
all_text = sub['Des_long'][sub['Des_long'].notna()]
all_text = np.array(all_text)
cleaned = clean_text2(all_text)
#--- prob new embed
prob1 = [592,1202,3808,4859,13934,16435,20519,21290,21615,22315,25693,25708,25709]
tokenized = do_token(cleaned)
print(len(cleaned) == len(tokenized))

#--- Create emb

emb1 = Word2Vec(tokenized, min_count = 1,vector_size = 100, window = 3, seed = 4243, workers = 1)
emb2 = Word2Vec(tokenized, min_count = 1,vector_size = 100, window = 3, sg = 1, seed = 4243, workers = 1) #skip gram

##### ONLY emb1 so far
#--- Get the embedded word vectors
word_vector1 = emb1.wv.vectors
word_vector2 = emb2.wv.vectors

#--- Normalize them
word_vector1 = word_vector1/np.linalg.norm(word_vector1)
word_vector2 = word_vector2/np.linalg.norm(word_vector2)



#%% KMEANS PLOTS
#--- opt clusters elbow technique
opt_clust(word_vector1)# between 6 and 10


opt_clust(word_vector2) #between 6 and 10 (the greater the better in term of invertia)
# emb2 has less inertia

#%% KMeans assigning words
#--- fitting
#--- assign a cluster class to each of my sentence by chatGPT
num_clusters1 = 8
num_clusters2 = 8 

clust_mod1, kmeans1 = cluster_words(emb1,num_clusters1)
clust_mod2, kmeans2 = cluster_words(emb2,num_clusters2)    
# %% Visualize words per clusters    
for i in range(num_clusters2):
    print(f"Cluster {i + 1} - emb 2:", print('Length : ',len(clust_mod2[f"Cluster {i + 1}"])),clust_mod2[f"Cluster {i + 1}"])

len(clust_mod2[0])
#%% Class assigning

# Initialize a list to store cluster memberships for each sentence


sentence_clusters = []
count_sent = 0
# Iterate through each sentence in the tokenized variable
for sentence in tokenized:
    #print(f'Iter for sentence numb {count_sent} \n ---------------------------------------------')
    # Initialize an empty array to store word vectors of the sentence
    sentence_word_vectors = []
    
    # Iterate through each word in the sentence
    for word in sentence:
        # Check if the word exists in the vocabulary of emb2
        #print(word)
        if word in emb2.wv:
            #print('YES')
            # Append the embedded word vector to the list
            sentence_word_vectors.append(emb2.wv[word])
        #print('-------')
    
    # Calculate the average word vector for the sentence
    if sentence_word_vectors:
        avg_word_vector = np.mean(sentence_word_vectors, axis=0)
    else:
        print('ELSE LOOP')
        # If no word vectors found for the sentence, use zeros
        avg_word_vector = np.zeros(emb2.vector_size)
    
    # Reshape the average word vector to match the input format expected by kmeans2.predict
    avg_word_vector = avg_word_vector.reshape(1, -1)
    
    # Predict the cluster label for the sentence using kmeans2
    cluster_label = kmeans2.predict(avg_word_vector)[0]
    
    # Append the cluster label to the list
    sentence_clusters.append(cluster_label)
    
    count_sent += 1

#--- Check is length is correct
len(sentence_clusters) == len(tokenized) ==len(cleaned)

DS = pd.DataFrame(cleaned,columns=['cleaned'])
DS['class2'] = sentence_clusters
DS['class2'] = DS['class2'].astype("category")
print(DS['class2'].value_counts())

#--- Assess the class but with embedding 1
sentence_clusters1 = KM_class_(emb1, kmeans1)
len(sentence_clusters1) == len(tokenized) ==len(cleaned)

DS['class1'] = sentence_clusters1
DS['class1'] = DS['class1'].astype("category")
DS['class1'].value_counts() # worse of the 3

#%% HIERARCHICAL CLUSTERING

#--- Finding optimal number of cluster using dendrograms

#dendro = shc.dendrogram(shc.linkage(word_vector2, method="ward"))

plt.title("Dendrogrma Plot")  
plt.ylabel("Euclidean Distances")  
plt.xlabel("Customers")  
plt.show()  

#---  Training Hierarchical model
random.seed(4243)
n_clusters  = 10 # maybe try 6
hc= AgglomerativeClustering(n_clusters= n_clusters, metric='euclidean', linkage='ward')  
hc.fit(word_vector2)

#--- Visualize words per clusters by GPT

# Initialize an empty dictionary to store words for each cluster
HC_clust_words = {}

# Iterate through each word index and its corresponding cluster label
for i, label in enumerate(hc.labels_):
    word_vector = emb2.wv.vectors[i]  # Get the word vector
    word = emb2.wv.index_to_key[i]  # Get the corresponding word
    # Check if the cluster label already exists in the dictionary
    if label in HC_clust_words:
        # If yes, append the word to the list of words for that cluster
        HC_clust_words[label].append(word)
    else:
        # If no, create a new list with the word as its first element
        HC_clust_words[label] = [word]

# Print the words in each cluster
for cluster_label, words in HC_clust_words.items():
    break
    print(f"Cluster {cluster_label + 1} words: Length : {len(words)}", words)
    
print(HC_clust_words[0])
   
#--- Get Clusters centers

clusters_center = {}
for c in range(n_clusters):
    embed_w = []
    for word in HC_clust_words[c]:
        embed_w.append(emb2.wv.get_vector(word))
    avg_clust = np.mean(embed_w, axis = 0) # think should be by columns => axis = 1
    clusters_center[c]= avg_clust



#--- Assign each sentence to a class

#- get avg embedded sentence
HC_class = []
for sentence in tokenized:
    
    # Initialize an empty array to store word vectors of the sentence
    sentence_word_vectors = []
    
    # Iterate through each word in the sentence
    for word in sentence:
        # Check if the word exists in the vocabulary of emb2
        
        if word in emb2.wv:
            # Append the embedded word vector to the list
            sentence_word_vectors.append(emb2.wv[word])
    
    # Calculate the average word vector for the sentence
    if sentence_word_vectors:
        avg_sent = np.mean(sentence_word_vectors, axis=0)
    
    #- Get the closest cluster to each sentence
    dist = []
    for c in range(n_clusters):
        dist.append(np.linalg.norm(clusters_center[c] - avg_sent))
    
    classe = np.argmin(dist)
    HC_class.append(classe)
    

DS['class3'] = HC_class
DS['class3'] = DS['class3'].astype("category")
print(DS['class3'].value_counts()) #more balanced

del(i,label,c,word) #,dendro)

#--- Get the 10 closest words by center
closest_words_per_cluster = {}

# Loop through each cluster
for c in range(n_clusters):
    word_distances = []
    
    # Calculate the distance of each word's vector to the cluster center
    for word in HC_clust_words[c]:
        # Compute Euclidean distance
        dist = np.linalg.norm(emb2.wv.get_vector(word) - clusters_center[c])
        # Append the word and its distance
        word_distances.append((word, dist))
    
    # Sort words by distance (smallest first)
    word_distances.sort(key=lambda x: x[1])
    
    # Take the top 10 words and store them
    closest_words_per_cluster[c] = [wd[0] for wd in word_distances[:10]]

# Printing the closest 10 words for each cluster
#for cluster_label, words in closest_words_per_cluster.items():
#    print(f"Cluster {cluster_label + 1} closest words:", words)

#--- Get 10 most occuring words by clusters
from collections import Counter
top_words_per_cluster = {}

# Loop through each cluster
for c in range(n_clusters):
    # Combine all words from all sentences in the cluster
    all_cluster_words = [word for word in HC_clust_words[c]]
    
    # Count the frequency of each word
    word_freq = Counter(all_cluster_words)
    
    # Sort words by frequency (most frequent first)
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Take the top 10 words and store them
    top_words_per_cluster[c] = [word[0] for word in sorted_words[:10]]

# Printing the top 10 words for each cluster
#for cluster_label, words in top_words_per_cluster.items():
#    print(f"Cluster {cluster_label + 1} top words by frequency:", words)



#%% ANALYZE CLUSTERS

#class 0
print(HC_clust_words[0][:15])
# 1
print(HC_clust_words[1][:15])

# 2
print(HC_clust_words[2][:15])

# 3
print(HC_clust_words[3][:15])

# 4
print(HC_clust_words[4][:15])

# 5
print(HC_clust_words[5][:15])

# 6
print(HC_clust_words[6][:15])

# 7
print(HC_clust_words[7][:15])

# 9
print(HC_clust_words[9][:15])
#%% CONSTRUCT FINAL DATASETS

#--- Text index that has been removed from OG DS
removed = [592,1202,3808,4859,13934,16435,20519,21290,21615,22315,25693,25708,25709]
fullDS = deepcopy(crash[crash['Des_long'].notna()])

fullDS.drop(index = removed, inplace = True)
len(fullDS) == len(DS)

#--- Replace Des_long by cleaned and add class
fullDS['Des_long'] = cleaned
fullDS.rename(columns = {'Des_long':'Cleaned'}, inplace = True)
textDS = fullDS['Cleaned']

fullDS['Class'] = HC_class
fullDS['Class'].value_counts()

#--- Remove useless columns
(fullDS['Code_Probl'].value_counts().sum()/len(fullDS))*100 #19 %
(fullDS['Age_Batiment'].value_counts().sum()/len(fullDS))*100 #7%
(fullDS['Bilan_20'].value_counts().sum()/len(fullDS))*100 #38%
(fullDS['Bilan_21'].value_counts().sum()/len(fullDS))*100 #38 %

for i in fullDS.columns:
    print(f' Column : {i} \t Prop non miss {round((fullDS[i].value_counts().sum()/len(fullDS))*100,2)}')
    
fullDS.drop(columns = ['Des','Groupe',"Code_Probl","Age_Batiment","Bilan_21","Bilan_20","Volume cuve (m³)","Statut util.","Priorité - Traduction de colonne Q","ID_PT"],inplace = True)
fullDS.dropna(inplace = True)

#--- Set binary variable
fullDS.dtypes
to_bin = ['PRISE_EAU',"HAUTE_TENSION","STOKAGE_EAU","TRAITEMENT","EEM"]
fullDS[to_bin]= fullDS[to_bin].replace("X",1)


#--- Keep temporal information about the start of intervention
fullDS['Year_st'] = fullDS['DD_start'].dt.year
fullDS['Month_st'] = fullDS['DD_start'].dt.month

fullDS.drop(columns = ['DD_Real','DD_plf','DD_start'], inplace = True)

#--- Set categorical variables
to_cat = ['Type_work','Equipe','Div. planif.','Localité',"Commune","FctCalculee","Class","Year_st","Month_st"]
fullDS[to_cat] = fullDS[to_cat].astype("category")
fullDS.dtypes
fullDS['Class'].value_counts()
#%%--- Save my dataset

fullDS.to_csv('C:/Users/User/Documents/Data Science/LSTAT2380 Statistical Consulting/Project_SWDE/Data/finalDS_stemmed.csv')
fullDS = fullDS.reset_index(drop= True)

#%% Analyze my dataset
#--- Load it
fullDS = pd.read_csv('C:/Users/User/Documents/Data Science/LSTAT2380 Statistical Consulting/Project_SWDE/Data/finalDS_stemmed.csv', index_col = 0)
to_cat = ['Type_work','Equipe','Div. planif.','Localité',"Commune","FctCalculee","Class","Year_st","Month_st","PRISE_EAU","HAUTE_TENSION","STOKAGE_EAU","TRAITEMENT","EEM"]
fullDS[to_cat] = fullDS[to_cat].astype("category")

fullDS.dtypes

#--- Classes
fullDS['Class'].nunique()
class_dist = round(((pd.DataFrame(fullDS['Class'].value_counts()))/len(fullDS))*100,2)

plt.bar(class_dist['Classes'], class_dist['count'],color = 'black')
plt.xlabel('Classes')
plt.ylabel('Percentage')
plt.title("Proportion of Classes")

plt.show()


class_dist['Classes'] = class_dist.index
fullDS['Class'].nunique()
round((len(DS)/len(crash))*100,2)
#--- Analyze it 
num_col = fullDS.select_dtypes(include=['int64','float64']).columns  
cat_col = fullDS.select_dtypes(include=['category']).columns


#--- Locality
fullDS['Localité'].nunique() 
fullDS['Commune'].nunique()
fullDS['Commune'].value_counts()

fullDS['Div. planif.'].value_counts()
fullDS['Div. planif.'].nunique()


plt.hist(fullDS['Localité'].value_counts())
plt.show()


#--- Commune
plt.hist(fullDS['Commune'].value_counts())
plt.show()

#---- Classes
plt.hist(fullDS['Class'].value_counts())
plt.show()
#%% Some plots (useless so far)
#--- Heatmap

CM = fullDS[num_col].corr()

# Generate a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(CM, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Plot of Variables')
plt.show()

#--- MCA

alt.data_transformers.enable("vegafusion")
alt.renderers.enable('browser')
mca = MCA( n_components = 3,
          random_state= 4243,
          one_hot= True)

mca_fitted = mca.fit(fullDS[cat_col])

mca_fitted.eigenvalues_summary #2 % of the variance explained...

#--- Plotting
chart = mca_fitted.plot(
    fullDS[cat_col],
    x_component=0,
    y_component=1,
    show_column_markers=True,
    show_row_labels=False,
    show_column_labels=False
)
chart

#--- Contingency table
from scipy.stats import chi2_contingency

CT = pd.crosstab(fullDS['Class'], fullDS['Commune'])
print(CT)
chi2, p, dof, expected = chi2_contingency(CT)
# Print the test statistic and p-value
print("Chi-square statistic:", chi2)
print("p-value:", p)


#%% Dataset Pascale
num_to_classe = {
    0: "Puits et Station de pompage",
    1: "Inclassable",
    2: "Interventions Techniques",
    3: "Inclassable",
    4: "Maintenance et Chloration",
    5: "Remplacement et Réparations",
    6: "Pompage, Vanne et Entretien",
    7: "Gestion du Travail, OT",
    8: "Pannes et Connexions",
    9: "Réservoirs et Fuites"
}

for i in range(len(fullDS)):
    fullDS['Class_lab'] = fullDS.loc[i,"Class"]
