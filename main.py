
# Group Members
# Sonu Kumar - 19k0169
# Sumeet Kumar - 19k0171


# Libraries
import os
import pickle
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from nltk.corpus import wordnet,stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler




def preprocess_text(text: str, remove_stopwords: bool) -> str:
    lemmatizer = WordNetLemmatizer()

    # remove links
    text = re.sub(r"http\S+", "", text)
    # remove numbers and special chars
    text = re.sub("[^A-Za-z]+", " ", text)
    # remove stopwords
    if remove_stopwords:
        # 1. creates tokens
        tokens = nltk.word_tokenize(text)
        # 2. checks if token is a stopword and removes it
        updated_tokens = []
        for i in range(len(tokens)):
            if tokens[i].lower() in stopwords.words("english"):
                continue
            else:
                # Apply lemmitizer to the token
                updated_tokens.append(lemmatizer.lemmatize(tokens[i].lower()))

        # 4. joins all tokens again
        # text = " ".join(updated_tokens)


    # returns cleaned text
    # text = text.lower().strip()
    return updated_tokens

def buildRelation(nouns):
    relation_list = defaultdict(list)

    for k in range(len(nouns)):
        relation = []
        for syn in wordnet.synsets(nouns[k], pos=wordnet.NOUN):
            for l in syn.lemmas():
                relation.append(l.name())
                if l.antonyms():
                    relation.append(l.antonyms()[0].name())
            for l in syn.hyponyms():
                if l.hyponyms():
                    relation.append(l.hyponyms()[0].name().split('.')[0])
            for l in syn.hypernyms():
                if l.hypernyms():
                    relation.append(l.hypernyms()[0].name().split('.')[0])
        relation_list[nouns[k]].append(relation)
    return relation_list

def buildLexicalChain(nouns, relation_list):
    lexical = []
    threshold = 0.5
    for noun in nouns:
        flag = 0
        for j in range(len(lexical)):
            if flag == 0:
                for key in list(lexical[j]):
                    if key == noun and flag == 0:
                        lexical[j][noun] += 1
                        flag = 1
                    elif key in relation_list[noun][0] and flag == 0:
                        syns1 = wordnet.synsets(key, pos=wordnet.NOUN)
                        syns2 = wordnet.synsets(noun, pos=wordnet.NOUN)
                        if syns1[0].wup_similarity(syns2[0]) >= threshold:
                            lexical[j][noun] = 1
                            flag = 1
                    elif noun in relation_list[key][0] and flag == 0:
                        syns1 = wordnet.synsets(key, pos=wordnet.NOUN)
                        syns2 = wordnet.synsets(noun, pos=wordnet.NOUN)
                        if syns1[0].wup_similarity(syns2[0]) >= threshold:
                            lexical[j][noun] = 1
                            flag = 1
        if flag == 0:
            dic_nuevo = {}
            dic_nuevo[noun] = 1
            lexical.append(dic_nuevo)
            flag = 1
    return lexical


def eliminateWords(lexical):
    final_chain = []
    while lexical:
        result = lexical.pop()
        if len(result.keys()) == 1:
            for value in result.values():
                if value != 1:
                    final_chain.append(result)
        else:
            final_chain.append(result)
    return final_chain

def getallFiles():
    file_paths = os.listdir(r'C:\Users\sonuk\PycharmProjects\IrProject\Doc50\\')
    return file_paths

def extractDataFromFiles():

    files = getallFiles()
    dataset = ""

    for i in files:
        path = r'C:\Users\sonuk\PycharmProjects\IrProject\Doc50'
        path = path + '\\' + i
        f = open(path,'r')
        dataset = preprocess_text(f.read(), remove_stopwords=True)
        # use lexical chains as the feature selection method
        nouns = []
        l = nltk.pos_tag(dataset)
        for word, n in l:
            if n == 'NN' or n == 'NNS' or n == 'NNP' or n == 'NNPS':
                nouns.append(word)

        relation = buildRelation(nouns)
        lexical = buildLexicalChain(nouns, relation)
        final_chain = eliminateWords(lexical)
        storeDocFeatures(i,final_chain)


def storeDocFeatures(filename,doc_dict):
    file_path = r"C:\Users\sonuk\PycharmProjects\IrProject\docFeatures\\" + filename + ".pkl"
    file = open(file_path, "wb")
    pickle.dump(doc_dict, file)
    file.close()


def buildVsmForDocuments():

    files = getallFiles()
    totalFeatures = []

    for i in files:
        a_file = open(r"C:\Users\sonuk\PycharmProjects\IrProject\docFeatures\\" + i + ".pkl", "rb")
        docFeaturesdict = pickle.load(a_file)
        for features in docFeaturesdict:
            for docFeature in features.keys():
                if docFeature not in totalFeatures:
                    totalFeatures.append(docFeature)


    print(totalFeatures)
    print(len(totalFeatures))

    final_training_Features = []

    for i in files:
        a_file = open(r"C:\Users\sonuk\PycharmProjects\IrProject\docFeatures\\" + i + ".pkl", "rb")
        docFeaturesdict = pickle.load(a_file)
        temp = []
        for j in totalFeatures:
            check = False
            for features in docFeaturesdict:
                if j in features.keys():
                    temp.append(features[j])
                    check = True
                    break
            if not check:
                temp.append(0)

        final_training_Features.append(temp)

    return final_training_Features

def storeDocumentVectors(documentVectors):
    file_path = r"C:\Users\sonuk\PycharmProjects\IrProject\docFeatures\documentVectors.pkl"
    file = open(file_path, "wb")
    pickle.dump(documentVectors, file)
    file.close()


def readDocumentVectors():
    file_path = r"C:\Users\sonuk\PycharmProjects\IrProject\docFeatures\documentVectors.pkl"
    a_file = open(file_path,"rb")
    X = pickle.load(a_file)
    a_file.close()
    return X

def kMeansClustering(X,maxClusters):

    print("----------------Selecting efficient K By Using Elbow Method-----------------")
    print()
    Sum_of_squared_distances = []

    pca = PCA(n_components=2, random_state=42)
    # pass X to the pca
    pca_vecs = pca.fit_transform(X)


    K = range(2,maxClusters)
    for k in K:
       km = KMeans(n_clusters=k, init='k-means++',max_iter=200, n_init=10)
       km = km.fit(X)
       labels = km.labels_
       Sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()



    # # initialize KMeans with 3 clusters
    K = 4
    print("Number of clusters = " + str(K))
    kmeans = KMeans(n_clusters=K, init='k-means++', random_state=42)
    y = kmeans.fit_predict(pca_vecs)
    clusters = kmeans.labels_


    filtered_label0 = pca_vecs[y == 0]
    filtered_label1 = pca_vecs[y == 1]
    filtered_label2 = pca_vecs[y == 2]
    filtered_label3 = pca_vecs[y == 3]

    # Plotting the results
    plt.scatter(filtered_label0[:, 0], filtered_label0[:, 1],color = 'purple')
    plt.scatter(filtered_label1[:, 0], filtered_label1[:, 1],color = 'blue')
    plt.scatter(filtered_label2[:, 0], filtered_label2[:, 1], color='red')
    plt.scatter(filtered_label3[:, 0], filtered_label3[:, 1], color='black')
    plt.show()


    # Getting the Centroids
    centroids = kmeans.cluster_centers_
    u_labels = np.unique(y)

    # plotting the results:

    for i in u_labels:
        plt.scatter(pca_vecs[y == i, 0], pca_vecs[y == i, 1], label=i)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
    plt.legend()
    plt.show()


    WCSS = kmeans.inertia_
    labels_pred = kmeans.labels_
    print ()
    print ("K-means labels:\n")
    print (kmeans.labels_ )
    print("\nWithin-Cluster Sum-of-Squares: " + str(WCSS))
    silhouette = metrics.silhouette_score(pca_vecs, labels_pred, metric='euclidean')
    print("Silhouette Coefficient: " + str(silhouette))


def kmeansBestClustering(data, max_clusters, scaling=True, visualization=True):
    n_clusters_list = []
    silhouette_list = []
    print()
    print()
    print("----------------Selecting efficient K By Using Best SilhouetteScore Method-----------------")
    print()
    if scaling:
        # Data Scaling
        scaler = MinMaxScaler()
        data_std = scaler.fit_transform(data)
    else:
        data_std = data

    for n_c in range(2, max_clusters + 1):
        kmeans_model = KMeans(n_clusters=n_c, random_state=42).fit(data_std)
        labels = kmeans_model.labels_
        n_clusters_list.append(n_c)
        silhouette_list.append(silhouette_score(data_std, labels, metric='euclidean'))

    # Best Parameters
    param1 = n_clusters_list[np.argmax(silhouette_list)]
    param2 = max(silhouette_list)
    best_params = param1, param2

    # Data labeling with the best model
    kmeans_best = KMeans(n_clusters=param1, random_state=42).fit(data_std)
    labels_best = kmeans_best.labels_
    labeled_data = np.concatenate((data, labels_best.reshape(-1, 1)), axis=1)

    if visualization:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.plot(n_clusters_list, silhouette_list, linewidth=3,
                label="Silhouette Score Against # of Clusters")
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Silhouette score")
        ax.set_title('Silhouette score according to number of clusters')
        ax.grid(True)
        plt.plot(param1, param2, "tomato", marker="*",
                 markersize=20, label='Best Silhouette Score')

        plt.legend(loc="best", fontsize='large')
        plt.show();
        print("Number of clusters = %i \nSilhouette_score = %.2f." % best_params)


    # store Document Features into particular document name file
#extractDataFromFiles()


# make vectors for each document
# X = buildVsmForDocuments()
# storeDocumentVectors(X)

# read Vectors for every document
X = readDocumentVectors()

# select K using elbow method used pca decomposition on the data
kMeansClustering(X,15)

# select K on basis of best silhouette_score used minmax scaling on the data
kmeansBestClustering(X,15,True,True)
