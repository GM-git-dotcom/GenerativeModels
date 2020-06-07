#LDA
'''This code is an adapted version of Shashank Kapadia's LDA model on the same NIPS dataset. Link to his profile: https://github.com/kapadias'''
import pandas as pd
import os
import re
from wordcloud import WordCloud
import warnings
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
from sklearn.decomposition import LatentDirichletAllocation as LDA
###############################################################################################################

os.chdir(r'C:\Users\KIIT\AppData\Local\Programs\Python\Python36\Scripts')
papers = pd.read_csv('NIPS Papers/papers.csv')
#Removing meta columns
papers = papers.drop(columns = ['id', 'event_type', 'pdf_name'], axis=1)

#Remove punctuation
papers['paper_text_processed'] = papers['paper_text'].map(lambda x: re.sub('[,/.!?]', '', x))

#Convert the titles to lowercase
papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: x.lower())


#Join the different processed titles together
long_string = ','.join(list(papers['paper_text_processed'].values))

###############################################################################################################

#Create a WordCloud object
wc = WordCloud(background_color = "white", max_words = 1000, contour_width = 3, contour_color='steelblue')

#Generate a word cloud
wc.generate(long_string)

#Visualize the word cloud
wc.to_image() #This line will create a word cloud at the output of the IDE used.

###############################################################################################################

#Load the library with the CountVectorizer method

#helper function
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse = True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))
    
    plt.figure(2, figsize = (15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='autumn_r')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
    
#Initialize the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words = 'english')

#Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(papers['paper_text_processed'])

#Visualize the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer) #Generates a pyplot bar graph

###############################################################################################################

#Now we tweak a number of topic parameters
warnings.simplefilter("ignore", DeprecationWarning)

#Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #{}:".format(topic_idx))
        print(" ".join([words[i] for i in topic.argsort()[:-n_top_words -1:-1]]))

#Tweak the two parameters below (use int values below 15)
number_topics = 5
number_words = 10

#Create and fit the LDA model from sklearn as imported on line 13
lda = LDA(n_components=number_topics)
lda.fit(count_data)

#Print the topics found by the LDA model
print("Topics found via LDA: ")
print_topics(lda, count_vectorizer, number_words)

###############################################################################################################
#Optional section, for better visualization and interpretation of results.
%%time
from pyLDAvis import sklearn as sklearn_lda
import pickle
import pyLDAvis

#Visualize the topics
pyLDAvis.enable_notebook()

LDAvis_data_filepath = os.path.join('./ldavis_prepared_'+str(number_topics))
# # This is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)
    
    with open(LDAvis_data_filepath, 'w') as f:
        pickle.dump(LDAvis_prepared, f)
#Load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath) as f:
    LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared, './ldavis_prepared_' + str(number_topics) + '.html')
LDAvis_prepared

###############################################################################################################
