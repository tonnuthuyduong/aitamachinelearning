# Am I the A-hole? A moral classification 

r/AmItheAsshole is one of the most popular communities on Reddit. People use it to post their personal conundrums and moral conflicts and ask for judgements from strangers. The verdicts include NTA (Not the Asshole), YTA (You're the Asshole), ESH (Everyoe sucks here), and NAH (No Assholes here). This is a neat setup for a classification problem.
Using filtered data from r/AITAFiltered, I built a classifer to determine if you're the asshole in your story or not. 

## Getting data through Reddit API
<details><summary><u>Reddit Scrapper</u></summary>

<p>

```python

import requests
app_id = 'appid'
secret = 'secret'
auth = requests.auth.HTTPBasicAuth(app_id, secret)

headers = {'User-Agent': 'Tutorial2/0.0.1'}
res = requests.post('https://www.reddit.com/api/v1/access_token',
auth=auth, data=data, headers=headers)

token = res.json()['access_token']
headers['Authorization'] = 'bearer {}'.format(token)
requests.get('https://oauth.reddit.com/api/v1/me', headers=headers)

token = ""
api = 'https://oauth.reddit.com'
res = requests.get('{}/r/AITAFiltered/new'.format(api), headers=headers, params={'limit': '50000'})
result = res.json()

```
</p>
</details>

The result is saved as json file for further analysis
## Loading the data
<p>

```python
import json
with open('aita_top_posts.json') as f:
    json_data = f.read()
    aita = json.loads(json_data)
import pandas as pd
df = pd.DataFrame({'title':[],'text':[],'verdict' : []})
for i in range(2,len(aita)):
    try:
        df.loc[i,['title','text','verdict']]=  [aita[i]['title'],aita[i]['crosspost_parent_list'][0]['selftext'],
                                    aita[i]['crosspost_parent_list'][0]['link_flair_text']]
    except:
        pass 
df = df[df.text != '[deleted]']
df = df[df.verdict != 'Not enough info']
```
</p>

## Cleaning and pre-processing
Checking distribution across values

<p>

```python
import seaborn as sns
sns.countplot(x = 'verdict', data = df, palette = "blend:#7AB,#EDA")
```
</p>

As we can see, there is some slight imbalance between Not the A-hole and other categories. Furtheremore, we just want Asshole or not A-hole. So, I'm going to group Everyone sucks with Asshole (because the poster is also at fault here) and No A-holes here with Not the A-hole (beacause the poster is not at fault here).
<p>

```python
df.loc[df['verdict'] == 'Everyone Sucks', 'verdict'] = 'Asshole'
df.loc[df['verdict'] == 'No A-holes here', 'verdict'] = 'Not the A-hole'
sns.countplot(x = 'verdict', data = df, palette = "blend:#7AB,#EDA")
```
</p>
After grouping, we have a really nice distribution between two classes. 

### Text cleaning and Preprocessing

<p>

```python
import nltk
from gensim.parsing.preprocessing import remove_stopwords
from nltk.corpus import opinion_lexicon
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.sentiment import SentimentIntensityAnalyzer as SIA
from nltk.corpus import stopwords
stop = stopwords.words('english')
import nltk.sentiment
tokenizer = RegexpTokenizer(r'\w+')
```
</p>

So there are two texts that I will work on: one is the tile for topic modelling. The second is the text for classification.
I use title for topic modelling because the title is more condense and straight to the point what the post is about. When done on the selftext, topic modelling picks out lots of pronouns and generic verbs like make, do, take, etc. Furthermore, topic modelling done on title takes shorter time.

<p>

```python
df['title'] = df['title'].apply(lambda x: x.replace("'m"," am"))
df['title'] = df['title'].apply(lambda x: x.replace("n't"," not"))
df['title'] = df['title'].apply(lambda x: x.replace("'ve"," have"))
df['title'] = df['title'].apply(lambda x: x.replace("'s"," is"))
df['title'] = df['title'].apply(lambda x: x.replace("aita"," "))
df['title'] = df['title'].apply(lambda x: x.replace("'re"," are"))
df.title = df.title.apply(lambda x: x.replace("AITA"," "))
df.title = df.title.apply(lambda x: x.replace("WIBTA"," "))
df.title = df.title.apply(lambda x: x.replace("for",""))

```
</p>

<p>

```python
df.title  = df.title.apply(remove_stopwords)
df
tokenizer = RegexpTokenizer(r'\w+')
df['title_tokenized'] = df['title'].apply(lambda x: tokenizer.tokenize(x.lower()))
binarized_df = pd.get_dummies(df['verdict'])
df = pd.concat([df, binarized_df], axis=1)

```
</p>

<p>

```python

from sklearn.feature_extraction.text import CountVectorizer
transformer = CountVectorizer()
vect = transformer.fit(df['text'])
#print(len(vect.vocabulary_))
#print(len(vect.get_feature_names()))
#transform all dataset
data_transform = vect.transform(df['text'])
#apply tfidf
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_trans = TfidfTransformer()
tfit = tfidf_trans.fit(data_transform)
data_tfidf = tfit.transform(data_transform)
```
</p>

## Topic Modelling

<p>

```python
df['tokens'] = df['tokenized'].apply(lambda x: tokenizer.tokenize(x.lower()))
from gensim import corpora
from gensim import models

aitadict = corpora.Dictionary(df.title_tokenized.tolist())
aitacorp = [aitadict.doc2bow(x) for x in df.title_tokenized.tolist()]
import gensim
from gensim import models

aitamodel = gensim.models.ldamodel.LdaModel(corpus=aitacorp,id2word=aitadict,
num_topics=10, passes = 10)
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
visual =  gensimvis.prepare(aitamodel, aitacorp, aitadict)
pyLDAvis.enable_notebook()
pyLDAvis.display(visual)

```
</p>

## Classification

<p>

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from textblob import TextBlob
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
import re
```
</p>

<p>

```python

def text_preprocessor(text):
    return " ".join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?#", " ", text.lower()).split())

df['processed_text'] = df['text'].apply(lambda x: text_preprocessor(x))
from sklearn.model_selection import train_test_split
ASS_train, ASS_test, label_train, label_test = train_test_split(df['text'], df['Asshole'], test_size=0.2, random_state=1)
```
</p>

### Naive Bayes
#### MultinomialNB

<p>

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
pipelineMNB = Pipeline([
    ('vect', CountVectorizer(preprocessor=text_preprocessor)),
    ('tfidf',TfidfTransformer()),
    ('clf', MultinomialNB())
])

pipelineMNB.fit(ASS_train, label_train)
Make predictions on the test data
predictions = pipelineMNB.predict(ASS_test)
Print classification report
print(classification_report(label_test, predictions))
```
</p>

### Support Vector Machine Classification

<p>

```python
pipeline_svc = Pipeline([
    ('vect', CountVectorizer(preprocessor=text_preprocessor)),
    ('tfidf',TfidfTransformer()),
    ('clf', SVC(kernel='linear', C=2.0))
])


#Fit the pipeline to the training data
pipeline_svc.fit(ASS_train, label_train)

# Make predictions on the test data
predictions_svc = pipeline_svc.predict(ASS_test)
# Set the range of hyperparameter values to search over
param_grid = {
    'vect__max_df': [0.5, 0.75, 1.0],
    'clf__C': [0.1, 1, 10]
}

# Perform grid search cross-validation to find the best hyperparameters
grid_search = GridSearchCV(pipeline_svc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(ASS_train, label_train)

# Print the best hyperparameters and the corresponding accuracy score on the validation set
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Make predictions on the test set using the best hyperparameters
best_pipeline = grid_search.best_estimator_
predictions = best_pipeline.predict(ASS_test)
accuracy = accuracy_score(label_test, predictions)
print("Test accuracy:", accuracy)

# Define the pipeline
pipeline_svc = Pipeline([
    ('vect', CountVectorizer(preprocessor=text_preprocessor, max_df=0.5)),
    ('tfidf',TfidfTransformer()),
    ('clf', SVC(kernel='linear', C=1))
])

# Fit the pipeline to the ASS_training data
pipeline_svc.fit(ASS_train, label_train)

# Make predictions on the test data
predictions_svc = pipeline_svc.predict(ASS_test)

# Print confusion matrix
print(confusion_matrix(label_test, predictions_svc))

# Compute accuracy score
accuracy = accuracy_score(label_test, predictions_svc)
print("Test accuracy:", accuracy)

```
</p>

### Random Forest
Set the range of hyperparameter values to search over

<p>

```python
pipeline_rf = Pipeline([
    ('vect', CountVectorizer(preprocessor=text_preprocessor,ngram_range=(1, 2))),
    ('tfidf',TfidfTransformer()),
    ('clf', RandomForestClassifier(max_depth=990))
])
param_grid = {
    'clf__max_depth': range(990,1000)
}

# Perform grid search cross-validation to find the best hyperparameters
grid_search = GridSearchCV(pipeline_rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(ASS_train, label_train)

# Print the best hyperparameters and the corresponding accuracy score on the validation set
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
# Random forrest Classifier with hyperparameter
pipeline_rf = Pipeline([
    ('vect', CountVectorizer(preprocessor=text_preprocessor,ngram_range=(1, 2))),
    ('tfidf',TfidfTransformer()),
    ('clf', RandomForestClassifier(max_depth=990))
])
# Fit the pipeline to the ASS_training data
pipeline_rf.fit(ASS_train, label_train)

# Make predictions on the test data
predictions_rf = pipeline_rf.predict(ASS_test)
# Print accuracy score
print("Accuracy:", accuracy_score(label_test, predictions_rf))
```
</p>

#### Logistic Regression
<p>

```python
pipeline_l = Pipeline([
    ('vect', CountVectorizer(preprocessor=text_preprocessor)),
    ('tfidf',TfidfTransformer()),
    ('clf', LogisticRegression())
])

# Fit the pipeline to the ASS_training data
pipeline_l.fit(ASS_train, label_train)

# Make predictions on the test data
predictions_l = pipeline_l.predict(ASS_test)
# Print accuracy score
print("Accuracy:", accuracy_score(label_test, predictions_l))
# Print confusion matrix
print(confusion_matrix(label_test, predictions_l))

#### Neural Net
from sklearn.linear_model import Perceptron

# Build the pipeline with Perceptron classifier and TF-IDF transformation

pipeline_perc = Pipeline([
    ('vect', CountVectorizer(preprocessor=text_preprocessor)),
    ('tfidf', TfidfTransformer()),
    ('clf', Perceptron())
])

# Fit the pipeline to the ASS_training data
pipeline_perc.fit(ASS_train, label_train)

# Make predictions on the test data
predictions_perc = pipeline_perc.predict(ASS_test)
# Print accuracy score
print("Accuracy:", accuracy_score(label_test, predictions_perc))
# Print confusion matrix
print(confusion_matrix(label_test, predictions_perc))


```
</p>

## Demo

Write function to call the classifier. I will use MNB classifier for the demo.
<p>

```python
def show(text):
    a = []
    a.append(text)
    #if pipeline_rf.predict(a)[0]==1:
    #if pipelineMNB.predict(a)[0]==1:
    if pipeline_svc.predict(a)[0]==1:
        b = "You are the asshole"
    else:
        b = "You're not the asshole"
    return b
```
</p>

Use gradio for the interace
<p>

```python
import gradio as gr
iface = gr.Interface(fn=show, inputs="text", outputs="text")
iface.launch(share = True)
```
</p>
