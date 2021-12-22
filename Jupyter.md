```python
import nltk
nltk.download("stopwords")
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/mirakotamaki/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!





    True




```python
import numpy as np
import json
import glob

#Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

#spacy
import spacy
from nltk.corpus import stopwords

#visualization
import pyLDAvis
import pyLDAvis.gensim_models
```


```python
def load_data(file):
    with open (file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (data)
        
def write_data(file, data):
    with open (file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
```


```python
stopwords = stopwords.words("english")
```


```python
print (stopwords)
```

    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]



```python
data = load_data("Desktop/MiraKotamaki%5Fkorpus%2Etxt.txt")
```


    ---------------------------------------------------------------------------

    JSONDecodeError                           Traceback (most recent call last)

    /var/folders/8l/4ds3f6v13cjc0d7d78x8shn80000gn/T/ipykernel_43816/638152039.py in <module>
    ----> 1 data = load_data("Desktop/MiraKotamaki%5Fkorpus%2Etxt.txt")
    

    /var/folders/8l/4ds3f6v13cjc0d7d78x8shn80000gn/T/ipykernel_43816/3312094966.py in load_data(file)
          1 def load_data(file):
          2     with open (file, "r", encoding="utf-8") as f:
    ----> 3         data = json.load(f)
          4     return (data)
          5 


    ~/opt/anaconda3/lib/python3.9/json/__init__.py in load(fp, cls, object_hook, parse_float, parse_int, parse_constant, object_pairs_hook, **kw)
        291     kwarg; otherwise ``JSONDecoder`` is used.
        292     """
    --> 293     return loads(fp.read(),
        294         cls=cls, object_hook=object_hook,
        295         parse_float=parse_float, parse_int=parse_int,


    ~/opt/anaconda3/lib/python3.9/json/__init__.py in loads(s, cls, object_hook, parse_float, parse_int, parse_constant, object_pairs_hook, **kw)
        344             parse_int is None and parse_float is None and
        345             parse_constant is None and object_pairs_hook is None and not kw):
    --> 346         return _default_decoder.decode(s)
        347     if cls is None:
        348         cls = JSONDecoder


    ~/opt/anaconda3/lib/python3.9/json/decoder.py in decode(self, s, _w)
        335 
        336         """
    --> 337         obj, end = self.raw_decode(s, idx=_w(s, 0).end())
        338         end = _w(s, end).end()
        339         if end != len(s):


    ~/opt/anaconda3/lib/python3.9/json/decoder.py in raw_decode(self, s, idx)
        353             obj, end = self.scan_once(s, idx)
        354         except StopIteration as err:
    --> 355             raise JSONDecodeError("Expecting value", s, err.value) from None
        356         return obj, end


    JSONDecodeError: Expecting value: line 3 column 5 (char 19)



```python

```


```python

```


```python
#lemmatization
def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []
    for text in texts:
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return (texts_out)

lemmatized_texts = lemmatization(data)
print (lemmatized_texts)
            
```

    ['row']



```python
def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return (final)

data_words = gen_words(lemmatized_texts)
```


```python
id2word = corpora.Dictionary(data_words)

corpus = []
for text in data_words:
    new = id2word.doc2bow(text)
    corpus.append(new)
    
print (corpus)

word = id2word[[0][:1][0]]
```

    [[(0, 1)]]



```python
#frequency
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=15,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha="auto")
```


```python
#visualization
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds="mmds", R=15)
vis
```

    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/pyLDAvis/_prepare.py:246: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only
      default_term_info = default_term_info.sort_values(
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/past/builtins/misc.py:45: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      from imp import reload
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/past/builtins/misc.py:45: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      from imp import reload
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/past/builtins/misc.py:45: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      from imp import reload
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/past/builtins/misc.py:45: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      from imp import reload
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/past/builtins/misc.py:45: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      from imp import reload
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/past/builtins/misc.py:45: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      from imp import reload
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/past/builtins/misc.py:45: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      from imp import reload
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/past/builtins/misc.py:45: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      from imp import reload
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: divide by zero encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: divide by zero encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: divide by zero encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: divide by zero encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:125: RuntimeWarning: invalid value encountered in double_scalars
      if(old_stress - stress / dis) < eps:
    /Users/mirakotamaki/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_mds.py:130: RuntimeWarning: invalid value encountered in double_scalars
      old_stress = stress / dis






<link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.3.1/pyLDAvis/js/ldavis.v1.0.0.css">


<div id="ldavis_el438161405647003312487565394247"></div>
<script type="text/javascript">

var ldavis_el438161405647003312487565394247_data = {"mdsDat": {"x": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "topics": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], "cluster": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "Freq": [66.9806626222267, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725, 1.1385978406128725]}, "tinfo": {"Term": ["row", "row", "row", "row", "row", "row", "row", "row", "row", "row", "row", "row", "row", "row", "row", "row", "row", "row", "row", "row", "row", "row", "row", "row", "row", "row", "row", "row", "row", "row", "row"], "Freq": [1.0, 0.6698067784309387, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293, 0.011385980993509293], "Total": [1.0, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082, 1.0000002272427082], "Category": ["Default", "Topic1", "Topic2", "Topic3", "Topic4", "Topic5", "Topic6", "Topic7", "Topic8", "Topic9", "Topic10", "Topic11", "Topic12", "Topic13", "Topic14", "Topic15", "Topic16", "Topic17", "Topic18", "Topic19", "Topic20", "Topic21", "Topic22", "Topic23", "Topic24", "Topic25", "Topic26", "Topic27", "Topic28", "Topic29", "Topic30"], "logprob": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "loglift": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, "token.table": {"Topic": [1], "Freq": [0.9999997727573434], "Term": ["row"]}, "R": 1, "lambda.step": 0.01, "plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [25, 1, 2, 29, 28, 27, 26, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 30]};

function LDAvis_load_lib(url, callback){
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = callback;
  s.onerror = function(){console.warn("failed to load library " + url);};
  document.getElementsByTagName("head")[0].appendChild(s);
}

if(typeof(LDAvis) !== "undefined"){
   // already loaded: just create the visualization
   !function(LDAvis){
       new LDAvis("#" + "ldavis_el438161405647003312487565394247", ldavis_el438161405647003312487565394247_data);
   }(LDAvis);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/LDAvis
   require.config({paths: {d3: "https://d3js.org/d3.v5"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.3.1/pyLDAvis/js/ldavis.v3.0.0.js", function(){
        new LDAvis("#" + "ldavis_el438161405647003312487565394247", ldavis_el438161405647003312487565394247_data);
      });
    });
}else{
    // require.js not available: dynamically load d3 & LDAvis
    LDAvis_load_lib("https://d3js.org/d3.v5.js", function(){
         LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.3.1/pyLDAvis/js/ldavis.v3.0.0.js", function(){
                 new LDAvis("#" + "ldavis_el438161405647003312487565394247", ldavis_el438161405647003312487565394247_data);
            })
         });
}
</script>




```python

```
