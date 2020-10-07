## Benchmark Datasets

Laptops and Restaurant Reviews (3-6K sentences each)
- SemEval2014 Task 4: http://alt.qcri.org/semeval2014/task4/
- SemEval2016 Task 5: http://alt.qcri.org/semeval2016/task5/

## Dataset format:
The sentences in the datasets are annotated using XML tags.

The following example illustrates the format of the annotated sentences of the restaurants dataset.

```
<sentence id="813">
          <text>All the appetizers and salads were fabulous, the steak was mouth watering and the pasta was delicious!!!</text>
          <aspectTerms>
                    <aspectTerm term="appetizers" polarity="positive" from="8" to="18"/>
                    <aspectTerm term="salads" polarity="positive" from="23" to="29"/>
                    <aspectTerm term="steak" polarity="positive" from="49" to="54"/>
                    <aspectTerm term="pasta" polarity="positive" from="82" to="87"/>
          </aspectTerms>
          <aspectCategories>
                    <aspectCategory category="food" polarity="positive"/>
          </aspectCategories>
</sentence>
```

## With Un-labeled Data

#### 0. Naive TextBlob Approach

#### 1. Rule Based Approach (POS Tags + Dependency Parser + Coreference Resolution)
- POS Tag and Stanford Dependency parser to extract Opinion pairs
- Co-reference resolution over the review. In a sentence, co-reference resolution maps the pronouns to the nouns </br>
which they are referring

  Links:
  - https://medium.com/@nitesh10126/aspect-based-sentiment-analysis-in-product-reviews-unsupervised-way-fb0b38ead501
  - http://ceur-ws.org/Vol-1874/paper_6.pdf
  - https://nlp.stanford.edu/projects/coref.shtml
  
#### 2. Pre-trained ABSA model (https://github.com/ScalaConsultants/Aspect-Based-Sentiment-Analysis)
 
#### 3. MUSEGMMTopicModel to extract topics/aspects (https://gecgithub01.walmart.com/nextech/mgtm)

#### 4. MUSE embeddings, train classifier on standard benchmark dataset (similar to https://github.com/deepopinion/domain-adapted-atsc)

## With Semi-Labeled Data(1 aspect per aspect category)
Build a an LDA-based topic model extended with additional variables, with biased topic modelling hyperparameters </br>
based on continuous word embeddings, and combined with unsupervised pre-trained Maximum Entropy Classifier model for </br>
aspect-term/opinion-word separation.

<img width="790" alt="Screen Shot 2020-07-01 at 8 35 19 AM" src="https://gecgithub01.walmart.com/storage/user/42911/files/f5586480-bb75-11ea-8ec7-14864514c879">

System uses Brown clusters to model examples of aspect-terms and opinion-words and train a MaxEnt-based classification model
1) Brown clusters are computed from the domain unlabelled corpus with no additional supervision, and are used as the features for the two words context window, [-2,+2], of each training example. 
2) The training instances are obtained leveraging the occurrences of the initial configuration with aspects and polarity seed words, assuming that domain aspect seed words are aspect-terms and polarity-words are opinion-words.
3) The occurrences of seed words are bootstrapped from the domain corpus and they are modelled according to their context window
4) Next, context words are replaced by their corresponding Brown cluster to build each training instance
5) A Maxent model is trained using these generated training instances

Topic and Polarity Modeling
1) Hyper-parameters for LDA are biased using a similarity calculation among the words of the domain corpus and the topic seed words of the initial configuration. This similarity measure is based on the cosine distance between the dense vector representation (WORD2VEC) of the topic defining seeds and each word of the vocabulary
2) Change alpha, beta, and delta hyperparameters

####Benchmark Baselines

<img width="768" alt="Screen Shot 2020-07-01 at 8 52 49 AM" src="https://gecgithub01.walmart.com/storage/user/42911/files/63eaf180-bb79-11ea-9377-543baf05843e">

<img width="759" alt="Screen Shot 2020-07-01 at 8 51 08 AM" src="https://gecgithub01.walmart.com/storage/user/42911/files/0b672480-bb78-11ea-913a-3b1a4f585d18">

Supervised Methods (eg. Multinomial Naive Bayes Classifier) have been able to get to a score of 0.77 for domain-aspect extraction and 0.80 for aspect-poloarity detection

BERT yields the best subset accuracy of 79.13%, followed by XLNet’s 78.41%. RF + LP takes the third position across all the baseline models. Both BERT and XLNet have been reported as the state-of-the-art approaches in NLP-related learning tasks. 

## Supervised Methods
- Naive Bayes
- MLP
- BERT-ADA






  

  
