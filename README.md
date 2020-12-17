# Aspect Based Sentiment Analysis Research

## Definition(s)
Sentiment Analysis is the process of computationally identifying and categorizing opinions expressed in a piece of text, especially in order to determine the writer's polarity (eg. positive, negative, neutral) towards a piece of text. This text can be a clause, sentence, paragraph, or a whole document.

The main difference between **sentiment analysis** and **aspect based sentiment analysis** is that the former only detects the sentiment of an overall text, while the latter analyzes each text to identify various aspects and determine the corresponding sentiment for each one. To understand this better, it is important to state the difference between a sentiment and an aspect.

**Sentiment:** polarity of opinions about a particular text <br />
**Aspect:** the thing or topic that is being talked about. There can be multiple aspects in one piece of text

There are mainly three approaches in sentiment analysis:
  *	**Lexicon based** - considers lexicon dictionary for identifying polarity of the text
  *	**Machine learning based approach** - Needs to develop classification model, which is trained using prelabeled dataset of positive, negative, neutral content.
  *	**Hybrid approach** - Which uses lexicon dictionary along with pre-labelled data set for developing classification model.
  
 
![sentiment analysis techniques](https://gecgithub01.walmart.com/storage/user/42911/files/623efc80-afc3-11ea-96a1-d52b4bd00947)

Terms commonly associated with sentiment analysis are: opinion mining/analysis, sentiment classification, semantic orientation/analysis, subjectivity mining/analysis.

Other Definitions:
 * Lexicon - i.e. lists of words and expressions
 * Affixes - affix (length 2-4 characters) of a word at a given position.
 * Tf-idf - term frequency-inverse document frequency of a word.
 * Learned dictionary - a dictionary of aspect terms from training data.
 * Bag of words - a representation of text that describes the occurrence of words within a document
 * Bigrams - a pair of consecutive written units such as letters, syllables, or words

References:
 * https://arxiv.org/pdf/1612.01556.pdf
 * https://monkeylearn.com/blog/aspect-based-sentiment-analysis/#:~:text=The%20big%20difference%20between%20sentiment,corresponding%20sentiment%20for%20each%20one.


## Benchmarks

**SentiWordNet** is a lexical resource for sentiment classification and opinion mining applications. It is used to identify objective sentences and later to identify the polarity for the opinion words.

The most popular and widespread datasets are:
  *	Stanford Twitter Sentiment (http://help.sentiment140.com/)
  *	Sentiment Strength Twitter Dataset (http://sentistrength.wlv.ac.uk/documentation/)
  *	Amazon Reviews for Sentiment Analysis (https://www.kaggle.com/bittlingmayer/amazonreviews)
  *	Large Movie Review Dataset (http://ai.stanford.edu/~amaas/data/sentiment/)
  *	Sanders Corpus (http://www.sananalytics.com/lab/twitter-sentiment/)
  *	SemEval (Semantic Evaluation) dataset (http://alt.qcri.org/semeval2017/task4/index.php?id=data-and-tools)

*Setting a baseline sentiment accuracy rate:*
When evaluating the sentiment (positive, negative, neutral) of a given text document, research shows that human analysts tend to agree around 80-85% of the time. This is the baseline we (usually) try to meet or beat when we're training a sentiment scoring system. Most sentiment models can exceed this 80-85% baseline.

As a classification problem, sentiment analysis uses the evaluation metrics of Recall, F-score, and Accuracy. Also, average measures like macro, micro, and weighted F1-scores are useful for multi-class problems.

*Precision:*
A measure of how often a sentiment rating was correct. For documents with tonality, precision tracks how many of those that were rated to have tonality were rated correctly.

*Recall:*
A measure of the completeness, or sensitivity, of a classifier. Higher recall means less false negatives, while lower recall means more false negatives. Generally, high recall scores are very difficult in tests of broad subject matter, as the system is required to understand ever-larger sets of words and language.

*F1 Score:*
Also called F-Score or F-Measure, this is a combination of precision and recall. The score is in a range of 0.0 - 1.0, where 1.0 would be perfect. The F1 Score is very helpful, as it gives us a single metric that rates a system by both precision and recall.

References:
https://www.lexalytics.com/lexablog/sentiment-accuracy-baseline-testing


## Sub Tasks

**Subtask 1: Aspect term extraction**

Given a set of sentences with pre-identified entities (e.g., restaurants), identify the aspect terms present in the sentence and return a list containing all the distinct aspect terms. An aspect term names a particular aspect of the target entity.

`Eg: "I liked the service and the staff, but not the food” --> {service, staff, food}`

This task is usually performed using a **domain specific NER system**. NER can be implemented with **Conditional Random Fields** (supervised metthod, regarded as the state-of-the-art method for NER) making use of different types of contextual info with a variety of features such as word prefixes and shapes that are helpful in predicting the NE classes. If this method doesn't perform well, the fallback is to identify the nouns through POS tagging and mark those as aspects. 

Abstract Term Extraction can be enhanced by generating a **Dependency Graph** to observe dependencies (edges) between the words (nodes) 

<img width="1394" alt="Screen Shot 2020-06-15 at 10 35 10 PM" src="https://gecgithub01.walmart.com/storage/user/42911/files/a54d9f80-afc4-11ea-8a85-de0864e025ac">

**Subtask 2: Aspect term polarity**

For a given set of aspect terms within a sentence, determine whether the polarity of each aspect term is positive, negative, neutral or conflict (i.e., both positive and negative).

`Eg: "I liked the service and the staff, but not the food” --> {service: positive, staff: positive, food: negative}`

Sentiment Analysis method - Rule Based/ML

**Subtask 3: Aspect category detection**

Given a predefined set of aspect categories (e.g., price, food), identify the aspect categories discussed in a given sentence. Aspect categories are typically coarser than the aspect terms of Subtask 1, and they do not necessarily occur as terms in the given sentence.

`Eg: given the set of aspect categories {food, service, price, ambience, anecdotes/miscellaneous}
“The restaurant was expensive, but the menu was great” → {price, food}`

A model (SVM is widely used) for predicting the category which aspect belongs to. To enhance this, the model can take into consideration a list of synonyms and antonyms, to make it able to identify that sentences like “the food was expensive” and “the food was inexpensive” will both fall under one category “price”. An Attention layer can further be added to focus on certain categories.

**Subtask 4: Aspect category polarity**

Given a set of pre-identified aspect categories (e.g., {food, price}), determine the polarity (positive, negative, neutral or conflict) of each aspect category.

`Eg: “The restaurant was expensive, but the menu was great” → {price: negative, food: positive}`

Sentiment Analysis method - Rule Based/ML

References:
http://alt.qcri.org/semeval2014/task4/

## Semantic Evaluation (SemEval)

SemEval is a series of international natural language processing (NLP) research workshops whose mission is to advance the current state of the art in semantic analysis and to help create high-quality annotated datasets in a range of increasingly challenging problems in natural language semantics. Each year’s workshop features a collection of shared tasks in which computational semantic analysis systems designed by different teams are presented and compared.

The first major area in semantic analysis is the identification of the intended meaning at the word level (taken to include idiomatic expressions). This is word-sense disambiguation (a concept that is evolving away from the notion that words have discrete senses, but rather are characterized by the ways in which they are used, i.e., their contexts). The tasks in this area include **lexical sample and all-word disambiguation, multi- and cross-lingual disambiguation, and lexical substitution.** Given the difficulties of identifying word senses, other tasks relevant to this topic include **word-sense induction, subcategorization acquisition, and evaluation of lexical resources.**

The second major area in semantic analysis is the understanding of how different sentence and textual elements fit together. Tasks in this area include **semantic role labeling, semantic relation analysis, and coreference resolution. Other tasks in this area look at more specialized issues of semantic analysis, such as temporal information processing, metonymy resolution, and sentiment analysis.** The tasks in this area have many potential applications, such as information extraction, question answering, document summarization, machine translation, construction of thesauri and semantic networks, language modeling, paraphrasing, and recognizing textual entailment. In each of these potential applications, the contribution of the types of semantic analysis constitutes the most outstanding research issue.

Most of the research in aspect-based sentiment analysis are tasks/sub-tasks defined for each SemEval dataset.

References:
* https://semeval.github.io/
* https://en.wikipedia.org/wiki/SemEval

## ABSA vs. TABSA (Target Aspect Based Sentiment Analysis) 

ABSA identifies aspects within a piece of text assuming that only a single entity is being talked about.
`Eg: “The design of the space is amazing but the service is horrid!”` refers to one place.

On the other hand, TABSA doesn't make this assumption and maps aspects to entities.
`Eg: “The design of the space is great in McDonalds but the service is horrid, on the other hand, the staff in KFC are very friendly and the food is always delicious.”` talks about two places (or entities)

The article refered below demonstrates ABSA on SemEval 2014 dataset and TABSA on SentiHood dataset.

References:
* https://towardsdatascience.com/day-104-of-nlp365-nlp-papers-summary-sentihood-targeted-aspect-based-sentiment-analysis-f24a2ec1ca32

## Models

From [this paper](https://arxiv.org/pdf/1805.01984.pdf), here are some ways to encode aspect for modeling: 

`Eg: “the battery life of the phone is too short”`
1)	ID encoding – [0 x x 0 0 0 0 0 0] or [0 x 0 0 0 0 0 0] (apply zero-padding to make lengths equal)
2)	Bit Masking – [0 1 1 0 0 0 0 0 0]
3)	Location Encoding – [1 1 2 3 4 5 6] (for each aspect term ai and sentence sj, we encode the location of each context word ck with respect to the aspect term in the sentence)

The introduction of the SemEval competition resulted in a rise to the number of proposed methods for aspects extractions. SOme of the proposals are:

In [this paper](https://doi.org/10.1063/1.4994463), sentiment analysis and classification techniques are used to determine the sentiment polarity of restaurants reviews using the SemEval 2014 dataset:
 * *Feature extraction* using Chi Square, **resulting in higher computational speed** than Gini index despite reducing the system performance. 
 * *Naïve Bayes classification of sentiment polarity* was used to classify both aspects and sentiments. The evaluation results indicated that the system performed well with a highest score of 78.12% for the F1-Measure. 

[This paper](https://doi.org/10.1016/j.jocs.2017.11.006) aimed to investigate three different tasks; aspect category identification, aspect opinion target expression (OTE), and aspect sentiment polarity identification on SemEval 2016:
When compared to baseline researches, the results indicated that **SVM outperforms the deep RNN** in all the investigated tasks. However, the **deep RNN was found to be more appropriate and faster** in terms of training and testing execution time.

[This paper](https://doi.org/10.1007/s13042-018-0799-4) used two applications of deep learning long short-term memory (LSTM) neural networks for aspect-based sentiment analysis of SemEval 2016:
character-level bidirectional LSTM with a conditional random field classifier used for aspect OTE extraction - 39% improvement
aspect-based LTSM for aspect sentiment orientation classification - 6% improvement

[This paper](https://www.sciencedirect.com/science/article/pii/S2210832719303163#b0115) shows competitive resultst for different languages and domains

In [this paper](https://doi.org/10.1016/j.asoc.2017.07.056), ontology-based feature level sentiment analysis model is implemented for the domain of “Smartphones” for tweets:
The system provided **good accuracy** and demonstrated the importance of using emoji and emoticons detection as well as the use of attribute specific lexicons.

**Other Papers with Model Implementation**
 * https://www.sciencedirect.com/science/article/pii/S0950705115001471
 * https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf 
 * (China) T-ASBA using BERT via Auxillary https://towardsdatascience.com/day-103-nlp-research-papers-utilizing-bert-for-aspect-based-sentiment-analysis-via-constructing-38ab3e1630a3 
 * (China) ASBA using BERT via Auxillary https://www.aclweb.org/anthology/N19-1035.pdf 
 * (Sweden) ASBA using BERT Paper: https://www.aclweb.org/anthology/W19-6120.pdf 
 * (Stanford) CNN https://cs224d.stanford.edu/reports/WangBo.pdf a framework consisting of two deep learning models for aspect and sentiment prediction respectively
 * (India) Using KNN and SVM (downloaded locally) https://www.researchgate.net/publication/266776421_ASPECT_BASED_SENTIMENT_ANALYSIS_SEMEVAL-2014_TASK_4 
 * https://www.sciencedirect.com/science/article/pii/S2210832719303163 
 * (Chicago) Various Approaches to ASBA https://arxiv.org/pdf/1805.01984.pdf 

**Rule-Based Approaches:**

The image below demonstrates result of an experiment to extract aspects using a rule-based approach starting from baseline, and adding components/features to the rules. These rules settings are adopted to handle numerous challenges in sentiment analysis such as handling negation (N), intensification (I), downtoners (D), repeated characters (R) and special cases of negation-opinion rules (S).

<img width="677" alt="Screen Shot 2020-06-16 at 1 28 29 PM" src="https://gecgithub01.walmart.com/storage/user/42911/files/5a885380-afd5-11ea-88fb-2ffb6d9b2429">

References:
 * https://paperswithcode.com/task/aspect-based-sentiment-analysis
 * https://www.sciencedirect.com/science/article/pii/S2210832719303163#b0125
 * https://pypi.org/project/aspect-based-sentiment-analysis/ 
 * https://github.com/ScalaConsultants/Aspect-Based-Sentiment-Analysis 


## Use Cases

*Product Feedback*

Today, there’s an abundance of feedback on social media, your Net Promoter Score (NPS), websites and much more, and all this textual customer feedback is key to discovering and solving customer problems. Here's how aspect-based sentiment analysis can be used to make sense of customer feedback:
Understand specific aspects that customers like and dislike about your brand.
Analyze service and product reviews to discover the successes and failures of your brand, and compare them to your competitor’s.
Track how customer sentiment changes toward specific features and attributes of a service or product.

*Customer Support*

Customers don’t like waiting for a solution to their problems, which means customer support teams need to respond quickly and effectively. If not, chances are customers will look elsewhere. That’s why businesses need high-quality machine learning software like aspect-based sentiment analysis to:
Automate tagging of all incoming customer support queries.
Quickly find out why customers are unhappy.
Send queries and complaints to team members that are best equipped to respond.
Gain insights into how your customer support team handles customers.

Large enterprises perform sentiment analysis to analyze public opinion, conduct market research, monitor brand, product reputation, and understand customer experiences.

Various products often provide integration of sentiment analysis APIs/ plugins for customer experience management, social media monitoring, or workforce analysis, in order to deliver useful insights to their customers.

References:
 * https://monkeylearn.com/blog/aspect-based-sentiment-analysis/
 * https://medium.com/@Intellica.AI/aspect-based-sentiment-analysis-everything-you-wanted-to-know-1be41572e238

## Edge Cases/Problems/Difficulties

The most common challenges faced in building ABSA models are:

* Sensitivity to Context
* Subjectivity and Tone
* Comparisons
* Irony and Sarcasm
* Defining Neutral

References:
* https://monkeylearn.com/blog/what-is-aspect-based-sentiment-analysis/


## Examples

**Libraries**

* NLTK:- excellent for learning and exploring NLP concepts, but not always suitable for production
* Sentiwordnet: used to extract emotions
* TextBlob:- it is built on top of NLTK. It is used for fast prototyping or building applications that don’t require highly optimized performance. 
* Stanford’s CoreNLP:- used to extract entities and dependencies. It is Java Library with python wrapper functions, fast and reliable and used in many places: The recursive RNN model builds up a representation of whole sentences based on the sentence structure. It computes the sentiment based on how words compose the meaning of longer phrases. This way, the model is not as easily fooled as previous models. 
* Aspect-based-sentiment-analysis PyPI package
* SpaCy: It is comparatively new. It is fast, streamlined and production ready.
* Gensim is most commonly used for topic modeling and similarity detection. It’s not a general-purpose NLP library, but for the tasks it does handle, it does them well.
* Syuzhet package in R implements some of research from the NLP Group at Stanford. (emotion extraction)
* Open NLP

**Existing companies providing sentiment as a service:**
* [Aylien](https://aylien.com/research/) - one of focus areas is training hierarchical models with many semantic tasks: NER, Mention Detection, Relation Extraction 
* [Intellica](https://intellica.ai/#how-we-work) - Aspect based with "noun" POS tagging 
* [MonkeyLearn](https://monkeylearn.com/blog/aspect-based-sentiment-analysis/) 
* [ibm-watson](https://github.com/watson-developer-cloud/python-sdk/tree/master/examples) - API by IBM (tone analyser, personality insights) 
* [Aspectiva](https://www.aspectiva.com/)
* [Intel NLP Architect](https://github.com/microsoft/nlp-recipes/tree/master/examples/sentiment_analysis/absa)
