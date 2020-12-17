# ABSA (E2E-BERT-ABSA)

**Architecture:**

![bert-absa-architecture](https://gecgithub01.walmart.com/storage/user/42911/files/1f07aa00-d17b-11ea-9cea-2e584eb7570e)

**Exploration:**
- Ran fast_run.py for hours on local and saved the checkpoint weights
- Torch not imported on terminal so copied code to jupyter notebook and ran
- Evaluated against SemEval2014, 2015, and 2016 restaurants dataset
- Found that one run needs 3 GPUs (Blaize to come back with compute resource on Friday)
- For Cerebro: Could be a way of exposing the feature when the business has only text and no other  information

**Evaluation Results: (with partial training)**
- class_count: [1524.  500.  263.]
- macro-f1: 0.6104446342147495
- precision: 0.6842308024079615
- recall: 0.7665060050257957
- micro-f1: 0.7225373307113764

**Reference:**
- https://github.com/lixin4ever/BERT-E2E-ABSA

# ABSA (PyPI package)

**Exploration:**
- Implemented pre-trained *classifier-01*
- Passed short paragraph of + 7 aspects and returned sentiment polarity
- For Cerebro: Could be a way of explosing the feature given that the business has text and a list of aspects to determine polarity for

**Reference:**
- https://github.com/ScalaConsultants/Aspect-Based-Sentiment-Analysis
- [Link to notebook in this repo](https://gecgithub01.walmart.com/nextech/sentiment-analysis-research/blob/master/evaluations/ScalaConsultants%20ABSA%20Model.ipynb) 

# Spacy Rule-Based Approach

**Exploration:**
- Wrote *target-extraction* module using spacy for converting piece of text into tokens, and extract roots of noun-chunks as aspects
- Added more functionality to include *coreference-resolution* and rules to extract *target-opinion pairs*
- Evaluated against SemEval2016 Restaurants dataset

**Evaluation Results:**
- Match if all of actual-aspects is present in predicted-aspects list
- 58% match

**Reference:**
- [Link to notebook in this repo](https://gecgithub01.walmart.com/nextech/sentiment-analysis-research/blob/master/evaluations/Evaluation%20of%20rule-based%20approach.ipynb)


