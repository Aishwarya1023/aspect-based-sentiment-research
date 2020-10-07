import spacy
import os
#import neuralcoref
from spacy import displacy
from textblob import TextBlob
import argparse

nlp = spacy.load('en_core_web_sm')

def create_nlp_pipe(coreference = True):
    """Creates the nlp pipeline using spacy. Adds neuralcoref if True
    
    Arguments:
        coreference {bool}: If True, coreference is added to pipeline
    
    Returns:
        nlp {obj}: nlp object
    """
    
#    if coreference:
#        coref = neuralcoref.NeuralCoref(nlp.vocab)
#        nlp.add_pipe(coref, name='neuralcoref')
    return nlp

def get_pos_tags(doc):
    """Prints the pos tags for each token in the document
    
    Arguments:
        doc {obj}: nlp document object
    
    Returns:
        token.pos_ {unicode}: POS tag of each token
    """
    
    for token in doc:
        print(token.text, token.pos_)

# nlp.add_pipe(nlp.create_pipe('merge_noun_chunks'))

def merge_nouns(doc):
    """If two consecutive tokens are nouns, it concatenates them into one \
    representing one aspect term.
    
    Arguments:
        doc {obj}: nlp document object
    
    Returns:
        new_text {str}: text for retokenization after nouns have been merged
    """
    
    i=0
    new_text = ''
    while i<len(doc):
        if doc[i].dep_ == 'compound':
            compound_noun = doc[i].text + doc[i+1].text
            new_text = new_text + ' ' + compound_noun
            i += 2
        else:
            new_text = new_text + ' ' + doc[i].text
            i += 1
    return new_text

def get_aspect_terms(doc):
    """This function returns the root noun present in noun chunks of the document.
    
    Arguments:
        doc {obj}: nlp document object
    
    Returns:
        aspects {list{str}}: list of root nouns as aspects
    """
                                            
    aspects = [(chunk.root.text) for chunk in doc.noun_chunks if chunk.root.pos_ == 'NOUN']
    return aspects

def get_dependencies(doc):
    """Prints the dependency tree of the document
    
    Arguments:
        doc {obj}: nlp document object
    
    Returns:
        token.text {unicode}: Verbatim text content
        token.dep_ {unicode}: Syntactic dependency relation
        token.head.text {unicode}: The syntactic parent, or “governor”, of a token
        token.head.pos_ {unicode}: POS tag of the governor
        children {list}: list of children of a token
        
    """
    
    for token in doc:
        print(token.text, token.dep_, token.head.text, token.head.pos_,
                [child for child in token.children])

def get_opinion_pairs(doc):
    """This function returns the opinion pairs based on pre-defined rules.
    
    Arguments:
        doc {obj}: nlp document object
    
    Returns:
        opinion_pairs {list{tuple}}: list of tuples consisiting of (aspect, opinion)
    """
    
    opinion_pairs = []
    for token in doc:
        if token.dep_ == 'nsubj' and TextBlob(token.head.text).polarity > 0.4:
            opinion_pairs.append((token.text, token.head.text))
        elif token.dep_ == 'dobj' and (token.head.pos_ == 'ADJ' or TextBlob(token.head.text).polarity > 0.4):
            opinion_pairs.append((token.text, token.head.text))
        elif token.dep_ == 'amod' and token.head.pos_ == 'ADJ':
            opinion_pairs.append((token.text, token.head.text))
    return opinion_pairs

def tokenize(text):
    """Parses the given text using nlp pipeline
    
    Arguments:
        text {str}: Text to be parsed
    
    Returns:
        doc {obj}: nlp document object
    """
    
    doc = nlp(text)
    return doc

def plot_dependencies(doc):
    """Plots the dependencies in the nlp document
    
    Arguments:
        doc {obj}: nlp document object
    
    Returns: 
        displacy plot
    """
    
    displacy.serve(doc, style="dep")

def get_sentiment_terms(doc):
    """This function return the adjectives and verbs that are not stopwords \
    or punctuations, indicating a descriptive/polarized word.
    
    Arguments:
        doc {obj}: nlp document object
    
    Returns:
        sentiment_terms {list{str}}: list of sentiment terms
    """
    
    sentiment_terms = []
    if doc.is_parsed:
        sentiment_terms.append([token.lemma_ for token in doc if \
                                (not token.is_stop and not token.is_punct \
                                 and (token.pos_ == "ADJ" or token.pos_ == "VERB"))])
    else:
        sentiment_terms.append('') 
    return sentiment_terms

def target_extraction_pipeline(text):
    """This is the main pipeline that runs for extracting targets using rule based method
    
    Arguments:
        text {str}: string of text to extract targets from
        
    Returns:
        aspects {list}: list of target terms
    
    """
    nlp = create_nlp_pipe(False)
    doc = tokenize(text)
    aspects = get_aspect_terms(doc)
    
    return aspects

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('text')
    args = parser.parse_args()
    aspects = target_extraction_pipeline(args.text)

    print(aspects)
