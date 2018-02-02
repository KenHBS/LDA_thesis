# LDA_thesis

## CascadeLDA

A new multi-label document classification technique called CascadeLDA is introduced
in this thesis. Rather than focusing on discriminative modelling techniques, CascadeLDA
extends a baseline generative model by incorporating two types of prior information.
Firstly, knowledge from a labeled training dataset is used to direct the generative model.
Secondly, the implicit tree structure of the labels is exploited to emphasise discriminative
features between closely related labels. By segregating the classification problem in an
ensemble of smaller problems, out-of-sample results are achieved at about 25 times the
speed of the baseline model. In this thesis, CascadeLDA is performed on datasets with
academic abstracts and full academic papers. The model is employed to assist authors in
tagging their newly published articles.

A formal and detailed coverage of baseline LDA, L-LDA, HSLDA and CascadeLDA can be found in `thesis_kenhbs.pdf`. The paper also gives an indepth explanation and derivation of Gibbs sampling and variational inference in the LDA setting. 

## Code content
Python code for multi-label topic modelling with prior knowledge on label hierarchy using Latent Dirichlet Allocation (LDA). This code implements:
    1) Labeled LDA (Ramage et al, 2009)
    2) Hierarchical Supervised LDA (Perotte et al, 2011)
    3) CascadeLDA

## Usage

The code is roughly divided in four parts: Loading and preparing data, train a model, test a model and finally evaluate the predictive quality of the model. This workflow is implemented for L-LDA and CascadeLDA like this


```
$ python3 evaluate_LabeledLDA.py -- help

Usage: evaluate_LabeledLDA.py [options]

Options:
-h, --help   show this help message and exit
-f FILE      dataset location
-d LVL       depth of lab level
-i IT        # of iterations
-s THINNING  save frequency
-l LOWER     lower threshold for dictionary pruning
-u UPPER     upper threshold for dictionary pruning
-a ALPHA     alpha prior
-b BETA      beta prior
-p           Save the model as pickle?

```

So for example:

```
$ python3 evaluate_LabeledLDA.py -f "abstracts_data.csv" -d 3 -i 200 -s 50 -l 0.01 -u 0.99 -a 0.1 -b 0.01 -p
Stemming documents ....
Starting training...
Running iteration # 1 
Running iteration # 2 
Running iteration # 3 
Running iteration # 4 
Testing test data, this may take a while...
Saved the model and predictions as pickles!
Model:              Labeled LDA
Corpus:              Abstracts
Label depth          None
# of Gibbs samples:  4
-----------------------------------
AUC ROC:                  0.696858414365
one error:                0.47198275862068967
two error:                0.5862068965517241
F1 score (macro average)  0.378575246979

```

### CascadeLDA

Simply replace `evaluate_LabeledLDA.py` with `evaluate_CascadeLDA` to perform CascadeLDA instead of L-LDA. The usage is exactly the same. 


## Datasets

Two datasets were used in the thesis. For copyright reasons, only the abstracts dataset is made available here. It consists of 4.500 labeled academic abstracts from the economics literature. The papers are labeled according to the JEL classification. 

## Summary of Challenges

Baseline LDA needs to be adapted to incorporate the following features:

1) Instead of latent topics, we need the topics to correspond exactly to the JEL code descriptions (i.e. explicit topic modelling).
2) Incorporating prior knowledge on document-topic assignment (i.e. we have a training dataset)
3) Many labels are very closely related and barely distinguishable. Even though topic-word distributions are accurate, they are nearly identical and do not allow for discrimination.
