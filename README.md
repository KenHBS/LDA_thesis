# LDA_thesis

Preliminary codes for topic modelling with prior knowledge about multiply labeled documents and label hierarchy.

4.000 academic abstracts from the economics literature are used to train a topic model. The labels for this data are the JEL classifications. JEL classifications are the standard way to categorise papers in the economic literature. 

Latent Dirichlet Allocation will form the basis. The unsupervised nature will be transformed into a supervised classification by incorporating two extensions to the standard LDA case:
Extension 1: Multi-labeled training data, Ramage et al, (2009)
Extension 2: Hierarchically Supervised LDA (HSLDA), Perotte et al (2011)

These two extensions allow for:
1) Instead of latent topics, we need the topics to correspond exactly to the JEL code descriptions (i.e. explicit topic modelling).
2) Incorporating prior knowledge on document-topic assignment.
3) Many topics (i.e. JEL codes) are very closely related and barely distinguishable based on the abstract only. HSLDA implicitly finds discriminative characteristics between 'neighbouring' labels.
4) Labels are not equally likely, prior differences in label frequencies should be accounted for

doc_prepare.py contains functions for text mining preparation on the corpus of documents.
CGS.py implements Collapsed Gibbs Sampling for:
    1) the standard LDA case 
    2) the supervised LDA case a la Ramage '09'
    3) HSLDA (in progress)
run_like_this.py contains examples of how to use the documents.

To do:
- Include estimation procedures with Variational Inference
- Finalise Hierarchically Supervised LDA (HSLDA by Perotte)
- Evaluate results

NOTE: For copyright reasons the dataset is not made available. Upon finalisation of the thesis a small subset will be made publicly available. 
