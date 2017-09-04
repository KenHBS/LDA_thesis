# LDA_thesis

Preliminary codes for topic modelling with prior knowledge about multiply labeled documents. 

This will be an extended version of Latent Dirichlet Allocation. A number of additional features are added:
1) (Multiple) labels are assigned to every document
2) Prior knowledge on topics, i.e. the topics are explicit, not latent
3) Labels may be correlated with each other.
4) Labels are not equally likely, prior differences in label frequencies should be accounted for
5) The results of the topic model will be used to classify unseen documents.

So far, Collapsed Gibbs Modelling according to Ramage '09 has been implemented. 

To do:
- Include points 3, 4 and 5.
- Include estimation procedures with Variational Inference
- Establish metrics for classification accuracy
