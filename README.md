# SpamHam
Implementation of a spam classifier using Bayesian Generative model over a multivariate Bernoulli distribution

- using a generative model can give a really good accuracy as on a very small training set.
- a good accuracy (96%) was achieved on the test data, without the "hard hams."
- dictionary was made out of the words used in the training set, and laplacian smoothing was applied to any test vector
- Naive Bayes classifer was used here, but support vector machines, logistic regression, Gaussian Discriminant Analysis may also be used.


- hard hams are specifically designed emails which are known to get misclassified by spam ham classifiers, as spams, but are actually legit emails.
