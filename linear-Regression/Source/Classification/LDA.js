// it uses Bayes classifier but here, we use a common covariance matrix among all the features and different
// mean.


// Assumptions : -
// data must be zero centered
// Supports only 2 classes
// each classes ideally have equal no. of samples
// the classes ideally should follow a normal distribution.

// all the classes assumed to have same co-variance matrix (homoscadasticity).

// objective function:- max ( ( mu_0 - mu_1 )**2 / (sigma**2 + sigma**2) ) 
