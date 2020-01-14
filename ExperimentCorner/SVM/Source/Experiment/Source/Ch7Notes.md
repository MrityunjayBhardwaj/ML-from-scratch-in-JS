
## Support Vector Machine
page 325-334 : covered in, seperatinghyperplane article and SVM article. 

One of the most popular approach to trainng support vector machine is called Squential minimal optimization (SMO) it takes the concept of dividing the optimization problem into smaller chunks to an extreme level( a.k.a **chunking**) i.e, it optimize w.r.t only 2 lagrange multipliers at a time, thereby avoiding numerical quadratic programming altogethee

the reason why kernel method works without suffering from curse of dimensionality. because any set of points in the original n-dimensional input space 'x' is constrained to lie exactly on a n-dimensional nonlinear manifold embedded in the k-dimensional feature space. where, k could even be infinity if we use RBF kernel.

although SVM does not provide probabilistic outputs (it only makes classification decisions for new input vectors. ) but we can go around this problem by fitting a logistic sigmoid to the outputs of a previously trained SVM. specifically, the required conditional probability is assumed to be of the form:

$$ p(t=1|x) = \sigma{(Ay(x) + B)}$$
inwhich, we used weights and bias of the previously trained SVM.
according to tipping, because the SVM training precedure is not specifially inteneded to encourage this 2 stage procedure, <u>SVM can give a poor approxmiation to the posterior probabilities</u>.

##Relation to logistic regression.

Both the logistic error and the hinge loss (constructed by SVM) can be viewed as a continuous approximations to the misclassification error. the key difference between them is that the flat region of hinge loss leads to a much sparser solution.

another continuous error function is squared error (rpresented in green color). it has a property of placing increasing emphasis on data points that are correctly classified but are a long way from the decision boundary on the correct side. such points will be strongly weighted at the expense of misclassified points thats why a monotonically decreasing error function (like logs etc) would be a better choice.

## Multiclass SVMs

// add figure 7.6
Although there are several options to scale the SVM for multi-class problems like for eg. one-vs-one ( which is used in almost all of the packages), one-vs-the-rest approach, some more intresting ones include using DAG to organize the pairwise classification into a data structure this method known as DAGSVM which has a total of K(K-1)/2 classifiers and to classify a new test point only K-1 pairwise classifiers is needed, with the particular classifiers used depending on which path through the graph is traversed.
the application of SVMs to multiclass classification still an open issue.

there is also a single-clas svm which solve an unsupervised learning problem related to probability density estimation these methods aim to find a smooth boundary enclosing a region of high density. the boundary is chosen to represent a quantile of the density i.e, the probability that a data point drawn from the distribution will land inside theat reagion is given by a fixed number between 0 and 1 that is specified in advance.

Two approaches to this problem using supportvector machines have been proposed. The algorithm of Sch ̈olkopfet al.(2001) triesto find a hyperplane that separates all but a fixed fractionνof the training data fromthe origin while at the same time maximizing the distance (margin) of the hyperplanefrom the origin, while Tax and Duin (1999) look for the smallest sphere in featurespace that contains all but a fractionνof the data points.  For kernelsk(x,x)thatare functions only ofx−x, the two algorithms are equivalent.


## SVM for regression

// add pic 7.6 and 7.7
the task of regression is similar to the task of classification in SVM but instead of using the squared error loss with a regularization term we introduce $$ \epsilon $$-insensitive error function, which gives zero error if the abs difference between the prediction y(x) and the target 't' is less than $$ \epsilon $$ where > 0 :-

// add 51 and 52

this formulation look suspeciously close to our classification formulation, and indeed it is! and we are going to be finding its natural parameters in the similar fashion. and in contrast with SVM we uses 3 complexity parameters instead of 2 for regression task. for more indepth mathematical derivation headover to pgno. 341-344

## Computational Learning Theory

The famous Statistical learning Theory a.k.a Computational leanring theory is originated from probably approximately correct or PAC.

for a given choice of model space( family of functions ) $$ F $$  PAC learning aims to provide lower bound on the amount of data required to achieve a generalization error which is bounded above by $$ \epsilon $$ :-

$$ E_{x,t} [ I(f(x; D) \neq t)] < \epsilon  $$

also, PAC framework requires that this holds with probability greater than $$ 1-\delta $$ for a data set D drawn randomly from p(x, t) .

the terminology 'probably approximately correct' comes from the requirement that with high probability (greater than $$ 1-\delta $$), the error rate be mall (less than $$ \epsilon $$).

we can also use VC-dimensions to calculate the complexity of a space of functions and which allows the PAC framework to be extended to spaces containing an infinte number of functions. bounds derived within the PAC framework are often described as worst-case because it doesn't assume any specific distribution for the family of function on which we are calculating the PAC bounds on.

we can improve the tightness of the PAC bounds is by using PAC-Bayesian framework which considers a distribution over the space $$ F $$ of functions, like a prior in a Bayesian treatment. but even after that, it is still very conservative.

# Relevance Vector Machine

why SVM is not good enough?
* as mentioned earlier, the outputs of an SVM represent decisions rather than posterior probabilities.
* hard to scale for more then 2 class problem
* need to find the complexity parameters using cross-validation etc.
* predictions are expressed as linear combinations of kernel functions that are centered on trining data points adnd that are required to be positive definate.

RVM is a bayesian sparse kernel technique for regression and classification which is similar to SVM whilst avoiding its principal limitations. additionally, it typically leads to much sparser models resulting in correspondingly faster performance on test data whilst maintaining a comparable generalization error.

in RVM, we used the same formulation as Bayesian linear regression 
$$ 
p(t|x,w,\beta) = N(t|y(x), \beta^{-1}) \\
\text{where, } \qquad y(x) = \sum_{i=1}^{M} {w_i\phi{x}} = w^T\phi(x)
$$

in order to mirror the structure of the SVM the basis fucntion used in RVM are given by kernels, with one kernel associated with each of the data points from the training set.

$$
 y(x) = \sum_{i=1}^{M} {w_i{k(x,x_n)}} + b= w^T{k(x,x')} + b
$$
which looks similar to SVM for regression task. but in contrast with SVM, there is no restriction to use only positive definite kernels, nor are the basis functions tied in eiter number or location to the training data points.

now lets find the likelihood function of our dataset :-

$$ p(t|X,w,\beta) = \Pi_{n=1}^{N} p(t_n | x_n, w,\beta)$$

just like in baysian linear regression, we introduce a prior over weights:-
$$ p(w|\alpha) = \Pi_{i=1}^{M} N(w_i| 0, \alpha_i^{-1}) $$

where, $$ \alpha_i $$ represent the precision of the corresponding parameter w_i.
....

it is important to note that, when we maximize the likelihood function with respect to these hyperperparams, a significant proportion of them goes to infinity and the corresponding weight params have posterior distrivutions that are concentrated at zero. the basis functionsassociated with these params therefore  <u> play no role in the predictions made by the model and so are effectively pruned out<u>, resulting in much sparser model.

$$ p(w|t,X,\alpha, \beta) = N(w|m, \sigma) $$

where, 
$$ m = \beta\sigma\Phi^Tt $$
$$ \Sigma = (A + \Beta\Phi^T\Phi)^{-1} $$


as usual, we can determine the values of \alpha and \beta evidence approximation.