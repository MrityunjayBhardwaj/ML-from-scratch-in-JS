
# Seperating Hyperplane : and how to find them

// TODO:
*  explain how we classify our points
*  explain Projection
*  explain how to calculate the margin

$$
\require{cancel}
$$

ok, so, I was always been a bit confused about the fact that, by calculating the weights and biases how its going to help
us in classifying the data points? 
but the answer is apprent by simply looking and playing with our formula of finding the predicted y :-
$$ w^Tx + b = y(x) \tag{1}$$
**add the visualization**
we can also define this plane by using a normal vector( a vector which is perpendicular to a surface ) and a point from which we want our hyperplane to pass through.

we can calculate our normal vector by considering 2 points that are 'on' the hyperplane i.e, y(x_0) = 0 and y(x_1) = 0 and then find the vector which is perpendicular to these points. i.e,

let $$ x_0 $$, $$ x_1 $$ be the points that are 'on' the hyperplane. also, let there be a vector $$ \hat{x} = x_0 - x_1$$ now,
if there is a vector whose dotproduct with our $$ \hat{X} $$ yields 0, then, that vector will be our desired normal vector.

using some simple deduction we know that,

$$ w^T(\hat{x}) = 0$$
$$ w^T({x_0 - x_1}) = 0$$

so, according to our above defination of normal vector, our desired normal vector is going to be: 

$$ n = \frac{w}{||w||}$$

Now, let's observe our visualization a bit more... we can see that by moving our data points to the left/right of our hyperplane, the sign of $$ y(x) $$  changes.. which is exactly what we use to assign the class to our data point. this behaviour is also the reason why we sometimes call class A as '+'ive class and class B as '-'ive class respectively.
In more mathy terms, a hyperplane creates a division in our vector space which is known as a **halfspace**, ( because it devides our space into to halfs '+'ive and '-'ive as described earlier.)

now with that gets out of our way, we can finally dive into our main topic which is the reason why are we studying hyperplanes, which is to find a hyperplane which seprates our classes most optimally. 
note: here, we are only focusing on the data which are linearly seprable (i.e, which can easily be seprated by drawing a line/plane etc between the 2 groups of classes.)

so,in order to do that, we need to find a hyperplane, which maximize the perpendicular distance from the hyperplane to all the data points in such a way that gives the least **missclassification error**. and to calculate this distance, we simply needs to take a dot product between our point and the normal vector.

before finding the distance there is one more thing we need to derive first which do come handy later on. 

let $$ x_0 $$ be the point that is on the hyperplane

which means,
$$ w^Tx_0 + w_0 = y(x) = 0 $$
$$ w^Tx_0 = -w_0 \tag{2}$$

Now, lets get back to finding the distance from an arbitrary point $$ x $$ to our hyperplane.
so, in order to calculate the distance from our point 'x' onto our hyperplane,we simply take the dot product of our normal vector 'n' to $$ \bar{x} $$ i.e,

$$ 
dist = | \bar{x} * n |  \tag{3}\\
 where, \quad \quad
\bar{x} = x_0 - x
$$ 

$$| \bar{x}*n | = \frac{w^T(x_0 - x)}{||w||} = \frac{w^Tx_0 - w^Tx}{||w||} $$
$$ \frac{-w_0 - w^Tx}{||w||} = \frac{y(x)}{||w||} \tag{using (1) and (2)} $$

i.e, the distance from an arbitrary point to our hyperplane is:
$$ dist = \frac{y(x)}{||w||} \tag{4}$$
which is also known as the **margin**.

Now, just to get the complete picture, lets calculate the distance from origin to our hyperplane, here, we will use (3) and assume x = 0 (zero vector) respectively. i.e,
$$| \bar{x}*n | = \frac{w^T(x_0 - x)}{||w||} = \frac{w^Tx_0 - w^T0}{||w||}\\
 = \frac{w^Tx_0}{||w||} = \frac{ w_0}{||w||} \tag{ using (2) }$$


 which means, the perpendicular distance from origin to our hyperplane is :
 $$ \frac{ w_0}{||w||} \tag{5} $$

so, just to recap what we have derived so far:-

our normal vector :-
$$ \frac{w}{||w||} $$

distance from a point $$x$$ to hyperplane:
$$  \frac{y(x)}{||w||} \tag{using(4)}$$

distance from a origin $$0$$ to hyperplane:
$$ \frac{ w_0}{||w||} \tag{using(5)} $$

now we need to find a hyperplane which correctly classifies our data which means, if we assume $$ t_n $$ to be the class of data point 'n'. then, correctly classifying our data means $$ t_ny(x_n) > 0 $$ ( think about it for a min.) Thus the distance of a point x_n to the decision surface is given by :-

$$\frac{t_ny(x_n)}{||w||} = \frac{t_n(w^T(x_n) + w_0)}{||w||}$$

using all that, we can now form our optimization problem:-

$$
 \argmax_{w, b}  \left \{  { \frac{1}{||w||} \min_n  [ {t_ny(x_n)}  ]} \right \}  \tag{6}
$$

now, I know it looks kinda scary but hang in there, you will understand everything in just a bit. 

lets deconstruct this objective function:-

\$$ \min_n  [ {t_ny(x_n)}  ]} $$ :- this function finds the points with minimum distance from our hyperplane which is exactly what we need to find our margin. this function achieves that by minimizing the distance from our hyperplane with respect to $$ n $$ which is written in our in (6) we also note that, $$ 1/||w|| $$ is outside of this minimization over $$n$$ because ||w|| is a constant which means it doesn't depends on our $$ n $$ points


\$$ \argmax_{w, b}  \left \{  { \frac{1}{||w||} margin } \right \} $$ : now that we have found our margin, we can then find our weights and biases which maximize this margin. in other words, by maximizing the margin we will get our optimal seperating hyperplane because that hyperplane will be farthest from our 

although, this is a valid optimization problem but it a really hard to actually solve it. so, instead of optimizing this problem we will reformulate a similar problem by observing that, by rescaling 'w' and b by a constant $$ \kappa $$ doesn't affect the distance between an arbirary point $$ x_n $$ and our decision surface.

which means, if we assume that the distance from the closest point is a constant $$ \kappa $$ :-
$$ t_n(w^T(x_n) + w_0 ) = \kappa \\
\qquad =\frac{1}{\kappa}{t_n(w^T(x_n)+ w_0)}  = \frac{\cancel{\kappa}}{\cancel{\kappa}} = 1 \\
\\  \qquad \\
 t_n(w^T(x_n) + w_0) = 1 \tag{7}
$$

which also means that all the points that are on and outside the margin follows :-
$$  t_n(w^Tx(x_n) + w_0) \geq 1, \qquad \qquad {n = 1,......N} \tag{8}$$

this is also known as the canonical representation of the decision hyperplane. this constraint said to be *active* for the data points that are 'on' the margin  because then, the eqality holds for those data points and *inactive* for the rest of the data points.
without getting too much into the details but the points for which this constraints are active are said to be the support vectors and by defination there will always be at least 2 active constraints after we maximize our margin. now, using all of this, we can reformulate our objective function as :-

$$
 \argmax_{w, b}  \left \{  { \frac{1}{||w||} \min_n  [ {t_ny(x_n)}  ]} \right \} = \argmax_{w,b} \left \{ \frac{1}{||w||} \right \}  \tag{ using (7)}

$$
which we can reformulate as,
$$
\argmin_{w, b}  \frac{1}{2}{||w||^2} \tag{9}
$$
here, the factor 1/2 is added for later convenience. this optimization problem belongs to a family of optimization problem called *quadratic programming problem* because here, our objective function is quadratic and our constraints (8) are linear. although we can solve this optimization problem using any black-box QP Solver but instead in the next post we will take a look at another reformulation ( specifically its dual ). why? because, form where this formualtion stands, it has alot of problem which prevents us from using it in any real world situation which is why vladimer vapnik suggested a ground breaking method known as **S**upport **V**ector **M**machine which is what everyone uses untill recently.

before getting into SVM lets take a look at why this technique fails.
## problems with this technique

* this technique is sensitive to outliers. ( add gif) f1 : origal data f2: add outlier f3: recalculate our hyperplane and show the result
* yield no solution in the case of overlapping classes (which is common) (add gif) f1: original data f2: show overlapped data and show no hyperplane and add an error clause
* unable to classify nonlinearly seperable classes ( which is really important from the practical point of view, because almost all the real world classes are not linearly seperable) (add gif) f1: original data f2: show non-linear data and show no hyperplane and add an error clause

so, I hope you enjoyed this post and I will see you in the next one...
Have a beautiful day ^_^.