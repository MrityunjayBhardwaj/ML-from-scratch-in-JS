
# Lagrange Multipliers

table of content:
* outline
* why should i care?
* finding minima by hand
* automate the process using mathematics

Suppose, you want to buy a car. but you are a bit confused which one to buy as there are alot of choices... but your wife knowns you pretty well and she gave you this function which tells you which car you'll like alot and which car you don't according to the price of the car...

// Add a graph which also shows the look of the car along with the curve

here, the 'x' axis denotes the price and 'y' axis denotes your preference

as you can see, this function is non-linear which makes sense because there might be some car who's price is high but you did'nt like its design or something... 

now, if you have infinit budget then you don't even have to look at the price of the car, you can just choose a car with the highest peak.. soo go ahead and set the line which shows the 'y' coordinate to the highest possible value... 

// add viz with adjustable contour line DONE!

So, the line you just adjusted is called a contour line which just slice the function by setting the function value to a constant i.e, $f(x) = c$. 

Now, you consult with you wife again and unsurprisingly the conversation goes something like this:

`ME: hey, thankyou for that pereference function earlier, i finally know which car i am gonna buy`

`Wife: Oh that awesome, how much does it cost?`

`ME: its $50 million`

`Wife: 50... what!!... what are you Bill Gates or something?`

After that "friendly" conversation with you wife, she gave you the budget and 2 bumps on your head. \
You see, in real life we almost always have some constraints which we need to satisfy and try to find the best possible values of our function without volating these constraints.

So, you went back to your graphic viz thingy and plot the function along with the feasible region ( the budget of your car. denoted in red color below) :

// add simple plot with constraint function

now, the purple hilighted part of our preference function represent the preferences of the cars which you can afford. which means, now you need to find the best possible value while remaining within this purple region... so you take that contour line again and search the best value of the preference function without violating the constraints ( coz ya would'nt wanna mess with ya wife again..)

// add viz of the previous plot + adjustable contour line DONE!

now, that you have found the best car which is still in your budget you went back to your wife with this result and guess what, she liked it too! 


What you did above by manually searching for the best possible car without violating your constraints is exactly the reason why we need optimimzation for.... **to make optimal decisions**.

imagine instead of a 1 dimensional function, you have a 10 or a 100 dimensional function ( for example, in this case suppose the openion/preference of the entire family matter not just yours, here, you can image a function with 5-6 dimensions each representing the preference function of all the members of the family)... you can't possibly be thinking of finding the optimal values manually right? that is where optimization comes in. its just a mathematical framework which can give you the most optimal value given the functions and constraints.


#

Now before moving forward, lets make ourselves a bit familier with the termonologies...

so, the preference function of which we were trying to find the optimal value of, is known as **objective function**... because our objective is to minimize or maximize it right?

the process by which we are finding the best possible value of our objective function without violating the constraints are known as **optimization**.. its same as adjusting that contour line...

after performing optimization, the best value we get is known as **optimal value** and the best point is known as **optimal point**.

points which satisfy all of our constraints are known as feasible points like those point that are hilighted in pruple (see fig 5)
// add image defining the optimal value and the optimal point.


now, with that gets out of the way, we can finally focus on converting intuition of finding the optimal into mathematics in order to automate the entire searching process....


Suppose, you have variables 'x' and 'y' and a function $f$ which takes this 2 variables as input and spits out a single value for e.g. $f(x_1,x_2) = x_1 + x_2$ now, let $x_1 = 2$ and $x_2 = 3$ then $f(2,3) = 5$...this type of function is known as multi-variate function for obvious reasons \
In context of our car example above section, we can think of $x_1$ as fuel consumption of the car maybe?

Anyways, so, if we plot this function it would look something like this...

// add a 3d plot showing this 2d function

and just like before, here, we are adding a constraint function $g(x,y)$ now, plotting both the constraint as well as objective function we get:-


again, just like before we get a contour plot gizmo but this time, instead of maximizing we are going to be minimizing.. so, grab that slider and find the minimum value of our objective function $f(x,y)$ which still satisfying our constraint function $g(x,y)$ ....

// add interactive viz with adjustable contour line 
// also, add the plot of the sliced $f(x,y)$

now that we have finally mastered the art of finding the optimal point we can go ahead and let math do the work for us...

As you might have noticed in the previous visualizer, we have also show the points that satisfies the constraint and is a part of thata sliced function. \
Now, if we also observe the gradient vector of objective function and constraint function at these feasible points:-

// use the previous visualizer + visualize the tangent line on the feasible points...

we can clearly see that at optimal points the gradient of both the objective function and constraint function points at exactly the same direction and differ only by the magnitude...\
we can leverage this observation to construct the mathematical equivalent of searching for optimal point...

Mathematically, we want the feasible point where, the gradient of the objective function points to the same direction as the gradient of the constriant function and differ only by the scaling...

$$ \nabla f  = \lambda \nabla g$$

$$ f_x(x_0, y_0) = \lambda g_x(x_0, y_0) $$
$$ f_y(x_0, y_0) = \lambda g_y(x_0, y_0) $$

where, $(x_0, y_0)$ can be the coordinates of our optimal point and $f_x$ is the derivative of function w.r.t $x$ and $f_y$ is the derivative of the function w.r.t $y$ respectively...

that scaling factor is known as **lagrange multiplier**.

Now, lets use this descovery to find the optimal point for this optimization problem without manually searching for optimal point..

// TODO: derivation...

if you are able to reach here... then congratulations you have mastered the art of lagrange multipliers... you can test your knowledge by solving the following exercises...

Have a beautiful day ^_^