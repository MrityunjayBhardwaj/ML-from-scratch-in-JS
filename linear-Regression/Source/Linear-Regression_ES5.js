/**
 * TODO:
 * Use es6 classes
 *
 * DONE:
 * decided
 * 1. implement the optimization step explicitly and use only square loss functions
 * 4. clean up the code. ( i.e, remove unused variables and dependencies etc.)
 * 5. check if preTrained Weights are tf.tensor object
 *
 *
 * NOTE:
 * 2. convert it into prototypes ( I can't convert it into prototypes because then i can access the private variables,
 * i need to expose it pubically in order to access it from the prototype methods which leads to security vulnaribility
 * also, the prototype patter can be a bit hard to grasp for the beginner js developer thats why i am sticking to creating function within the closure
 * which allows me to access the private variables and its the most cleanest thing ( only next to the e6 classes which is just a syntactical suger which ofcourse i don't want to use (because its dillusional :P)) )
 * at the expense of performance.... SO, TLDR: I am trading performance gain from prototype inheritence pattern for the sake of security.)
 *
 * 3. decide weather to use WebPack (for modules) or vanilla JS. ( i think i should)
 * > NO! beacuse not all browsers support es6 Modules and it order to make it work for these browser we need to use
 * special frameworks like bable and webpack (which can be a little hurdel ) + es6 Modules only work when setup a server
 * also, its not such a huge project with many interdependencies + the main agenda of this project is learning with as little
 * technical details as possible.
 */

/**
 * this function checks if the given input is a tf.tensor object or not
 * @param {Any} t
 * @return {boolean}
 */
function isTensor(t) {
  return t instanceof tf.Tensor;
}

// Only for Learning Purposes
/**
 *@param {tf.tensor} preTrainedWeights load in the pre trained weights ( must be of type tf.tensor)
 */
function LinearRegressionES5(preTrainedWeights = null) {
  {
    if (!isTensor(preTrainedWeights))
      throw new Error(
        'preTrainedWeights must be a tf.tensor object but given ' +
          typeof preTrainedWeights
      );
  }

  const _model = {
    weights: preTrainedWeights || null,
  };
  /**
   * @summary exporting our Model as a json for potential future usecase.
   * @return {string} json string
   */
  this.exportModel = function () {
    if (!_model.weights) {
      throw new Error('the weights are not trained yet!');
    }

    return JSON.stringify({ weights: _model.weights.arraySync() });
  };

  /**
   * @summary returns our model weights as a tf.tensor object
   * @return {tf.tensor} model weights
   */
  this.getWeights = function () {
    return _model.weights;
  };

  /**
   * Assign a new Weight vector to our model
   * @param {tf.tensor} w new weight vector
   */
  this.setWeights = function (w) {
    if (!isTensor(w))
      throw new Error(`input weights must be a tf.tensor object. `);

    _model.weights = w;
  };
}

/**
 * @param {object} data x and y data values,formatted like this:- {x: dataX, y : dataY}
 * @param {object} params a collection of parameters useful for training our model
 * @param {number} params.nEpoch number of training iterations;
 * @param {number} params.threshold terminate the training once loss < threshold
 * @param {number} params.learningRate weights our graidents at each step
 * @summary this funciton takes 2 arguments and spits out the best possible wights vector , first argument is just an data Inut matrix
 * @return {obect} this (for function chaining)
 */
LinearRegressionES5.prototype.fit = function (data, params = {}) {
  /*
   * initializing all the parameters
   */
  const { nEpoch = 100, threshold = 1e-3, learningRate = 1e-4 } = params;

  /*
   * concatinating constant ones for bias weights
   */
  const dataXWithBias = data.x.concat(
    tf.ones([data.x.shape[0], 1]),
    /* axis */ 1
  );

  /*
   * The gradient Descent Algorithm
   */
  let cWeights = this.getWeights() || tf.randomNormal([data.x.shape[1], 1]);
  for (let epoch = 0; epoch < nEpoch; epoch += 1) {
    // our prediction using the current weight values
    const yPred = dataXWithBias.matMul(cWeights);

    // checking for convergence using MSE Loss
    const cLoss = yPred
      .sub(data.y)
      .pow(2)
      .sum()
      .mul(1 / 2)
      .flatten()
      .arraySync()[0];
    if (cLoss < threshold) return cWeights;

    // updating our weights by moving in the direction of steepest descent
    const gradient = yPred.sub(data.y).transpose().matMul(dataXWithBias);
    cWeights = cWeights.sub(gradient.mul(learningRate));
  }

  /*
   * assigning weights to our model object
   */
  this.setWeights(cWeights);

  return this;
};

/**
 * @summary Given the test data, this function spits out the predicted value of Y using our pretrained weights.
 * @param {tf.tensor} testDataX
 * @return {tf.tensor}
 */
LinearRegressionES5.prototype.test = function (testDataX) {
  return tf.matMul(testDataX, this.getWeights());
};
