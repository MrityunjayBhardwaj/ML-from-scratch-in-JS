// import * as tf from '../../dependency/tensorflowJS/tf.js';
import { isTensor } from './linRegUtils.js';
/**
 * TODO:
 * 
 * DONE:
 * decided
 * 1. implement the optimization step explicitly and use only square loss functions
 * 4. clean up the code. ( i.e, remove unused variables and dependencies etc.)
 * 5. check if preTrained Weights are tf.tensor object
 * 6. Use es6 classes
 * 7. Make it visualization friendly
 * Try es6 Modules
 * Test the code
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
 * Linear Regression Class
 */
export default class LinearRegression {
  /**
   * @param {object} params a collection of parameters useful for training our model
   * @param {boolean} params.useBias Weather to calculate the intercept of the model or not
   * @param {number} params.nEpoch number of training iterations;
   * @param {number} params.threshold terminate the training once loss < threshold
   * @param {number} params.learningRate weights our graidents at each step
   * @param {number} params.verbose prints useful info for debugging purposes
   * @param {tf.tensor} preTrainedWeights load in the pre trained weights ( must be of type tf.tensor)
   */
  constructor(params = {}, preTrainedWeights = null) {
    if (preTrainedWeights && !isTensor(preTrainedWeights))
      throw new Error(
        `preTrainedWeights must be a tf.tensor object but given ${typeof preTrainedWeights}`
      );

    this._model = {
      weights: preTrainedWeights || null,
      params: {
        useBias: params.useBias || false,
        nEpoch: (typeof params.nEpoch === "number")? params.nEpoch : 100,
        threshold: (typeof params.threshold === "number")? params.threshold : 1e-3,
        learningRate: (typeof params.learningRate === "number")? params.learningRate : 1e-4,
        verbose: params.verbose || false,
      },
    };
    /**
     * exporting our Model as a json for potential future usecase.
     * @return {string} json string
     */
    this.exportModel = () => {
      if (!_model.weights) {
        throw new Error('the weights are not trained yet!');
      }

      return JSON.stringify({ weights: this._model.weights.arraySync() });
    };

    /**
     * returns our model weights as a tf.tensor object
     * @return {tf.tensor} model weights
     */
    this.getWeights = () => this._model.weights;

  }

  /**
   * this funciton takes 2 arguments and spits out the best possible wights vector , first argument is just an data Inut matrix
   * @param {tf.tensor} x dataX
   * @param {tf.tensor} y dataY
   * @return {obect} returns 'this' object itself
   */
  fit( x, y ) {
    /*
     * initializing all the parameters
     */
    const { useBias, nEpoch, threshold, learningRate, verbose } = this._model.params;

    /*
     * concatinating constant ones for bias weights
     NOTE: if useBias is false then it is assumed that data is centered so don't concat
     */
    const dataXWithBias = (!useBias)? x : x.concat(tf.ones([x.shape[0], 1]), /* axis */ 1);

    /*
     * The gradient Descent Algorithm
     */
    let cWeights = this.getWeights() || tf.zeros([dataXWithBias.shape[1], 1]);

    console.log("first usedweights", cWeights.print());
    
    for (let epoch = 0; epoch < nEpoch; epoch += 1) {

      // our prediction using the current weight values
      const yPred = dataXWithBias.matMul(cWeights);

      // checking for convergence using MSE Loss
      const cLoss = yPred
        .sub(y)
        .pow(2)
        .sum()
        .mul(1 / 2)
        .flatten()
        .arraySync()[0];
      if (cLoss <= threshold){
        console.log(`final loss:- ${cLoss} ${threshold}`);
        break;
      }

        if (verbose && epoch % (nEpoch/100) === 0)
          console.log(`${epoch}) weights: ${cWeights.arraySync()} | loss: ${cLoss}`)

      // updating our weights by moving in the direction of steepest descent
      const gradient = (yPred.sub(y).transpose()).matMul(dataXWithBias).transpose();
      cWeights = cWeights.sub(gradient.mul(learningRate/x.shape[0]));
    }

    /*
     * assigning weights to our model object
     */
    this._model.weights = cWeights;

    return this;
  }

  /**
   * Given the test data, this function spits out the predicted value of Y using our pretrained weights.
   * @param {tf.tensor} testDataX
   * @return {tf.tensor}
   */
  test(testDataX) {
    const { useBias } = this._model.params;
    const dataXWithBias = (!useBias)? testDataX : testDataX.concat(tf.ones([testDataX.shape[0], 1]), /* axis */ 1);
    return dataXWithBias.matMul( this.getWeights());
  }
}
