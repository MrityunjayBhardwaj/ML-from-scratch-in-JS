// import * as tf from '../../dependency/tensorflowJS/tf.js';
import { isTensor } from './linRegUtils.js';

/**
 * @class
 */
export default class LinearRegression {
  /**
   * @param {object} params a collection of parameters useful for training our model
   * @param {boolean} params.useBias Weather to calculate the intercept of the model or not
   * @param {number} params.nEpoch number of training iterations;
   * @param {number} params.threshold terminate the training once loss < threshold
   * @param {number} params.learningRate weights our graidents at each step
   * @param {number} params.verbose prints useful info for debugging purposes
   * @param {LinearRegression~callback} callback run this script after each epoch
   * @param {tf.tensor} preTrainedWeights load in the pre trained weights ( must be of type tf.tensor)
   */
  constructor(params = {}, callback = () => {}, preTrainedWeights = null) {
    if (preTrainedWeights && !isTensor(preTrainedWeights))
      throw new Error(
        `preTrainedWeights must be a tf.tensor object but given ${typeof preTrainedWeights}`
      );

    this._model = {
      weights: preTrainedWeights || null,
      params: {
        useBias: params.useBias || false,
        nEpoch: typeof params.nEpoch === 'number' ? params.nEpoch : 100,
        threshold:
          typeof params.threshold === 'number' ? params.threshold : 1e-3,
        learningRate:
          typeof params.learningRate === 'number' ? params.learningRate : 1e-4,
        verbose: params.verbose || false,
      },
      callback: callback,
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
  async fit(x, y) {
    /*
     * initializing all the parameters
     */

    console.log(this._model.params);
    const {
      useBias,
      nEpoch,
      threshold,
      learningRate,
      verbose,
    } = this._model.params;

    /*
     * concatinating constant ones for bias weights
     NOTE: if useBias is false then it is assumed that data is centered so don't concat
     */
    const dataXWithBias = !useBias
      ? x
      : x.concat(tf.ones([x.shape[0], 1]), /* axis */ 1);

    /*
     * The gradient Descent Algorithm
     */
    let cWeights = this.getWeights() || tf.zeros([dataXWithBias.shape[1], 1]);

    console.log('first usedweights', cWeights.print());

    let epoch = 0;

    while (true) {
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

      if (verbose && epoch % (nEpoch / 100) === 0)
        console.log(
          `${epoch}) weights: ${cWeights.arraySync()} | loss: ${cLoss}`
        );

      // updating our weights by moving in the direction of steepest descent
      const gradient = yPred
        .sub(y)
        .transpose()
        .matMul(dataXWithBias)
        .transpose();
      cWeights = cWeights.sub(gradient.mul(learningRate));

      // assigning the updated weights to our model object
      this._model.weights = cWeights;

      // invoking the callback function
      await this._model.callback(epoch, cLoss, cWeights, yPred);

      console.log('callback finished');

      // stopping criterion
      if (cLoss <= threshold || epoch > nEpoch) {
        break;
      }

      epoch++;
    }
  }

  /**
   * Given the test data, this function spits out the predicted value of Y using our pretrained weights.
   * @param {tf.tensor} testDataX
   * @return {tf.tensor}
   */
  test(testDataX) {
    console.log('testing...', this.getWeights().print(), testDataX.shape);
    const { useBias } = this._model.params;
    const dataXWithBias = !useBias
      ? testDataX
      : testDataX.concat(tf.ones([testDataX.shape[0], 1]), /* axis */ 1);
    return dataXWithBias.matMul(this.getWeights());
  }
}

/**
 * @callback LinearRegression~callback
 * @param {number} epoch current Epoch of our training loop
 * @param {number} cLoss current loss
 * @param {tf.tensor} weights current weight tensor
 * @param {tf.tensor} yPred current predicted output
 * @returns it returns a promise
 */
