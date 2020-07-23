// import { Tensor } from '../../dependency/tensorflowJS/tf'

/**
 * this function checks if the given input is a tf.tensor object or not
 * @param {Any} t
 * @return {boolean}
 */
export function isTensor(t) {
  return t instanceof Tensor;
}
