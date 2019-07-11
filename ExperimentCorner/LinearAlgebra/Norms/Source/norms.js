/**
 * 
 * @param {object} x  input must be a tf.tensor object of dim nx1
 * @param {number} p  p-value for our p-norm
 * @summary calculates the pNorm of the given vector
 */
function pNorm(x,p){
    return tf.pow(tf.sum(tf.pow(tf.abs(x), p), axis=0), 1/p);
}