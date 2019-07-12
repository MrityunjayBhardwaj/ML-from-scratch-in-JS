/**
 * 
 * @param {object} x  input must be a tf.tensor object of dim nx1
 * @param {number} p  p-value for our p-norm
 * @summary calculates the pNorm of the given vector
 * @returns returns tf.scalar value. which represent the p-norm of the input vector
 */
function pNorm(x,p=1){
    return tf.pow(tf.sum(tf.pow(tf.abs(x), p), axis=0), 1/p);
}