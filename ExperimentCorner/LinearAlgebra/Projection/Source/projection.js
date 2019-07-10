
/**
 * 
 * @param {object} A A is a subspace.
 * @param {object} b b is a vector.
 * @summary this function project vector 'b' onto a subspace 'A'.
 * @returns returns the projected vector of 'b'.
 */
function project(A,b){
    // (A^TA)^-1 * (A^T*b);
    return tf.matMul(A.transpose(),A).matMul( tf.matMul(A.transpose, b) )
}