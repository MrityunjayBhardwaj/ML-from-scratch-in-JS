
/**
 * 
 * @param {object} A A is a subspace.
 * @param {object} b b is a vector.
 * @summary this function project vector 'b' onto a subspace 'A'.
 * @returns returns the projected vector of 'b'.
 */
function project(A,b){
    // (A^TA)^-1 * (A^T*b);
    const part1 = tf.matMul(A.transpose(),A);

    const invP1 =  tfpinv(part1);
    const fac = invP1.mul( tf.matMul(A.transpose(), b) );

    return A.matMul(fac);
}

