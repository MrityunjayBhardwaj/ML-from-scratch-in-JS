
/**
 * 
 * @param {object} derivatives {jacobian: tf.tensor, hessian: tf.tensor}
 * @summary given derivatives and initial values this function calculate local optima
 * @returns 
 */
function newtonRaphson(params){
    const newWeights = params.jacobian.div(params.hessian);
    return optimize({newWeights});
}