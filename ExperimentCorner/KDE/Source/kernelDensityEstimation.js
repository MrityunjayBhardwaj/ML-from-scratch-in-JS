function KDE() {
  this.model = {};

  this.kernelFn = function(type) {
    switch (type) {
      case "parzen":
        return function(x_prime, x, params={h:0.2}) {
          // hyperparameters
          const h = params.h;
          const N = x.shape[0];
          const D = x_prime.shape[1];

          // calculating K : no. of neighbours
          const u = x_prime.sub(x).div(h);
          const k = tf.clipByValue(
            tf
              .abs(u)
              .sub(1 / 2)
              .mul(1)
              .mul(100),
            0,
            1
          );
          const K = tf.sum(k);

          // calcualting the p(x_prime)
          const probability = tf.mul((1 / N) * Math.pow(h, D), K);

          return probability;
        };

      case "radial":
        return function(x, x_prime, params = {h: 0.5,}) {
          const h = params.h;
          const N = x_prime.shape[0];
          const gaussianKernel = tf.sum(
            tf.mul(
              1 / (2 * Math.PI * h ** 2) ** (1 / 2),
              tf.exp(
                tf.mul(
                  -1,
                  x.sub(x_prime)
                    .pow(2)
                    .div(2 * Math.pow(h, 2))
                )
              )
            )
          );

          const probability = tf.mul(1 / N, gaussianKernel);

          return probability;
        };

      default:
        break;
    }
  };

  this.train = function(data = { x: [], y: [] }) {};

  this.test = function(x, x_prime, params) {return this.kernelFn('radial')(x, x_prime, params)};
}

// how kernel density estimation works?

/**

    p(x) = K-Neighbours/(total number of neighbours in the window * Volume of the window);

 */
