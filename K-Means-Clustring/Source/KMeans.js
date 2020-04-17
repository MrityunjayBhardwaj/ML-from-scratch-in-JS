function KMeans(){

    this.model = {
        mean: null,
        noOfClusters: 0,

    },
    this.train = function(data, k=2,maxItr=100){

        // using EM algorithm for calculating K-Means Clustring

        let means = tf.randomUniform([k, data.shape[1]]).arraySync();
        let clusterAssign = tf.zeros([data.shape[0], k]); // initially, make all the data points belongs to cluster 0


        const tollerance = 0.01;

        let prevError = 0;
        let errorDif = 0;

        let iteration = 0;

        while(iteration < maxItr){

            // Expectation Block: freeze means
            let distFromAllMeans = 0;
            for(let i=0; i<k; i++){

                const currMeanCenteredData = tf.pow(data.sub(means[i]),2);

                const currDistFromMean = tf.norm(currMeanCenteredData, norm=2, axis=1, keepDims=1);

                if (i === 0){
                    distFromAllMeans = currDistFromMean;
                    continue;
                }
                distFromAllMeans = distFromAllMeans.concat(currDistFromMean, axis=1);
            }

            // calculating which cluster a point belongs to.... 
            const tileShape =(new Array(distFromAllMeans.shape.length + 1)).fill(1);
            tileShape[0] = k;

            let distTensor = distFromAllMeans.expandDims().tile(tileShape);
            let otherDist = distFromAllMeans.expandDims();

            distTensor = distTensor.sub(otherDist.transpose())

            clusterAssign = tf.clipByValue(distTensor.mul(1000000000), -1, 1).sum(axis=2).equal(k-1).mul(1).transpose();

            // calculate error
            const currError= distFromAllMeans.sum().flatten().arraySync()[0];

            errorDif = (prevError - currError)**2;
            prevError = currError;

            // if the difference between the previous distance and the current distance is below the tollerance then stop this loop 
            //otherwise, calculate new mean and repeat the process untill converges
            if (errorDif < tollerance)break; 

            // Maximization Block: freeze clusterAssign
            const clusterSize = clusterAssign.sum(axis=0); // how many points belongs to each cluster
            const clusterSum  = data.expandDims().tile(tileShape);

            means = clusterSum.mul(clusterAssign.expandDims().transpose()).sum(axis=1).div(clusterSize.expandDims().transpose()).arraySync()

            iteration++;
        }


        model.mean = means;
        model.noOfClusers = k;

        return { mean: means, cluster: clusterAssign};
    },

    /**
     * 
     * @param {tf.tensor} dataX tensor
     * 
     * @description given the data this function finds the cluster which this data belongs to...
     */
    this.test = function(dataX){
        const clusterAssign = 0;

        // calculating which cluster a point belongs to.... 
        const tileShape =(new Array(k)).fill(1);
        tileShape[0] = k;

        let distTensor = distFromAllMeans.expandDims().tile(tileShape);
        let otherDist = distFromAllMeans.expandDims();
        otherDist = otherDist.reshape([distFromAllMeans.shape[0], 1, distFromAllMeans[1]]);

        distTensor = distTensor.sub(otherDist);
        clusterAssign = tf.clipByValue(distTensor.mul(10000), -1, 0).add(1).reshape(distFromAllMeans.shape);

        return clusterAssign;
    }

}