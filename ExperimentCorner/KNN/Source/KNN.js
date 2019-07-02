
function KNN(){
    const model = {
        params : {
            threshold : 0.4,
            k : 1
        },
        data : null
    };

    /**
     * @param { object } data input => {x : features/predictors , y: one-hot encoded output vector }
     * @param { Number } threshold 
     * @param { Number } K number of neighbours 
     */
    this.train = function(data,K=1){
        const inputX = data.x;
        const inputY = data.y; 

        model.data = data; 
        model.params.k = K;
    }
    /**
     * @param testDataX takes a tf.tensor
     * @description classify input tensor using KNN
     * @returns tf.tensor of one-hot enconNeighded predicted class.
     */
    this.classify = function(testDataX){
        // input must be a test set 

        let categoryArray = testDataX.arraySync();
        categoryArray = categoryArray.map(
            ( currPt ) => {

                // init
                const trainDataX = model.data.x;
                const trainDataY = model.data.y;
                const nNeigh     = model.params.k;
                const tfCurrPt   = tf.tensor(currPt);
                const nCategory  = trainDataY[0].length;

                // calculating the norms.
                const currPtNorm = tfCurrPt.expandDims(1).transpose().tile([2,1]).norm('euclidean',axis=1).slice([0],[1])

                const dataNorm   = trainDataX.norm('euclidean',axis=1);
                const normMatrix = tf.fill( dataNorm.shape, currPtNorm.flatten().arraySync()  );

                // dist b/w pts and cPt
                const nPt = tfCurrPt.reshape([1,2]).tile([trainDataX.shape[0],1]);
                const dist = trainDataX.sub(nPt).pow(2);
                const finNorm = dist.norm('euclidean',axis=1);


                // fetching the nearest data points
                // const nearnest = tf.abs( tf.sub( normMatrix, dataNorm ) );
                const neighY   = tf.tensor( sortAB( finNorm.flatten().arraySync(), trainDataY )[1] );

                // fectch the classes of K Nearest Neighbours
                const kNeighClasses = neighY.slice([0,0],[nNeigh, -1]);

                // calculating the relative frequency table of all the neighbour classes
                const rf = kNeighClasses.transpose().matMul( tf.ones( [kNeighClasses.shape[0],1] ) );
                const xCategory = rf.flatten().arraySync().indexOf( rf.max().flatten().arraySync()[0] );
                
                // console.log(xCategory);
                // returns the one hot encoded vector of the best possible category
                return Array(nCategory).fill(0).fill(1, xCategory,xCategory+1);
                // return kNeighClasses.slice([xCategory,0],[1,-1]).flatten().arraySync(); 

            }
        )

        return tf.tensor(categoryArray) ;

    }
}