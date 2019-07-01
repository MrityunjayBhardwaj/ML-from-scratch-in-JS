
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

                // calculating the norms.
                const currPtNorm = tf.norm( tfCurrPt.transpose() );
                const dataNorm   = tf.norm( trainDataX, 'euclidean',axis=1);
                const normMatrix = tf.fill( dataNorm.shape, currPtNorm.flatten().arraySync()  );

                // fetching the nearest data points
                const nearnest = tf.pow( tf.sub( normMatrix, dataNorm ), 2 );
                const neighY   = tf.tensor( sortAB( nearnest.flatten().arraySync(), trainDataY )[1] );

                // fectch the classes of K Nearest Neighbours
                const kNeighClasses = neighY.slice([0,0],[nNeigh, -1]);

                // calculating the relative frequency table of all the neighbour classes
                const rf = kNeighClasses.transpose().matMul( tf.ones( [kNeighClasses.shape[0],1] ) );
                const xCategory = rf.flatten().arraySync().indexOf( rf.max().flatten().arraySync()[0] );
               
                // returns the one hot encoded vector of the best possible category
                return kNeighClasses.slice([xCategory,0],[1,-1]).flatten().arraySync(); 

            }
        )

        return categoryArray ;

    }
}