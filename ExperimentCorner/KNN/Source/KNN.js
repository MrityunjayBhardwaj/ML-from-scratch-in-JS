
function KNN(){
    model = {
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
    this.train = function(data,K){
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

        const categoryArray = testDataX.arraySync().map(
            ( currPt ) => {

                // init
                const trainDataX = model.data.x;
                const nNeigh     = model.params.k;
                const tfCurrPt   = tf.tensor(currPt);

                // calculating the norms.
                const currPtNorm = tf.norm( tfCurrPt.transpose() );
                const dataNorm   = tf.norm( trainDataX, 'euclidean',axis=1);
                const normMatrix = tf.fill( currPtNorm.flatten().arraySync(), dataNorm.shape );

                // fetching the nearest data points
                const nearnest = tf.pow( tf.sub( normMatrix, dataNorm ), 2 );
                const neighY   = sortAB( nearnest.flatten().arraySync(), trainDataY );

                // fectch the classes of K Nearest Neighbours
                const kNeighClasses = neighY.slice([0,0],[nNeigh, -1]);

                // calculating the relative frequency table of all the neighbour classes
                const rf = kNeighClasses.transpose().matMul( tf.ones( [neighY.shape[0],1] ) );
                const xCategory = rf.flatten().arraySync().indexOf(rf.max().flatten().arraySync());
               
                // returns the one hot encoded vector of the best possible category
                return rf[xCategory]; 
            }
        )

        return categoryArray ;

    }
}