
function KNN(){
    model= {
        params : {
            threshold : 0.4,
        },
        data : null
    };

    /**
     * @param { object } data input => {x : features/predictors , y: one-hot encoded output vector }
     * @param { Number } threshold 
     * @param { Number } K number of neighbours 
     */
    this.train = function(data,threshold = 0.5){
        const inputX = data.x;
        const inputY = data.y; 

        model.data = data; 
    }
    this.classify = function(testDataX){
        // input must be a test set 

        const categoryArray = testDataX.arraySync().map(
            ( currPt ) => {

                const tfCurrPt = tf.tensor(currPt);
                const trainDataX = model.data.x;
                
                const currPtVec = tfCurrPt.tile( [testDataX.shape[0], 1 ]);

                const currPtNorm = tf.norm( tfCurrPt.transpose() );
                const dataNorm = tf.norm( trainDataX, 'euclidean',axis=1);

                const normMatrix = tf.fill( currPtNorm.flatten().arraySync(), dataNorm.shape );

                const nearnest = tf.pow( tf.sub( normMatrix, dataNorm ), 2 );

                // fetching the nearest data points
                const b = sortAB( nearnest.flatten().arraySync(), trainDataY );

                const topK = b.slice([0,0],[K, -1]);

                // calculating the relative frequency table of all the neighbour classes
                const rf = topK.transpose().matMul( tf.ones( [neighY.shape[0],1] ) );

                const xCategory = rf.flatten().arraySync().indexOf(rf.max().flatten().arraySync());
               
                // returns the one hot encoded vector of the best possible category
                return rf[xCategory]; 
                
            }
        )

        return categoryArray ;

    }
}