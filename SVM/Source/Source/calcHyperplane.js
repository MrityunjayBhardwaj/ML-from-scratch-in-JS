function calcHyperplane(model, usrParams = {weights: tf.tensor([1]),bias: tf.tensor([0]), range:[{min: -5, max: 5},{min: -5, max: 5}] } ){

    let { 
      angle = 45,
      weights=tf.tensor([1]),
      bias=tf.tensor([0]),
      range=[{min: -5, max: 5},{min: -5, max: 5}],
      rndPt = [2, 1],
    } = usrParams;


    // if angle is given then calculate the weights accordingly
    if (!usrParams.weights && usrParams.angle != undefined){
      
      angle = (Math.PI/180)*usrParams.angle;
      weights = [ Math.cos(angle)*1, Math.sin(angle)*1];
    }

    // converting weights and biases as tf.tensor if it wasn't already
    weights = (Array.isArray(weights))? tf.tensor(weights) : weights;
    bias = (Array.isArray(bias) || typeof bias === 'number')? tf.tensor(bias) : bias;

    // TODO: make sure only tf.tensor will come through the code below is just a bad hack
    if (!(weights.shape && bias.shape) )throw new Error('invalid input type, weights and biases must be either Array or tf.tensor object!');
  
    
    // calcuating the hyperplane:-

    const division = 30;
    const inp0 = tf.linspace(range[0].min, range[0].max , division).flatten().arraySync();
    const inp1 = tf.linspace(range[1].min, range[1].max , division).flatten().arraySync();

    // TODO: implement this technique to all the algos in order to get the performance boost for free
    const inpMeshGridTensor = tf.tensor(meshGrid(inp0, inp1)).reshape([inp0.length*inp1.length, 2]);

    let outputTensor = 0;
    if (mySVM){
        outputTensor = model.test(inpMeshGridTensor);

    }else{
        outputTensor = tf.tensor( model.test(inpMeshGridTensor.arraySync()) );
    }

    const normalizedWeights = ((weights).div(tf.norm(weights)).transpose()).arraySync();

    const x0 = range[0];
    let unNormPt = [
    (- bias.flatten().arraySync()[0] - weights.flatten().arraySync()[0]*x0.min ),
    (- bias.flatten().arraySync()[0] - weights.flatten().arraySync()[0]*x0.max )
    ];
    const x1 = {
      min: unNormPt[0]/weights.flatten().arraySync()[1],
      max: unNormPt[1]/weights.flatten().arraySync()[1]
    };

    marginLength = 1;


    const db_right = [
        {x: x0.min, y: (unNormPt[0] + marginLength)/weights.flatten().arraySync()[1]},
        {x: x0.max, y: (unNormPt[1] + marginLength)/weights.flatten().arraySync()[1]},
    ];

    const db_left = [
        {x: x0.min, y: (unNormPt[0] - marginLength)/weights.flatten().arraySync()[1]},
        {x: x0.max, y: (unNormPt[1] - marginLength)/weights.flatten().arraySync()[1]},
    ];

    const extDB = [
        {x: x0.max, y: (unNormPt[1] + 100)/weights.flatten().arraySync()[1]},
        {x: x0.min, y: (unNormPt[0] + 100)/weights.flatten().arraySync()[1]}
    ]



    // TODO: make it tensor:-
    // const x0t = tf.tensor(range);
    // const x2 = weights.mul(x0t.transpose()).add(bias).div(weights).neg();
    // x2.print();


    const p0 = [x0.min, x1.min];
    const p1 = [x0.max, x1.max];

    // console.log(p0,p1)


    function proj(x, hyperplane){

        const u = tf.tensor(x).expandDims(1);
        hyperplane= tf.tensor(hyperplane);

        return u.sub( u.transpose().matMul(hyperplane).div(tf.norm(hyperplane).pow(2)).mul(hyperplane)).flatten().arraySync()
    }

    const projPlane = [ normalizedWeights[1] ,
                        normalizedWeights[0] 
    ];

    let projRndPt = proj(rndPt, projPlane);

    // let projRndPt = proj(rndPt, [normalizedWeights[1], normalizedWeights[0] ]  )
    // projRndPt = proj(rndPt, [normalizedWeights[1]* ( -( bias/( (tf.norm(tf.tensor(weights))).flatten().arraySync()[0] ) ) ), normalizedWeights[0]* ( -( bias/( (tf.norm(tf.tensor(weights))).flatten().arraySync()[0] ) ) ) ]  )

    projRndPt[0] = projRndPt[0] + normalizedWeights[1]*( -( bias/( (tf.norm(weights)).flatten().arraySync()[0] ) ) );
    projRndPt[1] = projRndPt[1] + normalizedWeights[0]*( -( bias/( (tf.norm(weights)).flatten().arraySync()[0] ) ) );

    const hyperplaneFac = bias.div(tf.norm(weights)).neg().flatten().arraySync()[0];
    return {

        heatMap : {x: inpMeshGridTensor.arraySync(),y: outputTensor.flatten().arraySync() },
        heatMapTF : {x: inpMeshGridTensor,y: outputTensor },

        hypNormal: [
            {x: 0, y: 0},
            {
                x: normalizedWeights[1]*( hyperplaneFac  ),
                y: normalizedWeights[0]*( hyperplaneFac  )
            }
        ], 

        hyperplane: [
            {
                x: p0[0],
                y: p0[1]
            },
            {
                x: p1[0],
                y: p1[1]
            }
        ],

        rndPt: [
            {
                x: rndPt[0],
                y: rndPt[1],
            }
        ],

        projRndPt: [
            {
                x: projRndPt[0],
                y: projRndPt[1],
            }
        ],

        marginRight: [
            {
                x: db_right[0].x,
                y: db_right[0].y
            },
            {
                x: db_right[1].x,
                y: db_right[1].y
            }
        ],
        marginLeft: [
            {
                x: db_left[0].x,
                y: db_left[0].y
            },
            {
                x: db_left[1].x,
                y: db_left[1].y
            }
        ],

        contourPlot: [
            {
                x: p0[0],
                y: p0[1]
            },
            {
                x: p1[0],
                y: p1[1]
            },
            {

                x:extDB[0].x,
                y:extDB[0].y
            },

            {

                x:extDB[1].x,
                y:extDB[1].y
            }

        ],

        hyerplaneParams : {
            weights: weights,
            bias: bias
        }

    
    }

}