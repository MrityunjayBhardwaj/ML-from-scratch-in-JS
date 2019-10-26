let mIrisX = tf.tensor(iris).slice([0,1],[100,2]);
// one hot encoded
// let mIrisY = tf.tensor( Array(100).fill([1,0],0,50).fill([0,1],50) );

let mIrisY = tf.tensor( Array(100).fill([+1], 0, 50).fill([-1], 50, 100));

const standardDataX = normalizeData(mIrisX, 1)

const {0: trainData,1: testData} = trainTestSplit(standardDataX,mIrisY,1/3);

// const classwiseDataSplit

const model = new svm(); 
model.fit(trainData, params = {threshold: .01, tollerance: 1, epoch: 10, });


// calculated weights and bias
// const calcWeights = tf.tensor([0.2688176, -2.5751076]).expandDims(1).transpose();// 0.2688176, -2.5751076 and bias: -.17151115159749847
// const calcBias = tf.tensor([-.17151115159749847]).expandDims(1);

const calcWeights = model.getParams().weights.reverse();
const calcBias    = model.getParams().bias;
const calcLegrangeMultipliers = model.getLegrangeMultipliers().flatten().arraySync();

function calcY(x, weights= [10, -2], bias = 1){
    const w = weights;
    const b = bias;

    // doing matMul with the given weights
    return tf.clipByValue( tf.round(tf.matMul(w , x, 0, 1).add(b) ) , -1, 1 );
}
// VISUALIZATION :-

// Helper Function:-

// NOTE: ONLY FOR 2D
// NOTE: input weights must either be tf.tensor obj or Array ( which gets automaticall converted by our funciton)
// whereas bias can be either a number/Array or a tf.tensor obj
// you can also give the angle and it will construct a 2d weight tf.tensor object form it
function calcHyperplane(usrParams = {weights: tf.tensor([l]),bias: tf.tensor([0]), range: [-5, 5]} ){

    let { 
      angle = 45,
      weights=tf.tensor([1]),
      bias=tf.tensor([0]),
      range=[-5, 5],
      rndPt = [2, 1],
    } = usrParams;


    // if angle is give then calculate the weights accordingly
    if (!usrParams.weights && usrParams.angle != undefined){
      
      angle = (Math.PI/180)*usrParams.angle;
      weights = [ Math.cos(angle)*1, Math.sin(angle)*1];
    }

    // converting weights and biases as tf.tensor if it wasn't already
    weights = (Array.isArray(weights))? tf.tensor(weights) : weights;
    bias = (Array.isArray(bias) || typeof bias === 'number')? tf.tensor(bias) : bias;

    // TODO: make sure only tf.tensor will come through the code below is just a bad hack
    if (!(weights.shape && bias.shape) )throw new Error('invalid input type, weights and biases must be either Array or tf.tensor object!');
  
    
    console.log('usrParams: ', usrParams);


  // calcuating the hyperplane:-

    const division = 50;
    const inp = tf.linspace(range[0], range[1], division).flatten().arraySync();

    // const boundry = inp.map((x) => fn(x, 1));

    // TODO: implement this technique to all the algos in order to get the performance boost for free
    const inpMeshGridTensor = tf.tensor(meshGrid(inp, inp)).reshape([inp.length**2, 2]);

    // for grid plot
    // calculate fn on the reshaped tensor (for faster calculation ) and then reshape it back
    const outputTensor = calcY(inpMeshGridTensor, weights, bias); // const output = outputTensor.arraySync(); 

    const normalizedWeights = ((weights).div(tf.norm(weights)).transpose()).arraySync();

    // point that is on the hyperplane
    // const x0 = [usrParams.rndPtX, usrParams.rndptY];
    const x0 = range; 
    const x1 = [
      -(weights.flatten().arraySync()[0]*x0[0] + bias.flatten().arraySync()[0])/weights.flatten().arraySync()[1],
      -(weights.flatten().arraySync()[0]*x0[1] + bias.flatten().arraySync()[0])/weights.flatten().arraySync()[1]
    ];

    // TODO: make it tensor:-
    // const x0t = tf.tensor(range);
    // const x2 = weights.mul(x0t.transpose()).add(bias).div(weights).neg();
    // x2.print();


    const p0 = [x0[0], x1[0]];
    const p1 = [x0[1], x1[1]];


    // point that are away from hyperplane :-
    function proj(x, hyperplane){

        const u = tf.tensor(x).expandDims(1);
        hyperplane= tf.tensor(hyperplane);

        return u.sub( u.transpose().matMul(hyperplane).div(tf.norm(hyperplane).pow(2)).mul(hyperplane)).flatten().arraySync()
    }

    // ( -( bias/( (tf.norm(tf.tensor(weights))).flatten().arraySync()[0] ) ) )

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

        hypNormal: [
            {x: 0, y: 0},
            {
                x: normalizedWeights[1]*( hyperplaneFac  ),
                y: normalizedWeights[0]*( hyperplaneFac  )
            }
        ], 

        hyperplane: [
            {
                x: p0[1],
                y: p0[0]
            },
            {
                x: p1[1],
                y: p1[0]
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

        margin: [
            {
                x: projRndPt[0],
                y: projRndPt[1],
            },
            {
                x: rndPt[0],
                y: rndPt[1],
            }

        ],

        hyerplaneParams : {
            weights: weights,
            bias: bias
        }

    
    }

}


const dataX = trainData.x.transpose().arraySync();
const dataY = trainData.y.flatten().arraySync();

const xRange0 = {min: 0*d3.min(dataX[0]) + -5, max: 5+ 0*d3.max(dataX[0]) };
const xRange1 = {min: 0*d3.min(dataX[1]) + -5, max: 5+ 0*d3.max(dataX[1]) };

const hyperplaneViz = calcHyperplane({
  weights: calcWeights,
  bias: calcBias,
  range: [
      xRange0.min,
      xRange0.max
  ] /* TODO: Calc this! */
});

const svgElemSize = 450;

const allMargin = 20;
const margin = {top: 40 || allMargin , bottom : 0 || allMargin ,
left: 0 || allMargin, right: 0 || allMargin };

// creating chart position properties:-
const innerHeight = svgElemSize - ( margin.bottom + margin.top);
const innerWidth  = svgElemSize - ( margin.right  + margin.left);
const svg = d3.select('#svgContainer');
svg.attr('width', svgElemSize );
svg.attr('height', svgElemSize );

console.log(svg);

// TODO:
// creating chart and axis lines:-
// plotting the data points
// data point decoration
// plotting a hyperplane
// real-time manipulation...


// creating chart and axis lines:-

const chartDecorGroup = svg.append('g').attr('class', 'chartDecor');
const chartDataGroup  = svg.append('g').attr('class', 'chartData');

const axisGroup = chartDecorGroup.append('g').attr('class', 'axis');

const xScale = d3.scaleLinear()
.domain([xRange0.min, xRange0.max ])
.range([margin.left, innerWidth]);

const yScale = d3.scaleLinear()
.domain([xRange1.min, xRange1.max])
.range([innerHeight, margin.top]); // its in reverse order 

// creating x and y Axis path for our chart:
const xAxis = d3.axisBottom(xScale)
// .tickFormat(d3.format('.1s'));

const yAxis = d3.axisLeft(yScale)
// .tickFormat(d3.format('.1s'));

// appending axis to our svg elems
axisGroup.append('g').call(xAxis)
.attr('transform', `translate( ${  0 }, ${( innerHeight/2 + margin.top/2)} )` );

axisGroup.append('g').call(yAxis)
.attr('transform', `translate( ${  (margin.left/2 + innerWidth/2) }, ${0})` );

// title components:-
const titlePos = {x: svgElemSize *.24, y: margin.top/2 };
const titleText = 'Support Vector Machine'

// inserting title :-
chartDecorGroup.append('g')
.attr('class', 'title')
.append('text')
.attr('transform', `translate(${titlePos.x}, ${titlePos.y})`)
.text(titleText)

/* title Style */
.attr('fill', 'white')
.attr('font-family', 'Helvetica')
.attr('font-weight', 600)
.attr('font-size', 20);

// heatmap
chartDataGroup.append('g').attr('class', 'heatMap')
.selectAll('circle').data(hyperplaneViz.heatMap.x).enter()
.append('circle')
.attr('transform', (d) => `translate(${xScale(d[0])},${yScale(d[1])}) `)
.attr('fill', (d,i) => {
    if (hyperplaneViz.heatMap.y[i] > 0)return 'rgba(0,0,255,0.4)';
    if (hyperplaneViz.heatMap.y[i] < 0)return 'rgba(255,0,0,0.4)';
    return 'white';
})
.attr('stroke', (d,i) => (hyperplaneViz.heatMap[i] === 0)? 'gray' : 'none')
.attr('r', 3)

const inputData = [];
const supportVectors = [];
for(let i=0;i< dataX[0].length;i++){
    const cPt = {x0: dataX[0][i], x1: dataX[1][i], class: 0, isSupportVector: 0};

    if (dataY[i] > 0)cPt.class = 1;
    if (calcLegrangeMultipliers[i] > 0){
        cPt.isSupportVector = 1
        supportVectors.push(cPt);
    }
    inputData.push(cPt)
}

console.log(inputData);
// plotting the data points
chartDataGroup.append('g').attr('class', 'dataPoints')
.selectAll('circle').data(inputData).enter()

.append('circle')
.attr('transform', (d) => {
return  `translate(${xScale(d.x0)},${yScale(d.x1)}) `

}
)
.attr('fill', (d) => {
    if (d.class){
        return `blue`
    }
        return `red`
    
})

.attr('r', 3)

// highlighting support Vectors:-
chartDataGroup.append('g').attr('class', 'supportVectors')
.selectAll('circle').data(supportVectors).enter()
.append('circle')
.attr('transform', (d) => `translate(${xScale(d.x0)},${yScale(d.x1)}) `)
.attr('stroke', 'black')
.attr('fill', 'none')
.attr('r', 5)

// data point decoration


// plotting a hyperplane
const lineGen = d3.line().curve(d3.curveCatmullRom);
const hypNormalLinePath = lineGen(hyperplaneViz.hypNormal.map(d =>  [xScale(d.x), yScale(d.y)]) ); 
const hyperplaneLinePath = lineGen(hyperplaneViz.hyperplane.map(d =>  [xScale(d.x), yScale(d.y)]) );
const marginLinePath = lineGen(hyperplaneViz.margin.map(d => [xScale(d.x), yScale(d.y)] ));

// adding hyperplane Normal path
const hypNormalElem = chartDataGroup.append('g').attr('id', 'hypNormal')

hypNormalElem.append('path')
.attr('d', hypNormalLinePath)
.attr('stroke','blue')
.attr('fill', 'none')
.attr('stroke-width', 3 );

const arrowGenerator = d3.symbol().type(d3.symbolTriangle).size(80);

hypNormalElem.append('g').attr('id', 'hypNormalArrow')
.append('path')
.attr('d', arrowGenerator())
.attr('transform', `translate(${ xScale(hyperplaneViz.hypNormal[1].x)}, ${yScale(hyperplaneViz.hypNormal[1].y)}) 
rotate(${0})`)
.attr('fill', 'blue');

//adding hyperplane path
hypNormalElem.append('g').attr('id', 'hyperplane')
.append('path')
.attr('d', hyperplaneLinePath)
.attr('stroke', 'red')
.attr('stroke-width', 2)
.attr('fill', 'none');


// real-time manipulation...


// inserting new points


