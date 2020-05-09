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


// const dataX = trainData.x.transpose().arraySync();
// const dataY = trainData.y.flatten().arraySync();

// const xRange0 = {min: 0*d3.min(dataX[0]) + -5, max: 5+ 0*d3.max(dataX[0]) };
// const xRange1 = {min: 0*d3.min(dataX[1]) + -5, max: 5+ 0*d3.max(dataX[1]) };

// const hyperplaneViz = calcHyperplane({
//   weights: calcWeights,
//   bias: calcBias,
//   range: [
//       xRange0.min,
//       xRange0.max
//   ] /* TODO: Calc this! */
// });

// const svgElemSize = 450;

// const allMargin = 20;
// const margin = {top: 40 || allMargin , bottom : 0 || allMargin ,
// left: 0 || allMargin, right: 0 || allMargin };

// // creating chart position properties:-
// const innerHeight = svgElemSize - ( margin.bottom + margin.top);
// const innerWidth  = svgElemSize - ( margin.right  + margin.left);
// const svg = d3.select('#svgContainer');
// svg.attr('width', svgElemSize );
// svg.attr('height', svgElemSize );

// console.log(svg);

// // TODO:
// // creating chart and axis lines:-
// // plotting the data points
// // data point decoration
// // plotting a hyperplane
// // real-time manipulation...


// // creating chart and axis lines:-

// const chartDecorGroup = svg.append('g').attr('class', 'chartDecor');
// const chartDataGroup  = svg.append('g').attr('class', 'chartData');

// const axisGroup = chartDecorGroup.append('g').attr('class', 'axis');

// const xScale = d3.scaleLinear()
// .domain([xRange0.min, xRange0.max ])
// .range([margin.left, innerWidth]);

// const yScale = d3.scaleLinear()
// .domain([xRange1.min, xRange1.max])
// .range([innerHeight, margin.top]); // its in reverse order 

// // creating x and y Axis path for our chart:
// const xAxis = d3.axisBottom(xScale)
// // .tickFormat(d3.format('.1s'));

// const yAxis = d3.axisLeft(yScale)
// // .tickFormat(d3.format('.1s'));

// // appending axis to our svg elems
// axisGroup.append('g').call(xAxis)
// .attr('transform', `translate( ${  0 }, ${( innerHeight/2 + margin.top/2)} )` );

// axisGroup.append('g').call(yAxis)
// .attr('transform', `translate( ${  (margin.left/2 + innerWidth/2) }, ${0})` );

// // title components:-
// const titlePos = {x: svgElemSize *.24, y: margin.top/2 };
// const titleText = 'Support Vector Machine'

// // inserting title :-
// chartDecorGroup.append('g')
// .attr('class', 'title')
// .append('text')
// .attr('transform', `translate(${titlePos.x}, ${titlePos.y})`)
// .text(titleText)

// /* title Style */
// .attr('fill', 'white')
// .attr('font-family', 'Helvetica')
// .attr('font-weight', 600)
// .attr('font-size', 20);

// // heatmap
// chartDataGroup.append('g').attr('class', 'heatMap')
// .selectAll('rect').data(hyperplaneViz.heatMap.x).enter()
// .append('rect')
// .attr('transform', (d) => `translate(${xScale(d[0])},${yScale(d[1])}) `)
// .attr('fill', (d,i) => {
//     if (hyperplaneViz.heatMap.y[i] > 0)return 'rgba(0,0,255,0.4)';
//     if (hyperplaneViz.heatMap.y[i] < 0)return 'rgba(255,0,0,0.4)';
//     return 'white';
// })
// .attr('stroke', (d,i) => (hyperplaneViz.heatMap[i] === 0)? 'gray' : 'none')
// .attr('width', 8)
// .attr('height', 7.2)

// const inputData = [];
// const supportVectors = [];
// for(let i=0;i< dataX[0].length;i++){
//     const cPt = {x0: dataX[0][i], x1: dataX[1][i], class: 0, isSupportVector: 0};

//     if (dataY[i] > 0)cPt.class = 1;
//     if (calcLegrangeMultipliers[i] > 0){
//         cPt.isSupportVector = 1
//         supportVectors.push(cPt);
//     }
//     inputData.push(cPt)
// }

// console.log(inputData);
// // plotting the data points
// chartDataGroup.append('g').attr('class', 'dataPoints')
// .selectAll('circle').data(inputData).enter()

// .append('circle')
// .attr('transform', (d) => {
// return  `translate(${xScale(d.x0)},${yScale(d.x1)}) `

// }
// )
// .attr('fill', (d) => {
//     if (d.class){
//         return `blue`
//     }
//         return `red`
    
// })

// .attr('r', 3)

// // highlighting support Vectors:-
// chartDataGroup.append('g').attr('class', 'supportVectors')
// .selectAll('circle').data(supportVectors).enter()
// .append('circle')
// .attr('transform', (d) => `translate(${xScale(d.x0)},${yScale(d.x1)}) `)
// .attr('stroke', 'black')
// .attr('fill', 'none')
// .attr('r', 5)

// // data point decoration


// // plotting a hyperplane
// const lineGen = d3.line().curve(d3.curveCatmullRom);
// const hypNormalLinePath = lineGen(hyperplaneViz.hypNormal.map(d =>  [xScale(d.x), yScale(d.y)]) ); 
// const hyperplaneLinePath = lineGen(hyperplaneViz.hyperplane.map(d =>  [xScale(d.x), yScale(d.y)]) );
// const marginLinePath = lineGen(hyperplaneViz.margin.map(d => [xScale(d.x), yScale(d.y)] ));

// // adding hyperplane Normal path
// const hypNormalElem = chartDataGroup.append('g').attr('id', 'hypNormal')

// hypNormalElem.append('path')
// .attr('d', hypNormalLinePath)
// .attr('stroke','blue')
// .attr('fill', 'none')
// .attr('stroke-width', 3 );

// const arrowGenerator = d3.symbol().type(d3.symbolTriangle).size(80);

// hypNormalElem.append('g').attr('id', 'hypNormalArrow')
// .append('path')
// .attr('d', arrowGenerator())
// .attr('transform', `translate(${ xScale(hyperplaneViz.hypNormal[1].x)}, ${yScale(hyperplaneViz.hypNormal[1].y)}) 
// rotate(${0})`)
// .attr('fill', 'blue');

// //adding hyperplane path
// // hypNormalElem.append('g').attr('id', 'hyperplane')
// // .append('path')
// // .attr('d', hyperplaneLinePath)
// // .attr('stroke', 'red')
// // .attr('stroke-width', 2)
// // .attr('fill', 'none');


// // real-time manipulation...


// // inserting new points


