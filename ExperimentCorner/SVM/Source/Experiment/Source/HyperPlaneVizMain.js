// Chart Params:-



// TODO: ADD 

const svgElemSize = 450;

const allMargin = 20;
const margin = {top: 40 || allMargin , bottom : 0 || allMargin ,
left: 0 || allMargin, right: 0 || allMargin };

// creating chart position properties:-
const innerHeight = svgElemSize - ( margin.bottom + margin.top);
const innerWidth  = svgElemSize - ( margin.right  + margin.left);


// adding real data!

function fn(x, weights= [10, -2], bias = 1){
    const w = tf.tensor(weights).expandDims(1);
    const b = bias;

    // doing matMul with the given weights
    return tf.clipByValue( tf.round(tf.matMul(w , x, 1, 1).add(b) ) , -1, 1 );
}


function calcHyperplane(usrParams){

    const range = [-5, 5]; 
    const division = 200;
    const inp = tf.linspace(range[0], range[1], division).flatten().arraySync();

    // const boundry = inp.map((x) => fn(x, 1));

    // TODO: implement this technique to all the algos in order to get the performance boost for free
    const inpMeshGridTensor = tf.tensor(meshGrid(inp, inp));


    // params of hyperplane:-
    let angle = (Math.PI/180)*usrParams.angle;
    let weights = [ Math.cos(angle)*1, Math.sin(angle)*1];
    let bias  = usrParams.bias;

    // calculate fn on the reshaped tensor (for faster calculation ) and then reshape it back
    // const outputTensor = fn(inpMeshGridTensor.reshape([inp.length**2, 2]), weights, bias).reshape([inp.length, inp.length]);
    // const output = outputTensor.arraySync(); 

    const normalizedWeights = (tf.tensor(weights).div(tf.norm(weights)).transpose()).flatten().arraySync();

    normalizedWeights[1] = ((normalizedWeights[1]))?normalizedWeights[1]: 1e-11 ;

    // point that is on the hyperplane
    const x1 = [-5, 5]
    let x0 = [-(normalizedWeights[0]*x1[0] + bias)/normalizedWeights[1], -(normalizedWeights[0]*x1[1] + bias)/normalizedWeights[1]];

    // point that are away from hyperplane :-

    function proj(x, weights ){
         x = tf.tensor(x).expandDims(1);
        weights = tf.tensor(weights).expandDims(1);

        return x.sub( x.transpose().matMul(weights).div(tf.norm(weights).pow(2)).mul(weights)).flatten().arraySync()
    }

    const rndPt = [usrParams.rndPtX, usrParams.rndPtY];
    // ( -( bias/( (tf.norm(tf.tensor(weights))).flatten().arraySync()[0] ) ) )

    const projPlane = [ normalizedWeights[1] ,
                        normalizedWeights[0] 
    ];

    let projRndPt = proj(rndPt, projPlane);

    let rndPtPredY = normalizedWeights[0]*rndPt[0] + normalizedWeights[1]*rndPt[1] + bias;

    // let projRndPt = proj(rndPt, [normalizedWeights[1], normalizedWeights[0] ]  )
    // projRndPt = proj(rndPt, [normalizedWeights[1]* ( -( bias/( (tf.norm(tf.tensor(weights))).flatten().arraySync()[0] ) ) ), normalizedWeights[0]* ( -( bias/( (tf.norm(tf.tensor(weights))).flatten().arraySync()[0] ) ) ) ]  )

    projRndPt[0] = projRndPt[0] + normalizedWeights[1]*( -( bias/( (tf.norm(tf.tensor(weights))).flatten().arraySync()[0] ) ) );
    projRndPt[1] = projRndPt[1] + normalizedWeights[0]*( -( bias/( (tf.norm(tf.tensor(weights))).flatten().arraySync()[0] ) ) );

    const hyperplaneFac = ( -( bias/( (tf.norm(tf.tensor(weights))).flatten().arraySync()[0] ) ) );

    return {

        hypNormal: [
            {x: 0, y: 0},
            {
                x: normalizedWeights[1]*( hyperplaneFac  ),
                y: normalizedWeights[0]*( hyperplaneFac  )
            }
        ], 

        hyperplane: [
            {
                x: x0[1],
                y: x1[1]
            },
            {
                x: x0[0],
                y: x1[0]
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
        rndPtPredY: [
            rndPtPredY
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

        hyperplaneParams : {
            weights: weights,
            bias: bias
        }

    
    }

}

/* Creating Hyperplane:- */
const hyperplaneViz = calcHyperplane(
  {angle: 40, bias: 2, rndPtX: 2, rndPtY: 2}
);

data = hyperplaneViz.hypNormal;

console.log(data);

// setting up svg Element
const svg = d3.select('#svgContainer')
// .attr('viewBox', [0, 0, svgElemSize, svgElemSize]);
svg.attr('width', svgElemSize).attr('height',svgElemSize);
// .style('background-color','gray');

const svgContainerCursor = svg.append('g').attr('cursor', 'grab');


/* creating bar-chart */
const chartGroup = svg.append('g');

const axisGroup = chartGroup.append('g');
axisGroup.attr('class', 'axis');

const xScale = d3.scaleLinear()
.domain([-5, 5+0*d3.max(data, d => d.x)])
.range([margin.left, innerWidth ]);

const yScale = d3.scaleLinear()
.domain([-5, 5+0*d3.max(data, d => d.y)])
.range([innerHeight, margin.top]);

// creating x and y Axis path for our Chart:-
const xAxis = d3.axisBottom(xScale)
.tickFormat(d3.format('.1s'));

const yAxis = d3.axisLeft(yScale)
.tickFormat(d3.format('.1s'));

// adding xAxis:-
axisGroup.append('g').call(xAxis)
.attr("transform", "translate(" + 0 + ", " + ( innerHeight/2 + margin.top/2)  + ")");

// adding yAxis:-
axisGroup.append('g').call(yAxis)
.attr('transform', 'translate(' +  (margin.left/2 + innerWidth/2) + ',' + 0 + ') scale(1, 1)' );

// adding title:-
chartGroup
.append('g')
.attr('class','title')
.append('text')
.attr("transform", "translate(" + svgElemSize*.24 + ", " + margin.top/2 + ")")
.text('Seperating Hyperplane')
.attr('fill', 'white')
.attr('font-family', 'Helvetica')
.attr('font-weight', 600)
.attr('font-size', 20);

// adding mid line:-
svg.append('g').attr('class', 'axisLine')
.append('path')
.attr('d',  d3.line()([[xScale(-5), yScale(0)],[xScale(5), yScale(0)]])  )
.style('stroke', 'green');

svg.append('g').attr('class', 'axisLine')
.append('path')
.attr('d',  d3.line()([[xScale(0), yScale(-5)],[xScale(0), yScale(5)]])  )
.style('stroke', 'red');














/* CHART INTERACTIVITY */

// this gives a function which allows us to find the closest 
// x index of the mouse
const bisect = d3.bisector( (d) => d.x ).left;

const hoverGroup  = svg.append('g') .attr('class', 'hoverGroup');
const hoverCircle = hoverGroup.append('g').attr('class', 'hoverCircle').append('circle')
.attr('r' , 10)
.style('opacity', 0)
.style('fill', 'none')
.style('stroke', 'red');

const hoverText = hoverGroup.append('g').attr('class', 'hoverText').append('text')
.style('opacity', 0)
.attr('text-anchor', 'left')
.attr('alignment-baseline', 'middle');

// create a rect on top of the svg area: this rect `recovers`
// mouse position.
svg.append('rect')
.style('fill', 'none')
.style('pointer-events', 'all')
.attr('width' , svgElemSize)
.attr('height', svgElemSize)
// .on('mouseover' , mouseover)
// .on('mousemove' , mousemove)
// .on('mouseout'  , mouseout)
;

function mouseover(val){
    console.log('mouseover',d3.mouse(this));
}
function mousemove(val){
    console.log('mousemove',);
}
function mouseout(val){
    console.log('mouseout',val);
}













// creating D3 Visualization:-
const lineGen = d3.line().curve(d3.curveCatmullRom);
const hypNormalLinePath = lineGen(data.map(d =>  [xScale(d.x), yScale(d.y)]) ); 
const hyperplaneLinePath = lineGen(hyperplaneViz.hyperplane.map(d =>  [xScale(d.x), yScale(d.y)]) );
const marginLinePath = lineGen(hyperplaneViz.margin.map(d => [xScale(d.x), yScale(d.y)] ));

// adding hyperplane Normal path

const hypNormalElem = svg.append('g').attr('id', 'hypNormal')

hypNormalElem.append('path')
.attr('d', hypNormalLinePath)
.attr('stroke','red')
.attr('fill', 'none')
.attr('stroke-width', 3 );

const arrowGenerator = d3.symbol().type(d3.symbolTriangle).size(80);

hypNormalElem.append('g').attr('id', 'hypNormalArrow')
.append('path')
.attr('d', arrowGenerator())
.attr('transform', `translate(${ xScale(hyperplaneViz.hypNormal[1].x)}, ${yScale(hyperplaneViz.hypNormal[1].y)}   ) 
    rotate(${(hyperplaneViz.hyperplaneParams.bias > 0)? 0 : 180})`)
.attr('fill', 'red')


// adding hyperplane path
svg.append('g').attr('id', 'hyperplane')
.append('path')
.attr('d', hyperplaneLinePath)
.attr('stroke', 'blue')
.attr('stroke-width', 2)
.attr('fill', 'none');

// adding random Point
svg.append('g').attr('id', 'rndPt')
.append('circle')
.attr('cx', xScale(hyperplaneViz.rndPt[0].x))
.attr('cy', yScale(hyperplaneViz.rndPt[0].y))
.attr('r', 4.5)
.attr('fill', 'none        y: innerHeight - 50 - Math.sin(x)*50*.4')
.attr('stroke', 'magenta')
.attr('stroke-width', 2)
.call(d3.drag()
    .on('start', dragStart)
    .on('drag', dragging)
    .on('end', dragEnd)
);

function dragStart(){
    console.log('its inside');
    d3.select(this).raise();
    svg.attr('cursor', 'grabbing');

}

// gathering some data:-
const angleSlider  = document.getElementById('angleSlider');
const biasSlider   = document.getElementById('biasSlider');
const weightsOutput = {
    x: document.getElementById('weightXOut'),
    y: document.getElementById('weightYOut') 
};
const rndPtOutput = {
    x: document.getElementById('rndPtXOut'),
    y: document.getElementById('rndPtYOut')
};
const biasOutput = document.getElementById('biasOut');
const predOutput = document.getElementById('predOut');

function dragging(d){
    // console.log( d3.mouse(this));

    const rndPt = {x: d3.mouse(this)[0], y: d3.mouse(this)[1]};


    // rndPt box constraint:-
	if ( margin.left > rndPt.x  || rndPt.x > innerWidth || margin.top > rndPt.y || innerHeight < rndPt.y)return;
	
	const newHyperplane = calcHyperplane({
        angle: angleSlider.value*360, 
        bias: (biasSlider.value)*5,
        rndPtX: d3.scaleLinear().domain([margin.left, innerWidth]).range([-5, 5])( (rndPt.x )),
        rndPtY: d3.scaleLinear().domain([innerHeight, margin.top]).range([-5, 5])( (rndPt.y )),
	});

	// visualizing values of our random point 
	rndPtOutput.x.innerHTML = newHyperplane.rndPt[0].x.toFixed(1);
	rndPtOutput.y.innerHTML = newHyperplane.rndPt[0].y.toFixed(1);

    updatePlot(newHyperplane);
}

function dragEnd() {
    svg.attr('cursor', 'normal');
}

// adding projection of the random Point onto the hyperplane :-
svg.append('g').attr('id', 'projRndPt')
.append('circle')
.attr('cx',xScale(hyperplaneViz.projRndPt[0].x) )
.attr('cy',yScale(hyperplaneViz.projRndPt[0].y) )
.attr('r', 4.5)
.attr('fill', 'none        y: innerHeight - 50 - Math.sin(x)*50*.4')
.attr('stroke', 'magenta')
.attr('stroke-width', 2);

// adding margin:-
svg.append('g').attr('id', 'margin')
.append('path')
.attr('d', marginLinePath)
.attr('stroke', 'white')
.attr('stroke-width', 2)
.attr('stroke-dasharray', '4 4')


function updatePlot(hyperplaneViz){

    // console.log(hyperplaneViz);

	// visualizing the the values of weights and bias
	weightsOutput.x.innerHTML = hyperplaneViz.hyperplaneParams.weights[0].toFixed(1);
    weightsOutput.y.innerHTML = hyperplaneViz.hyperplaneParams.weights[1].toFixed(1);

    biasOutput.innerHTML = hyperplaneViz.hyperplaneParams.bias.toFixed(1);

    predOutput.innerHTML = hyperplaneViz.rndPtPredY[0].toFixed(1);


    const transitionDuration = 2;

    const hypNormalLinePath = lineGen(hyperplaneViz.hypNormal.map(d =>  [xScale(d.x), yScale(d.y)]) );
    const hyperplaneLinePath = lineGen(hyperplaneViz.hyperplane.map(d =>  [xScale(d.x), yScale(d.y)]) );
    const marginLinePath = lineGen(hyperplaneViz.margin.map(d => [xScale(d.x), yScale(d.y)] ));

    svg.select('#hypNormal path').transition().duration(transitionDuration).attr('d', hypNormalLinePath);
    svg.select('#hyperplane path').transition().duration(transitionDuration).attr('d', hyperplaneLinePath );
    svg.select("#margin path").transition().duration(transitionDuration).attr('d', marginLinePath);

    // updating random Point
    svg.select('#rndPt circle')
    .transition().duration(transitionDuration)
    .attr('cx', xScale(hyperplaneViz.rndPt[0].x))
    .attr('cy', yScale(hyperplaneViz.rndPt[0].y));

    // updating projection of the random Point onto the hyperplane :-
    svg.select('#projRndPt circle')
    .transition().duration(transitionDuration)
    .attr('cx',xScale(hyperplaneViz.projRndPt[0].x) )
    .attr('cy',yScale(hyperplaneViz.projRndPt[0].y) );


    
    svg.select('#hypNormalArrow path')
    .transition().duration(transitionDuration)
    .attr('transform', `translate(${ xScale(hyperplaneViz.hypNormal[1].x)}, ${yScale(hyperplaneViz.hypNormal[1].y)})
        rotate(${angleSlider.value*360})
        scale(1,${( hyperplaneViz.hyperplaneParams.bias > 0)? -1 : 1})
        translate(0,6.7)
	`)
	


}

// INTERACTIVITY


function getUsrParams () {

    return{
        angle:angleSlider.value*360 , 
        bias: (biasSlider.value)*5 ,
        rndPtX: d3.scaleLinear().domain([margin.left, innerWidth]).range([-5, 5])( +(d3.select('#rndPt circle')).attr('cx')),
        rndPtY: d3.scaleLinear().domain([innerHeight, margin.top]).range([-5, 5])( +(d3.select('#rndPt circle')).attr('cy')),
    } 
};

angleSlider.addEventListener('input', () => updatePlot(calcHyperplane(getUsrParams())));
biasSlider.addEventListener('input', () => updatePlot(calcHyperplane(getUsrParams())));

// NOTE: its a dummy code it only server the purpose of automatically test 
//some functionality of our updatePlot function:D

let count = 0;

function updateChartLoop() {

    setTimeout(() => {

        console.log("updating Chart Loop");
        updatePlot(calcHyperplane(angle=Math.random()*360)); 
        if(count < 0){
            updateChartLoop();
            count++;
        }
    } , 1000);

}

// TODO: Create points for classification 

// updateChartLoop();


// function mouseover(){

//     hoverCircle.style('opacity', 1)
//     hoverText.style('opacity', 1)

// };

// function mousemove(){
//     // recover coordinate we need
//     const x0 = (d3.mouse(this)[0]);
//     const i  = bisect(data, x0, 1);
//     const selectedData = data[i];


//     console.log(d3.mouse(this)[0],x0, i, selectedData, );

//     hoverCircle
//     .attr('cx', (selectedData.x))
//     .attr('cy', (selectedData.y))

//     const textOffset = 15;
//     hoverText
//     .html(`x: ${selectedData.x.toFixed(1) }, y: ${(selectedData.y).toFixed(1)}`)
//     .attr('x', (selectedData.x + textOffset))
//     .attr('y', selectedData.y )
//     .style('opacity', 1)

// }

// function mouseout(){
//     console.log('mouseOut');
//     hoverCircle.style('opacity', 0);
//     hoverText.style('opacity', 0);
// }

// function updateLinePlot(newData){

//     d3.select('path','dot').transition().duration(500).attr('d', lineGen(newData.map(d => [d.x, d.y])));

//     const svgDots = d3.selectAll('circle');


//     svgDots.data(newData).enter()
//     .append('circle')
//     .merge(svgDots)
//     .transition()
//     .duration(400)
//     .attr('cx', d=> d.x)
//     .attr('cy', d=> d.y)
//     .attr('r', 4.5)
//     .attr('fill', 'none')
//     .attr('stroke', 'magenta')
//     .attr('stroke-width', 2);

// // svg.append('g').selectAll('dot')
// // .data(data).enter()
// // .append('circle')
// // .attr('cx', d=> d.x)
// // .attr('cy', d=> d.y)
// // .attr('r', 4.5)
// // .attr('fill', 'none')
// // .attr('stroke', 'magenta')
// // .attr('stroke-width', 2)

// }





// // adding interactivity:-
// const weightslider = document.getElementById('weightSlider');
// const weightValue  = document.getElementById('weightValue');

// weightslider.addEventListener('input',() => {

//     weightValue.innerHTML = weightSlider.value


//     // creating new Dummy Data

//     const newDummyData = [];
//     for(let i=0;i<nDummyData;i++){

//         const x = margin.left + i*30
//         const currPt = {
//             // x: margin.left + Math.floor(innerWidth*Math.random()), 
//             x: x,
//             // y: margin.top + Math.floor((innerHeight - margin.bottom)*Math.random())

//             y: innerHeight - 100 - Math.sin(x)*100*Math.random()
//         };

//         newDummyData.push(currPt);
//     }

//     data = newDummyData;
//     updateLinePlot(newDummyData)


// });


/**
 *  how i want it to be used:-
 *
 * const  
 * data = [];
 * 
 // currently only works on svg paths
 *const viz = simplyViz( #selector, data, params={width=400, height=400} );
 viz.svgContainer({width: 1, height: 2})
 viz.margin({left: 1, right: 10, top: 10, bottom: 10})
 viz.title('my Title')

 * .update('insert newData or fx')
 * 
 * onChange, () => viz.update('data');
 * 
 * 
 * 
 */


 /**navigator.
  * TODO: 
  * - Read the seperating hyperplane seciton again try to simplify the ideas through viz
  * - add weights visualizer next to the svg
  * - 
  */
