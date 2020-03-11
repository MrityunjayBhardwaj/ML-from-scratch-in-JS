// Chart Params:-

const svgElemSize = 400;

const allMargin = 25;
const margin = {top: 0 || allMargin , bottom : 0 || allMargin ,
left: 0 || allMargin, right: 0 || allMargin };

// creating chart position properties:-
const innerHeight = svgElemSize - ( margin.bottom + margin.top);
const innerWidth  = svgElemSize - ( margin.right  + margin.left);



// creating a dummy data
const inputData = {x: [1,2,3,4],  y: [2,4,6,8]};

// making it into useable format
let data3_2 = [];


function quadraticRoots(a,b,c){
	return [(-b + Math.sqrt(-(b**2) - 4*a*c))/(2*a),
			(-b - Math.sqrt(-(b**2) - 4*a*c))/(2*a)] 
}

function getFeasiblePoints4z(z){
 return {
		x: quadraticRoots(-120,13,400-4*z),
		y: quadraticRoots(-80,13,400-9*z),
	}
}

function getDerivative(pt){
	const normalizingConstant1 = Math.sqrt((2*pt.x)**2 + (2*pt.y)**2);
	const normalizingConstant2 = Math.sqrt((3)**2 + (2)**2);
	return {nablaF: [2*pt.x/normalizingConstant1,2*pt.y/normalizingConstant1], nablaC: [3/normalizingConstant2,2/normalizingConstant2] };
}

function getIntersectionPoints(a,b,c,r){
	let intersectionPts = [];

	const x0 = -a*c/(a*a+b*b), y0 = -b*c/(a*a+b*b);

	if (c*c > r*r*(a*a+b*b)+Number.EPSILON){

		console.log("no Points");
		intersectionPts = [[null, null], [null, null]]
	}
	else if (Math.abs(c*c - r*r*(a*a+b*b)) < Number.EPSILON) {
		console.log("1 Point")
		intersectionPts= [[x0,y0], [null, null]];
	}
	else {
		const d = r*r - c*c/(a*a+b*b);
		const mult = Math.sqrt(d / (a*a+b*b));
		let ax, ay, bx, by;
		ax = x0 + b * mult;
		bx = x0 - b * mult;
		ay = y0 - a * mult;
		by = y0 + a * mult;
		console.log("2 points");
		intersectionPts.push([ax,ay]);
		intersectionPts.push([bx,by]);
	}

	return intersectionPts;
}
//0000000000000000000000000

// creating D3 Visualization:-
const lineGen = d3.line().curve(d3.curveCatmullRom);
const linePath = lineGen(data3_2.map(d =>  [d.x, d.y]) )

// setting up svg Element

const svg = d3.select('#viz3_2Sketch');
svg.attr('width', svgElemSize).attr('height',svgElemSize);


// const defs = svg.append("defs")

svg.append('defs').attr('class', 'myDefs').append("marker")
.attr('id', 'arrowNablaF')
.attr('viewBox', '0 -5 10 10')
.attr("refX",5)
.attr("refY",0)
.attr("markerWidth",4)
.attr("markerHeight",4)
.attr("orient","auto")
.append("path")
.attr("d", "M0,-5L10,0L0,5")
.attr('fill', 'purple')
.attr("class","arrowHead");

svg.append('defs').attr('class', 'myDefs').append("marker")
.attr('id', 'arrowNablaC')
.attr('viewBox', '0 -5 10 10')
.attr("refX",5)
.attr("refY",0)
.attr("markerWidth",4)
.attr("markerHeight",4)
.attr("orient","auto")
.append("path")
.attr("d", "M0,-5L10,0L0,5")
.attr('fill', 'magenta')
.attr("class","arrowHead");



const optViz = svg.append('g').attr('class', 'optimizationProblem')





// path for objective function
optViz.append('g').attr('class', 'linePlot')
.append('path')
.attr('d', linePath)
.attr('stroke','blue')
.attr('fill', 'none');

const circleGen = d3.symbol()
.type(d3.symbolCircle)
.size(60);

const circlePathData = circleGen();


svg.append('g').selectAll('circle')
.data(data3_2).enter()
.append('circle')
.attr('cx', d=> d.x)
.attr('cy', d=> d.y)
.attr('r', 4.5)
.attr('fill', 'none        y: innerHeight - 50 - Math.sin(x)*50*.4')
.attr('stroke', 'magenta')
.attr('stroke-width', 2);


/* creating bar-chart */
const chartGroup = svg.append('g');

const axisGroup = chartGroup.append('g');
axisGroup.attr('class', 'axis');

// const xScale = d3.scaleBand()
// .domain(data.map( (d,i) =>i))
// .range([margin.left, innerWidth ]);

const xScale = d3.scaleLinear()
.domain([d3.min(x2_3Array), d3.max(x1_3Array)])
.range([margin.left, innerWidth ]);

// console.log()
const yScale = d3.scaleLinear()
.domain([d3.min(x2_3Array), d3.max(x2_3Array)])
.range([margin.top, innerHeight ]);
// .range([innerHeight - (margin.top + margin.bottom), margin.top]); // note: here, we are reversing the order of our y-axis so that our axis gets displayed correctly!

// creating x and y Axis path for our Chart:-
const xAxis = d3.axisBottom(xScale)
.tickFormat(d3.format('.1s'));

const yAxis = d3.axisLeft(yScale)
.tickFormat(d3.format('.2s'));

// adding xAxis:-
axisGroup.append('g').call(xAxis)
.attr("transform", "translate(" + 0 + ", " + innerHeight  + ")");

// adding yAxis:-
axisGroup.append('g').call(yAxis)
.attr('transform', 'translate(' +  margin.left + ',' + margin.top*2 + ') scale(1, 1)' );

// adding title:-
chartGroup
.append('g')
.attr('class','title')
.append('text')
.attr("transform", "translate(" + svgElemSize*.34 + ", " + margin.top + ")")
.text('Line Plot')
.attr('fill', 'white')
.attr('font-family', 'Helvetica')
.attr('font-weight', 600)
.attr('font-size', 20);


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
// .on('mouseout'  , mouseout);

optViz.append('g').attr('class', 'contourCircle')
.append('circle')
.attr('cx', xScale(0))
.attr('cy', yScale(0))
.attr('r', 20)
.attr('stroke','blue')
.attr('fill', 'none');


// Initializing for visualizing derivative vectors
const symbolGenerator = d3.symbol()
  .type(d3.symbolCircle)
  .size(100);

const optimalPointPathData = symbolGenerator();

svg.append('g').attr('class', 'optimalPoint')
.append('path')
.attr('d', optimalPointPathData)
.attr("transform", "translate(" + xScale(0)+ ", " + yScale(0) + ")")
.attr('fill', 'orange')

const optVizNabla = optViz.append('g').attr('class', 'derivatives');
const defs = svg.append("defs");

const optVizNablaPt1 = optVizNabla.append('g').attr('class', 'pt1');

optVizNablaPt1.append('g').attr('class', 'nablaF')
.append('path')
.attr("marker-end", "url(#arrowNablaF)")
.attr('d', linePath)
.attr('stroke','purple')
.attr('fill', 'none');


optVizNablaPt1.append('g').attr('class', 'nablaC')
.append('path')
.attr("marker-end", "url(#arrowNablaC)")
.attr('d', linePath)
.attr('stroke','magenta')
.attr('fill', 'none');

const optVizNablaPt2 = optVizNabla.append('g').attr('class', 'pt2');

optVizNablaPt2.append('g').attr('class', 'nablaF')
.append('path')
.attr("marker-end", "url(#arrowNablaF)")
.attr('d', linePath)
.attr('stroke','purple')
.attr('fill', 'none');

optVizNablaPt2.append('g').attr('class', 'nablaC')
.append('path')
.attr("marker-end", "url(#arrowNablaC)")
.attr('d', linePath)
.attr('stroke','magenta')
.attr('fill', 'none');

// path for constriant function
optViz.append('g').attr('class', 'constraintPlot')
// .append('path')
// .attr('d', circlePathData)
// .attr('stroke','red')
// .attr('fill', 'none');
.selectAll('circle')
.data(data3_2).enter()
.append('circle')
.attr('cx', d=> d.x)
.attr('cy', d=> d.y)
.attr('r', 4.5)
.attr('fill', 'none')
.attr('stroke', 'magenta')
.attr('stroke-width', 2);

// function mouseover(){

//     hoverCircle.style('opacity', 1)
//     hoverText.style('opacity', 1)

// };

// function mousemove(){
//     // recover coordinate we need
//     const x0 = (d3.mouse(this)[0]);
//     const i  = bisect(data3_2, x0, 1);
//     const selectedData = data3_2[i];


//     // console.log(d3.mouse(this)[0],x0, i, selectedData, );

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

function updateLinePlot(newData, constraintPlotData){

    optViz.select('.linePlot').select('path','dot').transition().duration(500).attr('d', lineGen(newData.map(d => [d.x, d.y])));

    const svgDots = d3.selectAll('circle');

    // svgDots.data(newData).enter()
    // .append('circle')
    // .merge(svgDots)
    // .transition()
    // .duration(400)
    // .attr('cx', d=> d.x)
    // .attr('cy', d=> d.y)
    // .attr('r', 4.5)
    // .attr('fill', 'none')
    // .attr('stroke', 'magenta')
    // .attr('stroke-width', 2);

}





// adding interactivity:-
contourSliderElement3.addEventListener('change',() => {

    const sliderValue = contourSliderElement3.value;
	const currZValue = (sliderValue/100)*200;

	const q = [];

	// visualizing the slice of the objective function
	const slice = y_3.lessEqual(currZValue+1).mul(1).mul(y_3.greaterEqual(currZValue-1).mul(1));

	slice.dtype="bool";

	console.log('r', Math.sqrt(currZValue), xScale(Math.sqrt(currZValue)));
	optViz.select('.contourCircle').select('circle').transition().duration(500).attr('cx', xScale(0)).attr('cy', yScale(0)).attr('r',Math.abs(xScale(0) -xScale(Math.sqrt(currZValue))));

	function sgn(x){
		return (x < 0)? -1 : 1;
	}

	// finidng the intersection points

	let intersectionPts = [];
	const r = Math.sqrt(currZValue) ;
	const a = 3;
	const b = 2; 
	const c = -20; // given as input
	const x0 = -a*c/(a*a+b*b), y0 = -b*c/(a*a+b*b);

	console.log("askldjf")
	if (c*c > r*r*(a*a+b*b)+Number.EPSILON){

		console.log("no Points");

		// optViz.select('.constraintPlot').select('path','dot').transition().duration(500).attr('d', lineGen([{x:null, y: null}].map(d => [d.x, d.y])));


		// hide all the feasible point viz
		optViz.select('.constraintPlot').attr('opacity', '0.0')

		optVizNabla.select('.pt1').select('.nablaF').attr('opacity', '0.0')
		optVizNabla.select('.pt1').select('.nablaC').attr('opacity', '0.0')

		optVizNabla.select('.pt2').select('.nablaF').attr('opacity', '0.0')
		optVizNabla.select('.pt2').select('.nablaC').attr('opacity', '0.0')

		intersectionPts = [[null,null],[null,null] ]
	}
	else if (Math.abs(c*c - r*r*(a*a+b*b)) < Number.EPSILON) {
		console.log("1 Point")
		intersectionPts= [[x0,y0],[null, null]];
	}
	else {
		const d = r*r - c*c/(a*a+b*b);
		const mult = Math.sqrt(d / (a*a+b*b));
		let ax, ay, bx, by;
		ax = x0 + b * mult;
		bx = x0 - b * mult;
		ay = y0 - a * mult;
		by = y0 + a * mult;
		console.log("2 points");
		intersectionPts.push([ax,ay]);
		intersectionPts.push([bx,by]);


		// make the components visible again...
		optVizNabla.select('.pt1').select('.nablaF').attr('opacity', '1.0')
		optVizNabla.select('.pt1').select('.nablaC').attr('opacity', '1.0')

		optVizNabla.select('.pt2').select('.nablaF').attr('opacity', '1.0')
		optVizNabla.select('.pt2').select('.nablaC').attr('opacity', '1.0')

		optViz.select('.constraintPlot').attr('opacity', '1.0')

	}

	let data4Plot = [];
	for(let i=0; i<intersectionPts.length; i++){

		const currPt = {
			x: xScale(intersectionPts[i][0]),
			y: yScale(intersectionPts[i][1]),
		};

		console.log(currPt)
		data4Plot.push(currPt);
	}

	const pt1Derivatives = getDerivative({x: intersectionPts[0][0], y:  intersectionPts[0][1]});
	const pt2Derivatives = getDerivative({x: intersectionPts[1][0], y:  intersectionPts[1][1]});

	const scalingFactor = -20;
	const nablaFPt1Data = [
						{x: (0), 
						 y: (0)},
						{x: (pt1Derivatives.nablaF[0]*scalingFactor),
						 y: (pt1Derivatives.nablaF[1]*scalingFactor) }
						];

	const nablaCPt1Data = [
						{x: (0), 
						 y: (0)},
						{x: (pt1Derivatives.nablaC[0]*scalingFactor),
						 y: (pt1Derivatives.nablaC[1]*scalingFactor) }
						];

	const nablaFPt2Data = [
						{x: (0), 
						 y: (0)},
						{x: (pt2Derivatives.nablaF[0]*scalingFactor),
						 y: (pt2Derivatives.nablaF[1]*scalingFactor) }
						];

	const nablaCPt2Data = [
						{x: (0), 
						 y: (0)},
						{x: (pt2Derivatives.nablaC[0]*scalingFactor),
						 y: (pt2Derivatives.nablaC[1]*scalingFactor) }
						];



	optVizNabla.select('.pt1').select('.nablaF').select('path','dot').transition().duration(500).attr('d', lineGen(nablaFPt1Data.map(d => [d.x, d.y]))).attr('transform','translate('+xScale(intersectionPts[0][0])+','+yScale(intersectionPts[0][1])+')');
	optVizNabla.select('.pt1').select('.nablaC').select('path','dot').transition().duration(500).attr('d', lineGen(nablaCPt1Data.map(d => [d.x, d.y]))).attr('transform','translate('+xScale(intersectionPts[0][0])+','+yScale(intersectionPts[0][1])+')');


	optVizNabla.select('.pt2').select('.nablaF').select('path','dot').transition().duration(500).attr('d', lineGen(nablaFPt2Data.map(d => [d.x, d.y]))).attr('transform','translate('+xScale(intersectionPts[1][0])+','+yScale(intersectionPts[1][1])+')');
	optVizNabla.select('.pt2').select('.nablaC').select('path','dot').transition().duration(500).attr('d', lineGen(nablaCPt2Data.map(d => [d.x, d.y]))).attr('transform','translate('+xScale(intersectionPts[1][0])+','+yScale(intersectionPts[1][1])+')');



	console.log(data4Plot)
	optViz.select('.constraintPlot')
    .selectAll('circle').data(data4Plot).enter().append('circle')
    .merge(optViz.select('.constraintPlot').selectAll('circle'))
    .transition()
    .duration(500)
    .attr('cx', d=> d.x)
    .attr('cy', d=> d.y)
    .attr('r', 4.5)
    .attr('fill', 'gray')
    .attr('stroke', 'gray')
	.attr('stroke-width', 2);
	

});