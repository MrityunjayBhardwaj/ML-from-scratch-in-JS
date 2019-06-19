let matrix = nd.array([[1,2],[1,2]]);

let layout = {
	margin: {
	t: 0}  ,
	uirevision:'true',
	xaxis: {autorange: true},
	yaxis: {autorange: true}
}

// <---->

const TESTER = document.getElementById("tester");

// Update the plot in each time-intervel.
var cnt = 0;
var interval = setInterval(function() {


// div element for writing some informations.
const cns = document.getElementById('console');


/* Rotate the basis of column space using slider */

// get the angle from the sliders
let angle0 = -Math.PI +  (document.getElementById('mtx0').value/100 )*2*Math.PI;
let angle1 = -Math.PI +  (document.getElementById('mtx1').value/100 )*2*Math.PI;

// print angles to the HTML
document.getElementById('val0').innerHTML = angle0;
document.getElementById('val1').innerHTML = angle1;

// construct rotation matrix for both the basis
let rotMtx0 = [ [Math.cos(angle0)*1   , - Math.sin(angle0)],[Math.sin(angle0) , Math.cos(angle0)]];
let rotMtx1 = [ [Math.cos(angle1)*1   , - Math.sin(angle1)],[Math.sin(angle1) , Math.cos(angle1)]];

// calculate the rotation vector from rotation Matrix
let rotVec0 = JSON.parse( nd.la.matmul( nd.array(rotMtx0) , nd.array([1,0]).T ).toString() );
let rotVec1 = JSON.parse( nd.la.matmul( nd.array(rotMtx1) , nd.array([0,1]).T ).toString() );

// combine the rotated vectors to form our final matrix
matrix = nd.array( [ [ rotVec0[0][0],rotVec1[0][0] ],[rotVec0[1][0],rotVec1[1][0]] ] );
// matrix = nd.array([[1,2],[3,1]]);

// print the eigen vector in the HTML:
cns.innerHTML = nd.la.eigen(matrix)[1];

// calcuate the eigen Vectors and eigen Values
const eigVecs = JSON.parse( nd.la.eigen(matrix)[1].toString() );
const eigVals = JSON.parse( nd.la.eigen(matrix)[0].toString() );

// converting the matrix to javascript multi-dimensional arrays
matrix = nd.array( [ [ rotVec0[0][0],rotVec1[0][0] ],[rotVec0[1][0],rotVec1[1][0]] ] );
matrix = JSON.parse(matrix.toString());

// normalizing the matrix
let norm0 = Math.sqrt( matrix[0][0]**2 + matrix[1][0]**2 );
let norm1 = Math.sqrt( matrix[0][1]**2 + matrix[1][1]**2 );

matrix[0][0] /= norm0;
matrix[1][0] /= norm0;

matrix[0][1] /= norm1;
matrix[1][1] /= norm1;

// enabling the plotly to persist the user-interaction changes like zoom,pan etc.
layout.xaxis.autorange = true;
layout.yaxis.autorange = true;

// constructing the plotly Line plot.
Plotly.react( TESTER, [ {
	x: [0 , eigVecs[0][0]],
	y: [0 , eigVecs[1][0]],
	mode:'lines+marker',
	type: 'Lines',
	line: {color: "red"}
}
, {
	x: [0 , eigVecs[0][1]],
	y: [0 , eigVecs[1][1]],
	mode: 'lines',
	type: 'Lines',
	line: {color: "violet"}
},
// plotting the column space
{
	x : [0 , matrix[0][0]],
	y : [0 , matrix[1][0]],
	name: "Col[0]",
	mode:'lines+marker',
	type: 'lines',
	line: {color: "green",dash:"dot",width: 2}
},{
	x : [0 , matrix[0][1]],
	y : [0 , matrix[1][1]],
	name: "Col[1]",
	mode:'lines+marker',
	type: 'lines',
	line: {color: "green", width:2,dash:"dot"}
},


],layout);
  if(cnt === 100) clearInterval(interval);
}, 100);

