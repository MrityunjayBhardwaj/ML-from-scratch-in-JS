// Creating Visualization 1

const contourSliderElement1 = document.getElementById("contourSlider1");

// specifying our objective function
function function1D(x){
    return x
}

function constraint1D(x){
    return x.lessEqual(5)
}

// creating values for our plot
let x_1 = tf.linspace(0, 10,100);
let y_1 = function1D(x_1); 

// feasible points


// converting all the data to array
let x_1Array = x_1.flatten().arraySync();
let y_1Array = y_1.flatten().arraySync();

let data = [
    {
        x : x_1Array,
        y : y_1Array,
        type: 'scatter',

    },
]
let layout ={width:400, 
               height: 400} 
Plotly.newPlot('viz1Sketch',data,layout)

// updating the sketch everytime our contour line changes
function updateSketch(sliderValue){

    const contour =   tf.min(y_1).mul(1-(sliderValue/100))
                      .add(tf.max(y_1).mul(sliderValue/100));

    const contourLine = tf.ones(x_1.shape).mul(contour);

    // converting all the data to array
    const contourLineArray = contourLine.flatten().arraySync();

    // creating updated data array
    const data = [
        {
            x : x_1Array,
            y : y_1Array,
            type: 'scatter',

        },
        {
            x: x_1Array,
            y: contourLineArray,
            type: 'scatter',
            mode: 'line'
        }
    ]

    Plotly.react('viz1Sketch',data,layout)


}

// putting event listener for our contour line slider
contourSliderElement1.oninput = (a) =>{
    const sliderValue = contourSliderElement1.value;
    updateSketch(sliderValue);
}
