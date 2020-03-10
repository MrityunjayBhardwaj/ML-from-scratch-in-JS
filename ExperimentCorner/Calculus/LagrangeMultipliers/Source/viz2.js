// Creating Visualization 1

const contourSliderElement2 = document.getElementById("contourSlider2");

// applying objective function for our plot
let x_2 = tf.linspace(0, 10,100);
let y_2 = function1D(x_2); 

// applying constraints
const constraintMask1D = constraint1D(x_2);



let x_2Feasible = x_2;
let y_2Feasible = function1D(x_2Feasible);


// converting all the data to array
let x_2Array = x_2.flatten().arraySync();
let y_2Array = y_2.flatten().arraySync();

const data_2 = [
    {
        x : x_2Array,
        y : y_2Array,
        type: 'scatter',

    },

];
layout ={width:400, 
               height: 400};

let x_2FeasibleArray = 0;
let y_2FeasibleArray = 0;

tf.booleanMaskAsync(x_2, constraintMask1D).then(
    function(v){
        const x_2Feasible = v

    
        x_2FeasibleArray = x_2Feasible.flatten().arraySync();
        y_2FeasibleArray = y_2Feasible.flatten().arraySync();

        data_2.push(
            {
                x : x_2FeasibleArray,
                y : y_2FeasibleArray,
                type: 'scatter',
            }
        )

        data_2.push(
            {
                x : x_2FeasibleArray,
                y : tf.zeros(x_2Feasible.shape).flatten().arraySync(),
                type: 'scatter',
            }
        )

    
        Plotly.newPlot('viz2Sketch',data_2,layout)
    }
)






// updating the sketch everytime our contour line changes
function updateSketch2(sliderValue){



    // here, we want our contour line to obey our constraints as well
    let contour =   tf.min(y_2).mul(1-(sliderValue/100))
                      .add(tf.max(y_2).mul(sliderValue/100));

    contour = contour.clipByValue(0,5); // its not a generalized solution but its enough for what we want to convay through this visualization


    const contourLine = tf.ones(x_2.shape).mul(contour);

    // converting all the data to array
    const contourLineArray = contourLine.flatten().arraySync();

    // creating updated data array
    const data = [
        {
            x : x_2Array,
            y : y_2Array,
            type: 'scatter',

        },
        {
            x: x_2Array,
            y: contourLineArray,
            type: 'scatter',
            mode: 'line'
        },
        {
            x : x_2FeasibleArray,
            y : y_2FeasibleArray,
            type: 'scatter',
        },
        {
            x : x_2FeasibleArray,
            y : tf.zeros(x_2Feasible.shape).flatten().arraySync(),
            type: 'scatter',
        }

    ]

    Plotly.react('viz2Sketch',data, layout)


}





// putting event listener for our contour line slider
contourSliderElement2.oninput = (a) =>{
    const sliderValue = contourSliderElement2.value;
    updateSketch2(sliderValue);
}
