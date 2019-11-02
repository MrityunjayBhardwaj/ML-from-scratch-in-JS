/* this is highly experimental and created with glue and duct tape */

let d3 = Plotly.d3;
let img_jpg = d3.select('#jpg-export');


// generate meshgrid
const gridRange = {min: -1.1, max: +1.1};
const gridDivision = 50;

const grid = meshGridRange(range={x: gridRange,y: gridRange },division=gridDivision);
const axisVals = tf.linspace(gridRange.min, gridRange, gridDivision).flatten().arraySync()

// calculate pNorm for each value of meshgrid
const  g = pNorm(tf.tensor(tf.tensor(grid[0]).transpose().arraySync()[0]).expandDims(1) )
let np =0;
const normArray = [.0001,0.001,0.01,0.1,0.25,0.5,0.75, 1.25,1.5,1.75]


function pGif(p){

    const pNormGrid = pNorm(tf.tensor(grid).transpose(), p).arraySync();
    console.log('grid ',grid);
    console.log('pNormGrid ',pNormGrid);

    // visualzing the pNorm Grid
    const pNormVizData = [{
        x: axisVals,
        y: axisVals,
        z : pNormGrid,
        type: 'contour',

        colorscale : [[0, darkModeCols.blue()], [0.25, darkModeCols.purple()],[0.5, darkModeCols.magenta()], [.75, darkModeCols.yellow()], [1, darkModeCols.red()]],
    }];

    const layoutSetting = {
        title: 'Norm-'+p,
        font : {
            size : 15,
            color: 'white',
            family : 'Helvetica'
        },
        paper_bgcolor : '#222633',


    }

    Plotly.newPlot('pNormViz',pNormVizData,layoutSetting).then(
        function(gd)
        {
        Plotly.toImage(gd,{height:800,width:800})
            .then(
                function(url)
            {
                console.log(url);  
                img_jpg.attr("src", url);

            let img = document.images[0];
                // atob to base64_decode the data-URI
                let image_data = atob(img.src.split(',')[1]);
                // Use typed arrays to convert the binary data to a Blob
                let arraybuffer = new ArrayBuffer(image_data.length);
                let view = new Uint8Array(arraybuffer);
                for (let i=0; i<image_data.length; i++) {
                    view[i] = image_data.charCodeAt(i) & 0xff;
                }
                try {
                    // This is the recommended method:
                    let blob = new Blob([arraybuffer], {type: 'application/octet-stream'});
                } catch (e) {
                    // The BlobBuilder API has been deprecated in favour of Blob, but older
                    // browsers don't know about the Blob constructor
                    // IE10 also supports BlobBuilder, but since the `Blob` constructor
                    //  also works, there's no need to add `MSBlobBuilder`.
                    let bb = new (window.WebKitBlobBuilder || window.MozBlobBuilder);
                    bb.append(arraybuffer);
                    let blob = bb.getBlob('application/octet-stream'); // <-- Here's the Blob
                }

                // Use the URL object to create a temporary URL
                let url = (window.webkitURL || window.URL).createObjectURL(blob);
                location.href = url; // <-- Download!

                    if(np < normArray.length)
                        pGif(normArray[np]);
                        np++;
                        sleep(1000);

                        return Plotly.toImage(gd,{format:'jpeg',height:800,width:800});
                    }
                    )
                });


}

pGif(np);