/* this is highly experimental and created with glue and duct tape */


var d3 = Plotly.d3;
var img_jpg = d3.select('#jpg-export');


// generate meshgrid
const grid = meshGridRange(range={x: {min: -1.1, max: +1.1},y: {min: -1.1, max: +1.1}},division=50);

// calculate pNorm for each value of meshgrid

const  g = pNorm(tf.tensor(tf.tensor(grid[0]).transpose().arraySync()[0]).expandDims(1) )

const maxP = 100;

let np =1;

function pGif(p){


    const pNormGrid = grid.map( (a) =>{
        const f = tf.tensor(a).transpose().arraySync();
        const w = f.map( (b) =>{ return 1*(pNorm(tf.tensor(b),p).flatten().arraySync()[0])});
        // console.log(w);
        return w;
    });
    console.log('grid ',grid);
    console.log('pNormGrid ',pNormGrid);

    // visualzing the pNorm Grid
    const pNormVizData = [{
        x : grid[0][0],
        y : grid[0][0],
        z : pNormGrid,
        type: 'heatmap'
    }];

    Plotly.newPlot('pNormViz',pNormVizData,{title: 'Norm-'+p})

    .then(
        function(gd)
        {
        Plotly.toImage(gd,{height:500,width:500})
            .then(
                function(url)
            {
                console.log(url);  
                img_jpg.attr("src", url);

                // saveAs(document.getElementById('jpg-export'), "pretty image.png");
                //  const canvas = document.getElementById('pNormViz');
                // canvas.toBlob(function(blob) {
                //     saveAs(blob, "pretty image.png");
                // });
            //   var context = canvas.getContext("2d");
            // no argument defaults to image/png; image/jpeg, etc also work on some
            // implementations -- image/png is the only one that must be supported per spec.
    var img = document.images[0];
        // atob to base64_decode the data-URI
        var image_data = atob(img.src.split(',')[1]);
        // Use typed arrays to convert the binary data to a Blob
        var arraybuffer = new ArrayBuffer(image_data.length);
        var view = new Uint8Array(arraybuffer);
        for (var i=0; i<image_data.length; i++) {
            view[i] = image_data.charCodeAt(i) & 0xff;
        }
        try {
            // This is the recommended method:
            var blob = new Blob([arraybuffer], {type: 'application/octet-stream'});
        } catch (e) {
            // The BlobBuilder API has been deprecated in favour of Blob, but older
            // browsers don't know about the Blob constructor
            // IE10 also supports BlobBuilder, but since the `Blob` constructor
            //  also works, there's no need to add `MSBlobBuilder`.
            var bb = new (window.WebKitBlobBuilder || window.MozBlobBuilder);
            bb.append(arraybuffer);
            var blob = bb.getBlob('application/octet-stream'); // <-- Here's the Blob
        }

        // Use the URL object to create a temporary URL
        var url = (window.webkitURL || window.URL).createObjectURL(blob);
        location.href = url; // <-- Download!

            np++;
            pGif(np);

            
            sleep(5000);

                return Plotly.toImage(gd,{format:'jpeg',height:500,width:500});
            }
            )
        });


}

pGif(np);