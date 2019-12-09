
const node_names = ['x1', 'x2', 'x3', 'x4'];
const dims1 = [2, 3, 2, 2];

// pad array just so that reference like x[1] later is easier to read
const x = (
    function(){

        const variableArray = [NaN];
        for(let i=0;i< node_names.length;i++){
            variableArray.push(new Variable(node_names[i], dims1[i]))
        }

        return variableArray;

    }()
);

const f3 = new Factor('f3', tf.tensor([0.2, 0.8]).expandDims(1));
const f4 = new Factor('f4', tf.tensor([0.5, 0.5]).expandDims(1));

// first index is x3, second index is x4, third index is x2
// looking at it like: arr[0][0][0]
const f234 = new Factor('f234', tf.tensor([
    [
        [0.3, 0.5, 0.2], [0.1, 0.1, 0.8]
    ], [
        [0.9, 0.05, 0.05], [0.2, 0.7, 0.1]
    ]
]));

// first index is x2
const f12 = new Factor('f12', tf.tensor([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]]));


const g = new FactorGraph(x[3], silent=false, debug=false);
g.append('x3', f234);
g.append('f234', x[4]);
g.append('f234', x[2]);
g.append('x2', f12);
g.append('f12', x[1]);
g.append('x3', f3);
g.append('x4', f4);

g.computeMarginals();


g.observe(['x2'], [2]).then(
    ()=>{

        console.log('after observe yes!')
        g.computeMarginals();
        const newMarginals = g.exportMarginals();

        newMarginals.x1.print();
        newMarginals.x2.print();
        newMarginals.x3.print();
        newMarginals.x4.print();
    }
)



// this.computeMarginals();

// this.exportMarginals();

// async function firstAsync2() {

//     // wait until the promise returns us a value
//     // let result = await promise; 
//     let result = await g.observe('x2', 2)

//         console.log('after observe yes!')
//     // g.computeMarginals();
//     // "Now it's done!"
// };firstAsync2();

// .then(
//     () =>{
//         console.log("observed", g.nodes['x1'].bfmarginal.print());
//     }
// )

// g.computeMarginals();

// g.nodes['x1'].marginal();
// g.observe('x3', )




// const nodeNames = ['covid', 'cough', 'headache', 'fatigue', 'fever', 'smoker', 'difficulty_breathing', 'sore_throat', 'asthema', 'sputum_production' ]
// const dims =      [2,         2,          2,         2,        2,       2,                2,                  2,         2,             2,];

// const nodes = (
//     function(){

//         const variableArray = [];
//         for(let i=0;i< nodeNames.length;i++){
//             variableArray.push(new Variable(nodeNames[i], dims[i]))
//         }

//         return variableArray;

//     }()
// );

// const cpdCovid       = new Factor(name='covid',    potentials=tf.tensor([[0.9],[0.1]]));

// const cpdCough       = new Factor(name='covid-cough',                potentials=tf.tensor([[0.50, 0.31  ], [0.50, 0.69  ]]));
// const cpdHeadache    = new Factor(name='covid-headache',             potentials=tf.tensor([[0.50, 0.13  ], [0.50, 0.87  ]]));
// const cpdFatigue     = new Factor(name='covid-fatigue',              potentials=tf.tensor([[0.50, 0.40  ], [0.50, 0.60  ]]));
// const cpdFever       = new Factor(name='covid-fever',                potentials=tf.tensor([[0.50, 0.11  ], [0.50, 0.89  ]]));
// const cpdSmoker      = new Factor(name='covid-smoker',               potentials=tf.tensor([[0.50, 0.78  ], [0.50, 0.22  ]]));
// const cpdBreathing   = new Factor(name='covid-difficulty_breathing', potentials=tf.tensor([[0.50, 0.813 ], [0.50, 0.187 ]]));
// const cpdSoreThroat  = new Factor(name='covid-sore_throat',          potentials=tf.tensor([[0.50, 0.86  ], [0.50, 0.14  ]]));
// const cpdAsthema     = new Factor(name='covid-asthema',              potentials=tf.tensor([[0.50, 0.60  ], [0.50, 0.40  ]]));
// const cpdSputum      = new Factor(name='covid-sputum_production',    potentials=tf.tensor([[0.50, 0.67  ], [0.50, 0.33  ]]));

// // console.log(nodes[0])
// const network = new FactorGraph(firstNode=nodes[0], silent=false,debug=true);

// network.append('covid', cpdCovid);

// network.append('covid', cpdCough);
// network.append('covid-cough', nodes[1]);

// network.append('covid', cpdHeadache);
// network.append('covid-headache', nodes[2]);

// network.append('covid', cpdFatigue);
// network.append('covid-fatigue', nodes[3]);

// network.append('covid', cpdFever);
// network.append('covid-fever', nodes[4]);

// network.append('covid', cpdSmoker);
// network.append('covid-smoker', nodes[5]);

// network.append('covid', cpdBreathing);
// network.append('covid-difficulty_breathing', nodes[6]);

// network.append('covid', cpdSoreThroat);
// network.append('covid-sore_throat', nodes[7]);

// network.append('covid', cpdAsthema);
// network.append('covid-asthema', nodes[8]);

// network.append('covid', cpdSputum);
// network.append('covid-sputum_production', nodes[9]);


// network.computeMarginals()



// network.observe(['covid'], [0]).then(
//     ()=>{

//         console.log('after observe yes!')
//         network.computeMarginals();
//         // const newMarginals = g.exportMarginals();

//         const marginals = network.exportMarginals();
//         for( node in marginals){
//             console.log('node: '+ node)
//             marginals[node].print()
//         }
//     }
// )


// const marginals = network.exportMarginals();
// for( node in marginals){
//     console.log('node: '+ node)
//     marginals[node].print()
// }