
const codeLines = document.getElementById('codeViz').getElementsByTagName('div');

let copyedCodeLines = Object.assign({}, codeLines);

let subStep = 0;

let lineStep = 0;

let specialCodeUpdateFns = [
    /* sub-steps for line 5 */
    [

        
        (cElem)=>{
            console.log('inside 1')

            const subElem = cElem.getElementsByTagName('span');

            subElem[0].style.border = "3px solid red";
            subElem[0].innerHTML = eGreedyModel.getModel().epsilon; // TODO: replace '5' with epsilon value 
        },
        (cElem)=>{

            const subElem = cElem.getElementsByTagName('span');

            subElem[1].style.border = "3px solid red";
            subElem[1].innerHTML = eGreedyModel.getUpdateAnalysis().randomVal.toFixed();
        },
        (cElem)=>{

            const subElem = cElem.getElementsByTagName('span');

            subElem[0].style.border = "";
            subElem[1].style.border = "";
        },

        (cElem)=>{

            const subElem = cElem.getElementsByTagName('span');


            subElem[0].innerHTML = '\(\epsilon\)'; // TODO: render it back to using mathJax
            subElem[1].innerHTML = 'rand()';

            if (eGreedyModel.getUpdateAnalysis().isGreedyStep){
                subElem[2].innerHTML = "<- True";
            }
            else{

                subElem[2].innerHTML = "<- False";

            }

        },
    ],
    [

        /* these set of animantion applies for both the line 6 and 8 */

        (cElem) =>{

            console.log('inside 2')
            const subElem = cElem.getElementsByTagName('span');

            subElem[0].style.border = "3px solid red";

        },

        (cElem) =>{

            const subElem = cElem.getElementsByTagName('span');

            subElem[0].style.border = "";

            subElem[1].style.color = "red";
            subElem[1].innerHTML = "<- slot Machine #"+eGreedyModel.getUpdateAnalysis().cAction;

        },
        (cElem) =>{

            const subElem = cElem.getElementsByTagName('span');

            subElem[0].style.border = "";
            subElem[1].style.color = "";
            subElem[1].innerHTML = "<- slot Machine #"+eGreedyModel.getUpdateAnalysis().cAction;

        }
    ],
    [

        (cElem)=>{
            console.log('inside 3');

            const subElem = cElem.getElementsByTagName('span');

            subElem[0].style.border = "3px solid red";

        },
        (cElem)=>{
            const subElem = cElem.getElementsByTagName('span');

            subElem[1].style.color="red";
            subElem[1].innerHTML = eGreedyModel.getUpdateAnalysis().cReward.toFixed(2);

        },
        (cElem)=>{
            const subElem = cElem.getElementsByTagName('span');

            subElem[0].style.border = "";
            subElem[1].style.color="white";

        }
    ]


]

const codeLength = 11;
let isCustomFns = Array(codeLength).fill(0);

isCustomFns[4]= 1;
isCustomFns[5]= 1;
isCustomFns[7]= 1;
isCustomFns[8]= 1;



function updateCodeViz(cAction, cSubStep=-1){
    let cLineNum = 0;
    for( let cLineElem of codeLines ){

        cLineElem.style.border= '';

        if(cLineNum === cAction){

            // default behaviour
            cLineElem.style.border= '2px solid white'

            // custom behaviour
            if(isCustomFns[cAction]){

                let interval = setInterval(
                    ()=>{

                        specialCodeUpdateFns[lineStep][subStep](cLineElem);

                        subStep++; 

                        if(subStep >= specialCodeUpdateFns[lineStep].length){
                            lineStep++;
                            subStep = 0;
                            clearInterval(interval);
                        }

                        
                    }, 1*baseIntervalTime/6 
                )


            }


        }

        cLineNum++;
    }



}

function resetCodeViz(){
    lineStep =0;

    // remove the if condition result 'True' or 'False'
    codeLines[4].getElementsByTagName('span')[2].innerHTML="";
    codeLines[5].getElementsByTagName('span')[1].innerHTML="";
    codeLines[7].getElementsByTagName('span')[1].innerHTML="";
    codeLines[8].getElementsByTagName('span')[1].innerHTML="";
}