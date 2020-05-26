
// initialization
function genReward(actionIndex){

  let reward =  tf.randomNormal([1]).flatten().arraySync()[0]+allMeans[actionIndex];
  return reward;
}


const timeInterval = 10;

const baseIntervalTime = 1000;

function onUpdateCallback(action, reward){

    // updating the reward array


    // update the slot machine name
    updateSlotMachine(action,reward);

    // console.log(action, reward)
    rewardArray.push(reward);

    updateTimeline(rewardArray);

    rewardRFArray[action]++;
    updateRewardRF(action, rewardRFArray);

    updateRewardRFPie(action, rewardRFArray);
    // highlighting the selected action
    updateRewardDistViz(action, eGreedyModel.getEstActionValue());

    if(steps === 0)
        shuffle(t=10);

    updateQVal(action, eGreedyModel.getEstActionValue());
    updateQValBar(action, eGreedyModel.getEstActionValue());

}

// const eGreedyModel = new eGreedy({epsilon: 0.6, getReward: genReward, nActions: 10, updateCallback: onUpdateCallback});
const eGreedyModel = new eGreedy({epsilon: 0.6, getReward: genReward, nActions: 10});

let steps = 0;

eGreedyModel.update();

// // /* our main loop */
// let intervelPromise = setInterval(

//     // callback function
//     () =>{
        
//         eGreedyModel.update();
  
//         // console.log('current Expected Action Value: ', eGreedyModel.getEstActionValue());
//         steps++;

//         // stopping criterion
//         if (steps >= maxSteps){

//             // shuffle(t=100);
//             clearInterval(intervelPromise)
//         }
    
//     }

// , timeInterval)

const vizStepFunction = [
    () =>{updateCodeViz(vizStep)},
    () =>{updateCodeViz(vizStep)},
    () =>{updateCodeViz(vizStep)},
    () =>{updateCodeViz(vizStep)},
    () =>{updateCodeViz(vizStep)},
    () =>{updateCodeViz(vizStep)},
    () =>{updateCodeViz(vizStep)},
    () =>{updateCodeViz(vizStep)},
    () =>{

        const action = eGreedyModel.getUpdateAnalysis().cAction;
        console.log("myV",vizStep);

        updateSlotMachine(action, eGreedyModel.getUpdateAnalysis().cReward, baseIntervalTime);

        // TODO: break it into 2 updates one for fetching random value and second when updating the estimated reward
        updateRewardDistViz(action, eGreedyModel.getEstActionValue(), baseIntervalTime); 
        updateCodeViz(vizStep)
    
        rewardArray.push(eGreedyModel.getUpdateAnalysis().cReward);

        updateTimeline(rewardArray, baseIntervalTime);
    },
    () =>{
        rewardRFArray[eGreedyModel.getUpdateAnalysis().cAction]++;

        updateRewardRFPie(eGreedyModel.getUpdateAnalysis().cAction, rewardRFArray, baseIntervalTime);
        updateCodeViz(vizStep, baseIntervalTime)
        
    },
    () =>{
        updateQValBar(eGreedyModel.getUpdateAnalysis().cAction, eGreedyModel.getEstActionValue(), baseIntervalTime);
        updateCodeViz(vizStep)
    },
    () =>{updateCodeViz(vizStep)},
    () =>{updateCodeViz(vizStep)},

];

let vizStep = 0;

let cCycle = 0;

let maxCycle = 100; // steps in an episode

let vizInterval = Array(10).fill(baseIntervalTime);

// vizInterval[8] = 5000;

// /* our main loop */
let intervelPromise = setInterval(

    // callback function
    () =>{
       
        console.log('vStep', vizStep)
        vizStepFunction[vizStep]();
    


        // logic for parsing through the code..
        if(vizStep === 4 ){

            if (eGreedyModel.getUpdateAnalysis().isGreedyStep){
                vizStep = 5;
            }else{
                vizStep = 7;
            }
        }else{

            if (vizStep === 5 || vizStep === 7 ){
                vizStep = 8;

                return null;
            }
            vizStep++;
        }


        if(vizStep > 11){

            console.log('completed loop')
            vizStep= 3;
            cCycle++;

            eGreedyModel.update();
            resetCodeViz();

            resetInterConnectionViz();

            resetSlotMachine();

            resetRewardDistViz();

            resetQValBarViz();

            resetRewardRFPieViz();

        }

        if (cCycle > maxCycle)
            clearInterval(intervelPromise)

    }

, vizInterval[vizStep])
