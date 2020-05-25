
// initialization
function genReward(actionIndex){

  let reward =  tf.randomNormal([1]).flatten().arraySync()[0]+allMeans[actionIndex];
  return reward;
}


const timeInterval = 10;

function onUpdateCallback(action, reward){

    // updating the reward array


    // update the slot machine name
    updateSlotMachine(action);

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

const eGreedyModel = new eGreedy({epsilon: 0.6, getReward: genReward, nActions: 10, updateCallback: onUpdateCallback});


let steps = 0;
const maxSteps = 150;

/* our main loop */
let intervelPromise = setInterval(

    // callback function
    () =>{
        
        eGreedyModel.update();
  
        // console.log('current Expected Action Value: ', eGreedyModel.getEstActionValue());
        steps++;

        // stopping criterion
        if (steps >= maxSteps){

            // shuffle(t=100);
            clearInterval(intervelPromise)
        }
    
    }

, timeInterval)