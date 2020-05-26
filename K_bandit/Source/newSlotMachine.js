
const slotMachineElem = document.getElementById('slotMachine');
let banditsListElem = document.getElementById('banditsList');
const casinoElem = document.getElementById('casino');
const slotMachineTitleElem = document.getElementById('slotMachineTitle');
const slotMachineBodyElem = document.getElementById('slotMachine_body');
const rewardTextElem = document.getElementById('rewardText');

let listMaster = document.createElement('ul');

listMaster.style.listStyle="none";
for(let i=0;i< nArms;i++){
    let l = document.createElement('li'); 

    l.style.backgroundColor = myColor(allMeans[i]);
    l.style.strokeWidth = '5px'
    l.style.fontWeight = 'bold'
    l.style.color='white'
    l.innerHTML = `SlotMachine #${i}`; 
    listMaster.appendChild(l)
}

banditsListElem.appendChild(listMaster);


const hueShiftAngle = [];

for(let i=0;i<nArms;i++){
    hueShiftAngle.push(Math.round(Math.random()*360))
}

function updateSlotMachine(action, reward, highlightInterval=1000){

    console.log("here");

    // shuffling slotMachine
    shuffle();

    slotMachineElem.style.border="2px solid red";
    rewardTextElem.innerHTML = (reward > 0)? "+"+reward.toFixed(2): reward.toFixed(2);
    rewardTextElem.style.color = myColor(allMeans[action]);

    let yCoord = 80;
    rewardTextElem.style.transform = `translate(150px, ${yCoord}px)`;
    rewardTextElem.style.opacity = 1;
    let rewardAnim = setInterval(()=>{
        rewardTextElem.style.transform = `translate(120px, ${yCoord}px)`;

        yCoord++;

        console.log('inside it', yCoord)
        if (yCoord === 110)
        clearInterval(rewardAnim)
    }, highlightInterval/200)


    setTimeout(()=>{

        slotMachineElem.style.border="";
        // rewardTextElem.innerHTML = "";

        let cOpactity = 1;
    let rewardAnimOpacity = setInterval(()=>{
        rewardTextElem.style.opacity = cOpactity;


        cOpactity -=.1;
        console.log('inside it', cOpactity)
        if (cOpactity < 0)
            clearInterval(rewardAnimOpacity)
    }, highlightInterval/200)

    }, highlightInterval);

    slotMachineTitleElem.innerHTML = "Slot Machine #"+action;
    // slotMachineTitleElem.style.backgroundColor = myColor(allMeans[action]);

    // casinoElem.style.webkitFilter = `hue-rotate(${hueShiftAngle[action]}deg`
    slotMachineBodyElem.style.fill = myColor(allMeans[action]);

    for(let i=0;i<nArms;i++){
        listMaster.childNodes[i].style.border = "";
        if(i === action)
            listMaster.childNodes[action].style.border = "5px solid red";
    }

    updateInterConnectionViz(action, highlightInterval);

}

function resetSlotMachine(){

    for(let i=0;i<nArms;i++){
        listMaster.childNodes[i].style.border = "";
    }
}


