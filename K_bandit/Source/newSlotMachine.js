let banditsListElem = document.getElementById('banditsList');
const casinoElem = document.getElementById('casino');
const slotMachineTitleElem = document.getElementById('slotMachineTitle');
const slotMachineBodyElem = document.getElementById('slotMachine_body');

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

function updateSlotMachine(action){

    slotMachineTitleElem.innerHTML = "Slot Machine #"+action;
    // slotMachineTitleElem.style.backgroundColor = myColor(allMeans[action]);

    // casinoElem.style.webkitFilter = `hue-rotate(${hueShiftAngle[action]}deg`
    slotMachineBodyElem.style.fill = myColor(allMeans[action]);

    for(let i=0;i<nArms;i++){
        listMaster.childNodes[i].style.border = "";
        if(i === action)
            listMaster.childNodes[action].style.border = "5px solid red";
    }
}


