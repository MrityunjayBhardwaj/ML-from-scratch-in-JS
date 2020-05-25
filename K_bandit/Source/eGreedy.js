function eGreedy(params={epsilon,valueFn,getReward, customValueFunction, nActions }){

    let valueFnFactory = {
            
        'sample-average': (cAction, cReward, estActionValues)=>{

            // console.log("cStep:", cStep)

            return  estActionValues[cAction] + (1/cStep)*(cReward - estActionValues[cAction]);
        },
        // TODO: add exponential recency averge + UCB
    }

    let model = {

        epsilon: params.epsilon ||  .2,
        estActionValues : [],
        nActions: params.nActions || 2,
      
        getReward: params.getReward || ((action) =>{

            return Math.random();
        }),
        valueFunction : (()=>{

            console.log('insidevalue fucn', params.valueFn)

            if(params.valueFn === undefined){
            // console.log('insidevalue fucn', params.valueFn, valueFnFactory['sample-average'])

             return valueFnFactory['sample-average'];
            }

            if(typeof(params.valueFn) === "string"){
             valueFnFactory[params.valueFn ];
            }

            return params.valueFn;
            
            })(),

        updateCallback: params.updateCallback || (()=>{}),

    }

    let updateAnalysis = {
        cEpisode: 0,
        cStep: 0,

        cRandomVal: 0,
        isGreedyStep: 0,

        cAction: 0,
        cReward: 0, 
    }

    let cStep = 1;

    this.getUpdateAnalysis = function(){
        return updateAnalysis;
    }

    this.getEstActionValue = function(){

        return model.estActionValues;

    }

    this.setEstActionValue = function(newEstActionValues){

        if(newEstActionValues.length)throw new Error('new action values much be equal to total no of action:'+nActions)

         model.estActionValues = newEstActionValues;
    }

    this.getModel = function(){
        return model;
    }

    this.updateRewards = function(cAction,cReward, valueFnParams ){

        model.estActionValues[cAction] = model.valueFunction(cAction, cReward, model.estActionValues);
        
    }

    this.update = ()=>{

        let cAction = 0;
        let cReward = 0;

        let randomVal = Math.random();
        let greedyAction = 0;

        if(model.epsilon >= randomVal){
            let maxEstVal = d3.max(model.estActionValues);

            cAction =(model.estActionValues.indexOf(maxEstVal));
            cReward = model.getReward(cAction);

            greedyAction = 1;

            this.updateRewards(cAction, cReward);
        }else{
            console.log('inside else')

            cAction = 0 + Math.floor(Math.random()*(Math.max(0,model.nActions)))
            cReward = model.getReward(cAction);

            this.updateRewards(cAction, cReward);
        }

        updateAnalysis.cAction = cAction;
        updateAnalysis.cReward = cReward;
        updateAnalysis.randomVal = randomVal;
        updateAnalysis.isGreedyStep = greedyAction;
        updateAnalysis.cStep = cStep;

        if(model.updateCallback){
            // console.log("casldkfj: ", cAction, cReward);
            model.updateCallback(cAction,cReward);
        }

        cStep++;

    }


    // initializing the model...
    function init(){

        // initialize all the action values to zero
        model.estActionValues = Array(model.nActions).fill(0);

    }

    init();

    

}