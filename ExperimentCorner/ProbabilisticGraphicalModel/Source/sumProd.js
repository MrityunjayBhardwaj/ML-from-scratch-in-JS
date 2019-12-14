/**
 * 
 * @constructor
 * @param {Number} name name of this node
 */

function Node(name){
    this.connections = [];
    this.inbox = [];
    this.name = name;

}

/**
 * @param {Object} toNode pass a node object with which you want to make a connection.
 * @description connect this node to 'toNode' and vice-versa. 
 */
Node.prototype.connect = function(toNode){

    this.connections.push(toNode);
    toNode.connections.push(this);
}

/**
 * @param {Number} stepNum current iteration number of our sum-product agorithm
 * @param {Message} message message
 * @description this function takes in the message and insert it into the inbox of this node
 */
Node.prototype.deliver = function(stepNum,message) {
    if (this.inbox[stepNum]){
        this.inbox[stepNum].push(message);
    }
    else{
        this.inbox[stepNum] = [message];
    }
}

/**
 * @constructor
 * @param {Node} fromNode who is the sender of this message
 * @param {Number} val value of the message
 * 
 */
function Message(fromNode, val){
    this.fromNode = fromNode;
    this.val = val.div(tf.sum(val));
}

/**
 * 
 * @param {String} name name of this factor node
 * @param {tf.tensor} potentials tensor of potential for this factor node
 */
function Factor(name, potentials){

    // Inheriting the methods from Node Object
    Node.call(this, name);

   this.potential = potentials;

}

// extending the Factor to Node object.
Factor.prototype = Object.create(Node.prototype);

// use the constructor of Factor function.
Factor.prototype.constructor = Factor;

/**
* @param {Node} recipient Node Object
* 
* @description creates a message for this factor node  which then gets delivered to the given recipient node
*/
Factor.prototype.makeMessage = function(recipient){
    /**
    * following the log rule from 5.1.4 from BRML by david barber.
    */

    if (this.connections.length > 1){
        const originalMessage = this.inbox[this.inbox.length-1]; // gather all the message from the last step 


        // collect all the messages that are not from the recipient node
        let messages = originalMessage.filter((msg) => {if(msg.fromNode !== recipient)return msg});

        let allMessages = messages.map((m) =>{return this.reformatMessage(m)}); // so that we preserve the shape infromation of our values when we send this message.
        allMessages = tf.tensor(allMessages);

        const lambdas = tf.log(allMessages);

        let maxLambda = tfNan2Num(lambdas);
        maxLambda = tf.max(maxLambda);

        const result = tf.sum(lambdas, axis=0).sub(maxLambda);
        const productOutput = tf.mul(this.potential, tf.exp(result));
        return tf.exp(
            tf.log(
                        this.summation(productOutput, recipient)
                ).add(maxLambda)
            );
    }
    else{
        return this.summation(this.potential, recipient);
    }

};

Factor.prototype.maxProductMessage = function(recipient){
    if (this.connections.length > 1){
        const originalMessage = this.inbox[this.inbox.length-1]; // gather all the message from the last step 


        // collect all the messages that are not from the recipient node
        let messages = originalMessage.filter((msg) => {if(msg.fromNode !== recipient)return msg});

        let allMessages = messages.map((m) =>{return this.reformatMessage(m)}); // so that we preserve the shape infromation of our values when we send this message.
        allMessages = tf.tensor(allMessages);

        const lambdas = tf.log(allMessages);

        let maxLambda = tfNan2Num(lambdas);
        maxLambda = tf.max(maxLambda);

        const result = tf.sum(lambdas, axis=0).sub(maxLambda);
        const productOutput = tf.mul(this.potential, tf.exp(result));
        return tf.exp(
            tf.log( 
                        this.maximum(productOutput, recipient)
                ).add(maxLambda)
            );
    }
    else{
        return tf.tensor(this.maximum(this.potential, recipient));
    }

}

/**
* @param {Message} message Message Object
* 
* @description 
    Returns the given message's val reformatted to be the same
    dimensions as self.potential, ensuring that message's values are
    expanded in the correct axes.
*/
Factor.prototype.reformatMessage = function(message){

    const dims =  this.potential.shape;
    const states = message.val.flatten().arraySync(); // TODO: solve this for multi-variate distribution case
    const whichAxis  = this.connections.indexOf(message.fromNode); // select the axis of the node who sends this message.

    const acc = tf.ones(dims).arraySync();

    return tensorMap(acc, dims, function(whichDim, states, val, coord) {
        const i = coord[whichAxis];
        return val*states[i];

    }.bind(this,whichAxis, states)).tensor;
};

/**
* @param {tf.tensor} p values
* @param {Node} node
* 
* @summary sum the given tensor('p') over the axis of the given node
*/
Factor.prototype.summation = function(p, node){

    const dims = p.shape;
    const states = p.flatten().arraySync();
    const whichAxis  = this.connections.indexOf(node); // select the axis of the node who sends this message.
    const out = tf.zeros([1, node.size]).flatten().arraySync();

    for (coord of ndIndex(dims)){
        i = coord[whichAxis];
        out[i] += states[coord2Index(coord, dims)];

    }

    return tf.tensor(out);
};

Factor.prototype.maximum = function(p, node){
    // take the maximum over all the axis accept the axis of this node
    const dims = p.shape;
    const whichAxis  = this.connections.indexOf(node); // select the axis of the node who sends this message.
    const allAxis = tf.linspace(0, dims.length, dims.length).flatten().arraySync();
    const maxAxis = allAxis.slice(whichAxis, 1);

    return tf.max(p, maxAxis);
}





function Variable(name,size){
    Node.call(this, name);

    this.bfmarginal = NaN;
    this.size = size;
    
}

// extending the Variable to Node object.
Variable.prototype = Object.create(Node.prototype);

// use the constructor of Variable function.
Variable.prototype.constructor = Variable;



/**
 * 
 * @summary this function marginalize w.r.t this variable by using message from all its connections in the current iteration step.
 */
Variable.prototype.marginal = function(){
    // here, we also used the max(log) trick. in order to prevent any numberical unstablilities.
    if ( this.inbox.length){
        const messages = this.inbox[this.inbox.length -1];
        const logVals = tf.log(messages.map((m) => {return m.val.arraySync()}));
        const validLogVals = tfNan2Num(logVals);
        const sumLogs = tf.sum(validLogVals, axis=0);
        const validSumLogs = sumLogs.sub(tf.max(sumLogs));// IMPORTANT for numerical stabality
        const prod = tf.exp(validSumLogs);

        return prod.div(tf.sum(prod));
    }
    else{
        // for first iteration use the simple uniform distribution.
        return tf.ones([1, this.size]).div(this.size);
    }
}

Variable.prototype.maximalState = function(){
    return tf.max(this.marginal());
}

/**
 * @param {Factor} recipient reciver of this message.
 * @summary create a message by summing over all the messages that are not from the recipient Factor node.
 */
Variable.prototype.makeMessage = function(recipient){
    if (this.connections.length > 1){

        const originalMessages = this.inbox[this.inbox.length -1];
        // collect all the messages that are not from the recipient node
        const messages = originalMessages.filter((msg) => {if(msg.fromNode !== recipient)return msg});

        // summing over this variable
        const logVals = tf.tensor( messages.map( function(m){return tf.log(m.val).arraySync();} ));
        return tf.exp(
                tf.sum(logVals, axis=0) 
        );
    }
    else{
        // if there is only one factor connected to this node, then just propagate the probability =1 
        return tf.ones([1, this.size]);
    }

}

function FactorGraph(firstNode=NaN, silent=false, debug=false){

    this.nodes = {};

    // adding dict like functionality
    Object.setPrototypeOf(this.nodes, {
        getKeys : 
            function(){
                const keys = [];
                // console.log(this);
                for(let k in this.nodes){
                    
                    if (this.nodes.hasOwnProperty(k))
                        keys.push(k);
                }
                return keys
            }.bind(this)
        ,
        getValues :
            function(){
                const vals = [];
                for(let k in this.nodes){

                    if (this.nodes.hasOwnProperty(k))
                        vals.push(this.nodes[k]);
                }
                return vals;
            }.bind(this)
    });



    this.isSilent = silent;
    this.debug = debug;

    if (firstNode){
        this.nodes[firstNode.name] = firstNode;
    }

    this.add = function(node){
        this.nodes[node.name] = node;
    }
    this.connect = function(name1, name2){
        this.nodes[name1].push(this.nodes[name2]);
    }
    this.append = function(fromNodeName, toNode){
        const toNodeName = toNode.name;

        if (!(this.nodes[toNodeName])){
            this.nodes[toNodeName] = toNode;
        }
        this.nodes[fromNodeName].connect(this.nodes[toNodeName]);

        return this;
    }

    this.leafNode = function(){
        return this.nodes.getValues().filter((node) => {if(node.connections === 1)return node});
    }

    // TODO: add observation support

    this.observe = async function(nameName, state){

        for(let j=0;j<nameName.length;j++){

            const node = this.nodes[nameName[j]];

            const factors = node.connections.filter((node) => {if (node instanceof Factor)return node;});

            if (state>=node.size)
                throw new Error('specified state must not exceed the size of the node');

            const factorArray = [];
            for(let factor of factors){


            }
        }

    }
    this.exportMarginals = function(){
         
        const retVal = {};
        for(let n in this.nodes){

            const cNode = this.nodes[n];
            if ( cNode instanceof Variable){
                retVal[cNode.name] = cNode.marginal();
                cNode.bfmarginal = retVal[cNode.name];
            }
        }

        // console.log(retVal);
        return retVal;

    }
    this.compareMarginals = function(marginalA, marginalB){


        let sum = 0;
        for(let nodeName in marginalA){

            if (marginalA[nodeName] && marginalB[nodeName])
                sum +=tf.sum(tf.abs(marginalA[nodeName].sub(marginalB[nodeName]))).flatten().arraySync()[0];
        }

        return sum;
    }
    this.computeMarginals = function(maxItrs=10, tolerance=1e-3, errorFunc){
        // belief propagation

        // for keeping track of the state
        let marginalDiffs = [1];
        let step = 0;


        // clearning the inbox before all the calculations.
        for (let node in this.nodes){
            this.nodes[node].inbox = []; 
        }

        let currMarginals = this.exportMarginals();

        // initialization
        for(let nodeKey in this.nodes){
            const node = this.nodes[nodeKey];
            if(  node instanceof Variable ){
                const message = new Message(node, tf.ones([1, node.size]));
                for(let recipient of node.connections){
                    recipient.deliver(step, message);
                }
            }

        }


        // TODO: add belief propagation

        while( step < maxItrs && tolerance < marginalDiffs[marginalDiffs.length -1]){
            const lastMarginals = currMarginals;
            step += 1
            if (!this.silent){
            }

            const factors = this.nodes.getValues().filter((node) => {if(node instanceof Factor)return node});
            const variables = this.nodes.getValues().filter((node) => {if(node instanceof Variable)return node});

            const senders = factors.concat(variables);// collect factors first then variables.

            for( sender of senders){
                const nextRecipients = sender.connections;
                for (recipient of nextRecipients){
                    const val = sender.makeMessage(recipient);
                    const message = new Message(sender, val);
                    if(this.debug){
                        console.log( 
                            step + " " + sender.name + " --> " +   recipient.name + 
                            " message: "+ message.val.arraySync()
                        );
                    }
                    recipient.deliver(step, message);
                }
            }
            currMarginals = this.exportMarginals();

            if (errorFunc){
                marginalDiffs.push(
                    errorFunc(currMarginals, lastMarginals)
                );

            }else{
                marginalDiffs.push(
                    this.compareMarginals(currMarginals, lastMarginals)
                );

            }
            if (1){

                console.log(marginalDiffs[marginalDiffs.length -1]);
            }
        }

        // console.log(currMarginals);

        window.currMarginals = currMarginals;

        return marginalDiffs.slice(1); // skipping only the first entry.

    }

    // this.query(nodeName,evidence) = () =>{


    // }
    

}