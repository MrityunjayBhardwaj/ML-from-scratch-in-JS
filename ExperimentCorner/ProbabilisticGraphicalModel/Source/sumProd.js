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




