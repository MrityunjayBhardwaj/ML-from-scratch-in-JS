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
