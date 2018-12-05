const TWO_PI = 2 * Math.PI;
const HALF_PI = Math.PI / 2;
const QUARTER_PI = Math.PI / 4;

const MAX_DEGREES = 360;
const HALF_DEGREES = MAX_DEGREES / 2;
const QUARTER_DEGREES = MAX_DEGREES / 4;

/**
 * A two-dimensional x and y coordinate point.
 */
class Point {
    /**
     * Sets x and y values.
     * @param x X value.
     * @param y Y value.
     */
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }

    /**
     * Calculate the distance between this point and the passed in point
     * @param point a Point object
     */
    distance(point) {
        distance(this.x, point.x, this.y, point.y)
    }

    /**
     * String representation.
     * @returns {string} Formatted string
     */
    toString() {
        return this.x.toString() + ', ' + this.y.toString();
    }
}

/**
 * Wrapper class to hold data for a priority queue.
 */
class PQNode {
    /**
     * Takes an item, item's priority value, and index
     * @param item The value to be stored
     * @param priority The item's priority value (affects queue placement)
     * @param index Index to store the node's place in priority queue heap array.
     */
    constructor(item, priority, index) {
        this.item = item;
        this.priority = priority;
        this.index = index;
    }

    /**
     * String representation.
     * @returns {string} A formatted string.
     */
    toString() {
        return this.item.toString();
    }
}

/**
 * A priority-based queue. Requires a callback function to determine priority of contents.
 */
class PriorityQueue {
    /**
     * Creates a PriorityQueue object with the passed in callback used to manage ordering of contents.
     * @param priorityCallback A callback function that takes a single parameter. Must return a numerical value, which
     * will then be used to determine an item's priority. This function will be called when an item is inserted to
     * determine it's priority value.
     */
    constructor(priorityCallback) {
        this.heap = [null];
        this.priorityCallback = priorityCallback;
    }

    /**
     * Inserts an item into the priority queue.
     * @param item An Object.
     */
    insert(item) {
        let index = this.size() + 1;
        let node = new PQNode(item, this.priorityCallback(item), index);
        let parent = this.parent(node);
        this.heap.push(node);

        while (parent !== null && node.priority < parent.priority) {

            // swap node and parent
            this.swapNodes(node, parent);

            // update parent
            parent = this.parent(node);
        }
    }

    /**
     * Removes and returns the highest-priority item in the queue.
     * @returns {*} A queue item.
     */
    remove() {
        if (!this.isEmpty()) {
            let root = this.peek();
            let node = this.heap[this.heap.length - 1];
            this.heap.pop();

            this.heap[root.index] = node;
            node.index = root.index;

            let leftChild = this.leftChild(node);
            let rightChild = this.rightChild(node);

            while ((this.hasLeftChild(node) && node.priority > leftChild.priority) || (this.hasRightChild(node)
                && node.priority > rightChild.priority)) {
                let children = this.children(node);
                let swapNode;

                if (children.length === 1) {
                    swapNode = children[0];
                } else if (leftChild.priority < rightChild.priority) {
                    swapNode = leftChild;
                } else {
                    swapNode = rightChild;
                }
                this.swapNodes(node, swapNode);

                leftChild = this.leftChild(node);
                rightChild = this.rightChild(node);
            }
            return root.item;
        }
    }

    /**
     * Returns, but does not remove, the highest priority item in the queue.
     * @returns {*} A queue item.
     */
    peek() {
        if (!this.isEmpty()) {
            return this.root();
        }
    }

    contains(item, callbackEquals) {
        for (let i = 1; i < this.heap.length; i++) {
            if ((isFunction(callbackEquals) && callbackEquals(this.heap[i].item, item))) {
                return true;
            } else if (item === this.heap[i]) {
                return true;
            }
        }

        return false;
    }

    isEmpty() {
        return this.size() === 0;
    }


    size() {
        return this.heap.length - 1;
    }

    root() {
        return this.heap[1];
    }

    parent(node) {
        return this.heap[Math.floor(node.index / 2)];
    }

    children(node) {
        let children = [];

        if (this.hasLeftChild(node)) {
            children.push(this.leftChild(node));
        }

        if (this.hasRightChild(node)) {
            children.push(this.rightChild(node));
        }

        return children;
    }

    leftChild(node) {
        let index = node.index * 2;
        if (index < this.heap.length) {
            return this.heap[index];
        }

        return null;
    }

    rightChild(node) {
        let index = node.index * 2 + 1;
        if (index < this.heap.length) {
            return this.heap[index];
        }

        return null;
    }

    hasLeftChild(node) {
        return this.leftChild(node) !== null;
    }

    hasRightChild(node) {
        return this.rightChild(node) !== null;
    }

    hasChildren(node) {
        return this.hasLeftChild(node) || this.hasRightChild(node);
    }

    swapNodes(node1, node2) {
        let temp = node1;
        let tempIndex = node1.index;

        this.heap[node1.index] = node2;
        this.heap[node2.index] = temp;

        node1.index = node2.index;
        node2.index = tempIndex;
    }

    toString() {
        return this.heap.toString();
    }
}

/**
 * This class allows color values to be dynamically adjusted instead of relying on the creation or modification of
 * color strings.
 *
 * Stores red, green, blue, and alpha values. Alpha is optional and assigned to 1 by default. To use the color value,
 * call the toString() method, which returns a formatted RGBA color string.
 *
 */
class RgbaColor {
    /**
     * Creates an RgbaColor object with assigned color values. Alpha is optional and defaults to 1. Color values
     * not in the range 0 <= x <= 255 will be assigned to the respective min or max value. This also applies to alpha
     * (0 <= x <= 1).
     * @param red Red value.
     * @param green Green value.
     * @param blue Blue value.
     * @param alpha Alpha (transparency) value.
     */
    constructor(red, green, blue, alpha = 1) {
        this.red = red;
        this.green = green;
        this.blue = blue;
        this.alpha = alpha;
    }

    /**
     * Randomizes the current color values. Does not change alpha value.
     */
    randomize() {
        const max = 255;
        this.red = randomInt(max, true);
        this.green = randomInt(max, true);
        this.blue = randomInt(max, true);
    }

    /**
     * Returns a string matching standard RGBA syntax for CSS, HTML, etc. E.g. with 0, 255, 100, .8, returns
     * 'rgba(0, 255, 100, .8)'.
     * @returns {string} Returns a formatted string.
     */
    toString() {
        return 'rgba(' + this.red + ', ' + this.green + ', ' + this.blue + ', ' + this.alpha + ')';
    }
}

RgbaColor.maxValue = 255;

function removeFromArray(array, item, isEqualCallback) {
    for (let i = 0; i < array.length; i++) {
        let currentItem = array[i];
        if (isEqualCallback(currentItem, item)) {
            array.splice(i, 1);
            return currentItem;
        }
    }

    return null;
}

function randomInt(max, isInclusive = false) {
    if (isInclusive) {
        max++;
    }

    return Math.floor(Math.random() * Math.floor(max));
}

function randomFloat(max) {
    return Math.random() * max;
}

function randomIntInRange(min, max, isInclusive = false) {
    min = Math.ceil(min);
    max = Math.floor(max);
    if (isInclusive) {
        max++;
    }
    return Math.floor(Math.random() * (max - min)) + min;
}

function randomFloatInRange(min, max) {
    return Math.random() * (max - min) + min;
}

function randomSign() {
    if (randomIntInRange(0, 2) === 0) {
        return -1;
    } else {
        return 1;
    }
}

function toRadians(degrees) {
    return degrees * Math.PI / HALF_DEGREES;
}

function toDegrees(angle) {
    return angle * (Math.PI / HALF_DEGREES);
}

function angleRadians(x1, y1, x2, y2) {
    return Math.atan2(y2 - y1, x2 - x1);
}

function angleDegrees(x1, y1, x2, y2) {
    return Math.atan2(y2 - y1, x2 - x1) * HALF_DEGREES / Math.PI;
}

function distance(x1, x2, y1, y2) {
    return Math.hypot(x1 - x2, y1 - y2);
}

/**
 * Creates dx and dy values based on angle and distance input.
 * @param angle Angle in radians.
 * @param distance Distance value.
 * @param isInvertedY Multiplies dy-value by -1 if true. (Should be true if traveling up across y-axis decreases the y-value).
 * @returns {{dx: number, dy: number}} Returns dx and dy values.
 */
function calculateDxDy(angle, distance, isInvertedY) {
    let dx = distance * Math.cos(angle);
    let dy = distance * Math.sin(angle);
    if (isInvertedY) {
        dy *= -1;
    }
    return {
        dx: dx,
        dy: dy
    };
}

/**
 * Creates dx and dy values based on angle and distance input. Automatically returns dy-value multiplied by -1 to
 * accommodate Canvas dimension system.
 * @param angle Angle in radians.
 * @param distance Distance value.
 * @returns {{dx: number, dy: number}} Returns dx and dy values (dy is multiplied by -1).
 */
function calculateDxDyCanvas(angle, distance) {
    return calculateDxDy(angle, distance, true);
}

function isInBounds1d(array, index) {
    return index >= 0 && index < array.length
}

function isInBounds2d(array, row, col) {
    return isInBounds1d(array, row) && isInBounds1d(array[0], col)
}

function isFunction(value) {
    return typeof value === 'function';
}

function arrayNeighbors(array, row, col, offset = 0) {
    // let iterations = 3;
    let iterations = 3 + offset * 2;
    let neighbors = [];
    for (let i = 0; i < iterations; i++) {
        for (let j = 0; j < iterations; j++) {
            let rowIndex = row - 1 - offset + i;
            let colIndex = col - 1 - offset + j;

            if (isInBounds2d(array, rowIndex, colIndex) && (rowIndex !== row || colIndex !== col)) {
                neighbors.push(array[rowIndex][colIndex]);
            }
        }
    }

    return neighbors;
}

function isDiagonalNeighbor(cell, neighborCell) {
    let maxDifference = 1;
    return cell.row !== neighborCell.row && cell.col !== neighborCell.col
        && Math.abs(cell.row - neighborCell.row) <= maxDifference
        && Math.abs(cell.col - neighborCell.col) <= maxDifference;
}

function isAdjacentCell(row1, col1, row2, col2) {
    return Math.abs(row1 - row2) <= 1 && Math.abs(col1 - col2) <= 1;
}

function isArrayContains(array, item, equalsCallback) {
    for (let i = 0; i < array.length; i++) {
        if (equalsCallback(array[i], item)) {
            return true;
        }
    }

    return false;
}

function roundDecimals(value, decimals) {
    return Number(Math.round(value + 'e' + decimals) + 'e-' + decimals);
}

function isAABBCollision(x1, y1, width1, height1, x2, y2, width2, height2) {
    return x1 < x2 + width2 && x1 + width1 > x2 && y1 < y2 + height2 && y1 + height1 > y2;
}

function mousePositionCanvas(canvas, event) {
    let rect = canvas.getBoundingClientRect();
    return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
    };
}