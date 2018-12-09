/**
 * Applies a toggleable shadow to the passed in context.
 */
class Shadow {
    /**
     * Creates a new shadow object.
     * @param context A 2D canvas context.
     * @param shadowOffsetX Offset the shadow on x-axis.
     * @param shadowOffsetY Offset the shadow on y-axis.
     * @param shadowColor A color String or RgbaColor object.
     * @param shadowBlur Blur value. Note: can cause significant performance drops.
     * @param isActive Shadow on or off.
     */
    constructor(context, shadowOffsetX, shadowOffsetY, shadowColor, shadowBlur = 0, isActive = true) {
        this.context = context;
        this.shadowOffsetX = shadowOffsetX;
        this.shadowOffsetY = shadowOffsetY;
        this.shadowColor = shadowColor;
        this.shadowBlur = shadowBlur;
        this.isActive = isActive;
    }

    /**
     * Turns on the shadow.
     */
    activate() {
        this.context.shadowOffsetX = this.shadowOffsetX;
        this.context.shadowOffsetY = this.shadowOffsetY;
        this.context.shadowColor = this.shadowColor;
        this.context.shadowBlur = this.shadowBlur;
        this.isActive = true;
    }

    /**
     * Turns off the shadow.
     */
    deactivate() {
        this.context.shadowOffsetX = undefined;
        this.context.shadowOffsetY = undefined;
        this.context.shadowColor = undefined;
        this.context.shadowBlur = undefined;
        this.isActive = false;
    }
}

/**
 * Functions as a base class to be extended to draw shapes. If used by itself, class functions as a point (single
 * pixel). Provides methods to move and rotate the object.
 */
class CanvasObject {
    /**
     * Basic 1-pixel shape.
     * @param context A 2D canvas context.
     * @param x X value.
     * @param y Y value.
     * @param direction Movement angle in radians.
     * @param speed Pixels to move per update.
     * @param angle Rotation angle in radians.
     * @param rotation Rotation per update in radians.
     * @param color Color value (string or RgbaColor Object).
     * @param shadow Shadow object.
     */
    constructor(context, x, y, direction, speed, angle, rotation, color, shadow = null) {
        this.context = context;
        this.x = x;
        this.y = y;
        this.direction = direction;
        this.speed = speed;
        this.angle = angle;
        this.rotation = rotation;
        this.dx = 0;
        this.dy = 0;
        this.color = color;
        this.shadow = shadow;

        this.updateDxDy();
    }

    /**
     * Sets direction value.
     * @param direction A direction angle in radians.
     */
    setDirection(direction) {
        this.direction = direction;
        this.updateDxDy();
    }

    /**
     * Sets speed value.
     * @param speed A speed value in pixels.
     */
    setSpeed(speed) {
        this.speed = speed;
        this.updateDxDy();
    }

    /**
     * Updates dx and dy values based on current direction and speed.
     */
    updateDxDy() {
        let moveData = calculateDxDy(this.direction, this.speed);
        this.dx = moveData.dx;
        this.dy = moveData.dy;
    }

    /**
     * Inverts the current rotation direction. (e.g. clockwise would become counterclockwise).
     */
    invertRotation() {
        this.rotation *= -1;
    }

    /**
     * Moves object by updating x and y values by adding respective dx and dy values.
     */
    move() {
        this.x += this.dx;
        this.y += this.dy;
    }

    /**
     * Activates shadow object if available. Shadow drawn in draw() method.
     */
    drawShadow() {
        if (this.shadow && !this.shadow.isActive) {
            this.shadow.activate();
        }
    }

    /**
     * Draws the object on the canvas.
     */
    draw() {
        this.drawShadow();
        this.context.fillStyle = this.color.toString();
        this.context.fillRect(this.x, this.y, 1, 1);
    }

    /**
     * Updates the current angle based on current rotation value or passed in value.
     * @param angle A new angle value.
     */
    updateAngle(angle = null) {
        if (angle || angle === 0) {
            this.angle = angle;
        } else {
            this.angle += this.rotation;
        }
    }

    /**
     * Moves and draws the object.
     */
    update() {
        this.updateAngle();
        this.move();
        this.draw();
    }

    /**
     * Checks if the object is in bounds of the passed in canvas. Use CanvasObject.bounds values to change bounds check.
     * @param canvas A Canvas object.
     * @param boundsCheck Type of bounds check to perform.
     * @returns {boolean} Returns true if in canvas bounds, else false.
     */
    isInBounds(canvas, boundsCheck = CanvasObject.bounds.whole) {
        return this.x >= 0 && this.x < canvas.width && this.y >= 0 && this.y < canvas.height;
    }
}

/**
 * Types of boundary checks.
 * @type {{edge: number, whole: number}}
 */
CanvasObject.bounds = {
    /**
     * Edge bounds check (shape is out of bounds if edge out of bounds).
     */
    edge: 0,
    /**
     * Whole bounds check (whole shape must be out of bounds).
     */
    whole: 1
};

/**
 * An Object for drawing circles on the canvas.
 */
class Circle extends CanvasObject {
    /**
     * Creates a new Circle object.
     * @param context A 2D canvas context
     * @param x X value.
     * @param y Y value.
     * @param radius Circle radius in pixels.
     * @param direction Movement angle in radians.
     * @param speed Pixels to move per update.
     * @param angle Rotation angle in radians.
     * @param rotation Rotation per update in radians.
     * @param color Color value (string or RgbaColor Object).
     * @param shadow Shadow object.
     */
    constructor(context, x, y, radius, direction, speed, angle, rotation, color, shadow = null) {
        super(context, x, y, direction, speed, angle, rotation, color, shadow);
        this.radius = radius;
    }

    /**
     * Draws the circle.
     */
    draw() {
        this.drawShadow();
        drawArc(this.context, this.x, this.y, this.radius, 0, TWO_PI, false,
            this.color.toString(), true);
    }

    /**
     * Gets a coordinate point from the circle's center point based on the chosen and distance from the center.
     * Chosen distance can exceed the radius of the circle.
     * @param angle The angle in radians from the circle's center point.
     * @param distance The pixel distance from the circle's center point.
     * @returns {Point} Returns a Point for the chosen point.
     */
    getPoint(angle = 0, distance = this.radius) {
        let offset = calculateDxDy(angle, distance);
        return new Point(this.x + offset.dx, this.y + offset.dy);
    }

    /**
     * Checks if the object is in bounds of the passed in canvas. Use CanvasObject.bounds values to change bounds check.
     * @param canvas A Canvas object.
     * @param boundsCheck Type of bounds check to perform.
     * @returns {boolean} Returns true if in canvas bounds, else false.
     */
    isInBounds(canvas, boundsCheck = CanvasObject.bounds.whole) {
        if (boundsCheck === CanvasObject.bounds.whole) {
            return this.x + this.radius >= 0 && this.x - this.radius < canvas.width
                && this.y + this.radius >= 0 && this.y - this.radius < canvas.height;
        } else {
            return this.x - this.radius >= 0 && this.x + this.radius < canvas.width
                && this.y - this.radius >= 0 && this.y + this.radius < canvas.height;
        }
    }
}

/**
 * An object for drawing rectangles on the canvas.
 */
class Rectangle extends CanvasObject {
    /**
     * Creates a new Rectangle object.
     * @param context A 2D canvas context.
     * @param x X value.
     * @param y Y value.
     * @param width Width in pixels.
     * @param height Height in pixels.
     * @param direction Movement angle in radians.
     * @param speed Pixels to move per update.
     * @param color Color value. A color String or RgbaColor value.
     * @param shadow Shadow object.
     */
    constructor(context, x, y, width, height, direction, speed, color, shadow = null) {
        super(context, x, y, direction, speed, 0, 0, color, shadow);
        this.width = width;
        this.height = height;
    }

    /**
     * Draws the rectangle.
     */
    draw() {
        this.drawShadow();
        this.context.fillStyle = this.color.toString();
        this.context.fillRect(this.x, this.y, this.width, this.height);
    }

    /**
     * Middle x value.
     * @returns {*} Returns the middle x value.
     */
    getMiddleX() {
        return this.x + this.width / 2;
    }

    /**
     * Middle y value.
     * @returns {*} Returns the middely y value.
     */
    getMiddleY() {
        return this.y + this.height / 2;
    }

    /**
     * Checks if the object is in bounds of the passed in canvas. Use CanvasObject.bounds values to change bounds check.
     * @param canvas A Canvas object.
     * @param boundsCheck Type of bounds check to perform.
     * @returns {boolean} Returns true if in canvas bounds, else false.
     */
    isInBounds(canvas, boundsCheck = CanvasObject.bounds.whole) {
        if (boundsCheck === CanvasObject.bounds.whole) {
            return this.x + this.width >= 0 && this.x < canvas.width && this.y + this.height >= 0
                && this.y < canvas.height;
        } else {
            return this.x >= 0 && this.x + this.width < canvas.width && this.y >= 0
                && this.y + this.height < canvas.height;
        }
    }
}

// class TrailedCircle extends Circle {
//     constructor(context, x, y, radius, dx, dy, color, shadow, trailColor, trailLength, isDynamic = false, trailRatio = 1) {
//         super(context, x, y, radius, dx, dy, color, shadow);
//         this.isDynamic = isDynamic;
//         this.trailRatio = trailRatio;
//         this.trailColor = trailColor;
//         this.baseTrailLength = trailLength;
//         this.trailLength = this.baseTrailLength;
//         this.updateTrailLength();
//     }
//
//     draw() {
//         if (this.shadow) {
//             this.shadow.activate();
//         }
//         let trail = this.getTrail();
//         if (trail.lengthValue > 0) {
//             drawTriangle(this.context, trail.tip.x, trail.tip.y, trail.side1.x, trail.side1.y, trail.side2.x, trail.side2.y, this.trailColor.toString(), true);
//         }
//         drawArc(this.context, this.x, this.y, this.radius, 0, TWO_PI, false, this.color.toString(), true);
//     }
//
//     update() {
//         this.x += this.dx;
//         this.y += this.dy;
//         this.updateTrailLength();
//         this.draw();
//     }
//
//     updateTrailLength() {
//         if (this.isDynamic) {
//             this.trailLength = Math.hypot(this.y - this.y + this.dy, this.x - this.x + this.dx) * this.trailRatio;
//         } else {
//             this.trailLength = this.baseTrailLength;
//         }
//     }
//
//     getTrail() {
//         let angle = angleRadians(this.x, this.y, this.x + this.dx, this.y + this.dy) + Math.PI;
//
//         let tipX = this.x + this.trailLength * Math.cos(angle);
//         let tipY = this.y + this.trailLength * Math.sin(angle);
//
//         let side1X = this.x + this.radius * Math.cos(angle + HALF_PI);
//         let side1Y = this.y + this.radius * Math.sin(angle + HALF_PI);
//
//         let side2X = this.x + this.radius * Math.cos(angle + Math.PI + HALF_PI);
//         let side2Y = this.y + this.radius * Math.sin(angle + Math.PI + HALF_PI);
//
//         return {
//             lengthValue: this.trailLength,
//             angle: angle,
//             tip: new Point(tipX, tipY),
//             side1: new Point(side1X, side1Y),
//             side2: new Point(side2X, side2Y)
//         }
//     }
//
//     isInBounds(canvas) {
//         let trail = this.getTrail();
//         return (this.x + this.radius >= 0 && this.x - this.radius < canvas.width && this.y + this.radius >= 0 && this.y - this.radius < canvas.height)
//             || (trail.tip.x >= 0 && trail.tip.x < canvas.width && trail.tip.y >= 0 && trail.tip.y < canvas.height);
//     }
// }

/**
 * Draws a timed spark circle on the canvas. Note: spark persists when timer is over and should ideally be removed at
 * this point.
 */
class Spark extends Circle {
    /**
     * Creates a new Spark object.
     * @param context A 2D canvas context
     * @param x X value.
     * @param y Y value.
     * @param radius Circle radius in pixels.
     * @param direction Movement angle in radians.
     * @param speed Pixels to move per update.
     * @param angle Rotation angle in radians.
     * @param rotation Rotation per update in radians.
     * @param color Color value (string or RgbaColor Object).
     * @param durationMs Spark active time in ms.
     * @param shadow Shadow object.
     */
    constructor(context, x, y, radius, direction, speed, angle, rotation, color, durationMs, shadow = null) {
        super(context, x, y, direction, speed, angle, rotation, color, shadow);
        this.ms = 0;
        this.endTime = performance.now() + durationMs;
        this.halfTime = this.endTime / 2;
    }

    /**
     * Updates alpha value, then moves and draws spark, then updates ms value.
     */
    update() {
        if (this.ms > this.halfTime) {
            this.color.alpha = (this.ms - this.halfTime) / (this.endTime - this.halfTime);
        }

        this.move();
        this.draw();
        this.ms++;
    }

    /**
     * Checks if the spark's active time value is still under the time limit.
     * @returns {boolean} Returns true if the time active is less than the active time limit, else false.
     */
    isActive() {
        return this.ms < this.endTime;
    }
}

/**
 * Utility method to set strokeStyle and fillStyle to the passed in parameter values.
 * @param context A 2D canvas context.
 * @param color A color String or RgbaColor value.
 */
function prepare(context, color) {
    context.strokeStyle = color;
    context.fillStyle = color.toString();
    context.beginPath();
}

/**
 * Finalizes a canvas path by calling fill(), stroke(), and closePath() as specified in parameters.
 * @param context A 2D canvas context.
 * @param isFilled Fill the path.
 * @param isClosed Close the path.
 */
function finalize(context, isFilled, isClosed) {
    if (isFilled) {
        context.fill();
    } else if (!isFilled && isClosed) {
        context.closePath();
        context.stroke();
    } else {
        context.stroke();
    }
}

/**
 * Draws a line.
 * @param context A 2D canvas context.
 * @param x1 Starting x value.
 * @param y1 Starting y value.
 * @param x2 Ending x value.
 * @param y2 Ending y value.
 * @param color The line color. A color String or RgbaColor object.
 */
function drawLine(context, x1, y1, x2, y2, color) {
    context.strokeStyle = color.toString();
    context.beginPath();
    context.moveTo(x1, y1);
    context.lineTo(x2, y2);
    context.stroke();
}

/**
 * Draws a filled or unfilled rectangle.
 * @param context A 2D canvas context.
 * @param x X value.
 * @param y Y value.
 * @param width Rectangle width in pixels.
 * @param height Rectangle height in pixels.
 * @param color The rectangle color. A color String or RgbaColor object.
 * @param isFilled Fill the rectangle.
 */
function drawRect(context, x, y, width, height, color, isFilled) {
    prepare(context, color);

    if (isFilled) {
        context.fillRect(x, y, width, height);
    } else {
        context.strokeRect(x, y, width, height);
    }
}

/**
 * Draws a filled or unfilled triangle.
 * @param context A 2D canvas context.
 * @param x1 Corner x value.
 * @param y1 Corner y value.
 * @param x2 Corner x value.
 * @param y2 Corner y value.
 * @param x3 Corner x value.
 * @param y3 Corner y value.
 * @param color The triangle color. A color String or RgbaColor object.
 * @param isFilled Fill the triangle.
 */
function drawTriangle(context, x1, y1, x2, y2, x3, y3, color, isFilled) {
    prepare(context, color);

    context.moveTo(x1, y1);
    context.lineTo(x2, y2);
    context.lineTo(x3, y3);

    finalize(context, isFilled, true);
}

/**
 * Draws a filled or unfilled arc.
 * @param context A 2D canvas context.
 * @param x Center point x value.
 * @param y Center point y value.
 * @param radius Radius in pixels.
 * @param startAngle Starting angle in radians.
 * @param endAngle Ending angle in radians.
 * @param anticlockwise Draw arc anticlockwise.
 * @param color The arc color. A color String or RgbaColor object.
 * @param isFilled Fill the arc.
 * @param isClosed Close the arc (connect starting and ending points with a line).
 */
function drawArc(context, x, y, radius, startAngle, endAngle, anticlockwise, color, isFilled, isClosed) {
    prepare(context, color);

    context.arc(x, y, radius, startAngle, endAngle, anticlockwise);

    finalize(context, isFilled, isClosed);
}

/**
 * Draws a cubic bezier curve.
 * @param context A 2D canvas context.
 * @param x1 Starting x value.
 * @param y1 Starting y value.
 * @param x2 Ending x value.
 * @param y2 Ending y value.
 * @param cp1x Control point 1 x value.
 * @param cp1y Control point 1 y value.
 * @param cp2x Control point 2 x value.
 * @param cp2y Control point 2 y value.
 * @param color Line color. A color String or RgbaColor object.
 * @param isFilled Fill the points.
 * @param isClosed Close the line.
 */
function drawCubicBezierCurve(context, x1, y1, x2, y2, cp1x, cp1y, cp2x, cp2y, color, isFilled, isClosed) {
    prepare(context, color);

    context.moveTo(x1, y1);
    context.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x2, y2);

    finalize(context, isFilled, isClosed);
}

/**
 * Draws a custom shape based on passed in array of coordinates.
 * @param context A 2D canvas context.
 * @param points An array of points to connect lines.
 * @param color The color of the shape or line. A color String or RgbaColor object.
 * @param isFilled Fill the shape.
 * @param isClosed Close the line.
 */
function drawPolygon(context, points, color, isFilled, isClosed) {
    if (points.length > 0) {
        let point = points[0];

        prepare(context, color);

        context.moveTo(point.x, point.y);
        for (let i = 1; i < points.length; i++) {
            point = points[i];
            context.lineTo(point.x, point.y);
        }

        finalize(context, isFilled, isClosed);
    }
}

/**
 * Sets the width and height of the passed in canvas.
 * @param canvas A canvas object.
 * @param width Width in pixels.
 * @param height Height in pixels.
 */
function setCanvasSize(canvas, width, height) {
    canvas.width = width;
    canvas.height = height;
}