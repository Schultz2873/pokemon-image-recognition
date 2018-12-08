class PokeBall extends Circle {
    constructor(context, x, y, radius, direction, shadow = null) {
        let speedModifier = .02;
        let speed = radius * speedModifier;
        let angle = randomFloatInRange(0, TWO_PI);

        super(context, x, y, radius, direction, speed, angle, 0, 'rgba(255, 0, 65, 1)', shadow);
        this.color2 = 'white';
        this.outline = 'black';
        this.rotationModifier = .005 * randomSign();
        this.speedModifier = speedModifier;

        this.updateRotation();
    }

    draw() {
        let endAngle = this.angle + Math.PI;
        let buttonRadiusOuter = Math.round(this.radius * .3);
        let buttonRadiusInner = Math.round(buttonRadiusOuter * .65);

        // set line width and stroke style
        this.context.lineWidth = this.radius * .05;
        this.context.strokeStyle = this.outline;

        // top section
        this.context.fillStyle = this.color;
        this.context.beginPath();
        this.context.arc(this.x, this.y, this.radius, this.angle, endAngle, true);
        this.context.fill();
        this.context.stroke();

        // bottom section
        this.context.fillStyle = this.color2;
        this.context.beginPath();
        this.context.arc(this.x, this.y, this.radius, this.angle, endAngle, false);
        this.context.closePath();
        this.context.fill();
        this.context.stroke();

        // button - outer
        this.context.fillStyle = this.color2;
        this.context.beginPath();
        this.context.arc(this.x, this.y, buttonRadiusOuter, 0, TWO_PI);
        this.context.fill();
        this.context.stroke();

        // button - inner
        this.context.lineWidth *= .8;
        this.context.beginPath();
        this.context.arc(this.x, this.y, buttonRadiusInner, 0, TWO_PI);
        this.context.stroke();

        this.context.lineWidth = null;
    }

    updateRotation(rotation) {
        if (rotation || rotation === 0) {
            this.rotation = rotation;
        } else {
            this.rotation = this.speed * this.rotationModifier;
        }
    }

    update() {
        this.updateRotation();
        this.updateAngle();
        this.move();
        this.draw();
    }
}

PokeBall.addToArray = function (array, pokeBall) {
    let i = 0;
    while (i < array.length && array[i].radius < pokeBall.radius) {
        i++;
    }

    array.splice(i, 0, pokeBall);
};

const pokeBallCanvas = function () {
    let canvas = document.querySelector('#poke-ball-canvas');
    let context = canvas.getContext('2d');

    setCanvasSize(canvas, innerWidth, innerHeight);

    let mouse = null;
    let isInitialSpawn = true;
    let pokeBalls = [];
    let maxPokeBalls = 12;
    let minRadius = 0;
    let maxRadius = 0;
    let directionOffset = toRadians(15);

    window.addEventListener('mousemove', function (event) {
        mouse = mousePositionCanvas(canvas, event);
    });

    function init() {
        setCanvasSize(canvas, innerWidth, innerHeight);
        isInitialSpawn = true;
        pokeBalls = [];
        minRadius = Math.floor(Math.min(canvas.width, canvas.height) * .03);
        maxRadius = Math.floor(Math.min(canvas.width, canvas.height) * .2);
    }

    function animate() {
        requestAnimationFrame(animate);
        context.clearRect(0, 0, canvas.width, canvas.height);

        // remove poke balls that are outside of the canvas dimensions
        for (let i = 0; i < pokeBalls.length; i++) {
            if (!pokeBalls[i].isInBounds(canvas)) {
                pokeBalls.splice(i, 1);
                i--;
            }
        }

        // add new poke balls if limit is not reached
        while (pokeBalls.length < maxPokeBalls) {
            let radius = randomIntInRange(minRadius, maxRadius);
            let x = -radius + 1;
            let y = randomIntInRange(radius, canvas.height - radius);

            // place on screen if initial spawn, else place on side
            if (isInitialSpawn) {
                x = randomIntInRange(1, canvas.width - radius - 1);
                y = randomIntInRange(1, canvas.height - radius - 1);

            } else {
                x = -radius + 1;
                y = randomIntInRange(radius, canvas.height - radius);
            }

            let direction = randomFloatInRange(-directionOffset, directionOffset);

            // add a new poke ball
            PokeBall.addToArray(pokeBalls, new PokeBall(context, x, y, radius, direction));
        }

        // initial spawn set to false after initial poke ball placement
        if (isInitialSpawn) {
            isInitialSpawn = false;
        }

        // update poke balls
        for (let i = 0; i < pokeBalls.length; i++) {
            pokeBalls[i].update();
        }
    }

    init();

    return {
        init: init,
        animate: animate
    };
}();

window.addEventListener('load', pokeBallCanvas.animate);
window.addEventListener('resize', pokeBallCanvas.init);