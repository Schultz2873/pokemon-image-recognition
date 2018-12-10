// auto-executing function for encapsulation
(function () {
    let predictionElement = null;
    let submitButton = null;

// loading icon assigned to auto-executing function for encapsulation
    const loadingIcon = (function () {
        let canvas = null;
        let context = null;
        let pokeBall = null;
        let sparks = [];
        let maxSparks = 50;
        let isUseSparks = false;
        let lastSparkSpawn = null;
        let animationFrame = null;

        function init() {
            canvas = document.querySelector('#icon-canvas');
            context = canvas.getContext('2d');
            setElementDisplay(canvas, 'none');
            sizeCanvas();
            sparks = [];
        }

        function createPokeBall() {
            let x = Math.floor(canvas.width / 2);
            let y = Math.floor(canvas.height / 2);
            let radius = Math.floor(canvas.width * .07);
            let rotation = toRadians(25);
            pokeBall = new PokeBall(context, x, y, radius, 0, 0, 0, rotation);
            pokeBall.isProportionalRotation = false;
            pokeBall.isProportionalSpeed = false;
        }

        function deletePokeBall() {
            pokeBall = null;
        }

        function addSpark() {
            let x = Math.floor(canvas.width / 2);
            let radius = pokeBall.radius * .1;
            let direction = randomFloatInRange(0, TWO_PI);
            let speed = canvas.width * .03;
            let color = new RgbaColor(255, 255, 255);
            let durationMs = 250;
            sparks.push(
                new Spark(context, sparks, x, x, radius, direction, speed, 0, 0, color, durationMs));
        }

        function clearSparks() {
            sparks = [];
        }

        function sizeCanvas() {
            let size = Math.floor(Math.min(innerWidth, innerHeight) * .36);
            setCanvasSize(canvas, size, size);
        }

        function isSpawnReady() {
            return sparks.length < maxSparks;
        }

        function update() {
            if (isUseSparks) {
                if (isSpawnReady()) {
                    addSpark();
                    lastSparkSpawn = performance.now();
                }

                for (let i = 0; i < sparks.length; i++) {
                    let spark = sparks[i];
                    if (!spark.update()) {
                        i--;
                    }
                }
            }

            pokeBall.update();
        }

        function animate() {
            animationFrame = requestAnimationFrame(animate);
            context.clearRect(0, 0, canvas.width, canvas.height);
            update();
        }

        function start() {
            createPokeBall();
            setElementDisplay(canvas, 'initial');
            animate();
        }

        function stop() {
            cancelAnimationFrame(animationFrame);
            setElementDisplay(canvas, 'none');
            deletePokeBall();
            clearSparks();
        }

        return {
            init: init,
            sizeCanvas: sizeCanvas,
            start: start,
            stop: stop
        }
    }());

    function setElementDisplay(element, display) {
        element.style.display = display;
    }

    function deleteChildren(element) {
        while (element.hasChildNodes()) {
            element.removeChild(element.childNodes[0]);
        }
    }

    function formatProbability(probability, decimalPlaces) {
        probability *= 100;
        return probability.toFixed(decimalPlaces);
    }

    function createPredictionElementContent(class_name, probability, decimalPlaces) {
        let classNameNode = document.createTextNode(class_name + ' ');
        let probabilityNode = document.createTextNode(formatProbability(probability, decimalPlaces));

        let color = '#33ce7a';
        if (probability < .5) {
            color = '#c42348';
        } else if (probability < .8) {
            color = '#ffbd00';
        }

        let probabilitySpan = document.createElement('span');
        probabilitySpan.style.color = color;

        predictionElement.appendChild(classNameNode);
        predictionElement.appendChild(probabilitySpan);
        probabilitySpan.appendChild(probabilityNode);
        predictionElement.appendChild(document.createTextNode('%'));
    }

    function clearElement(element) {
        setElementDisplay(element, 'none');
        deleteChildren(element);
    }

    function handleFormChange() {
        clearElement(predictionElement);
        let reader = new FileReader();

        reader.addEventListener('load', function (event) {
            document.querySelector('#img-preview').src = event.target.result;
        });

        reader.readAsDataURL(this.files[0]);
    }

    function handleImageUpload() {
        $('#upload-form').submit(function (event) {
            clearElement(predictionElement);
            loadingIcon.start();
            event.preventDefault();

            let post_url = $(this).attr('action');
            let request_method = $(this).attr('method');
            let form_data = new FormData(this);

            $.ajax({
                url: post_url,
                type: request_method,
                dataType: 'json',
                data: form_data,
                contentType: false,
                cache: false,
                processData: false
            }).done(function (response) {
                // $("#server-results").html(response);
                let responseData = {
                    status: response.status,
                    class_name: response.class_name,
                    probability: parseFloat(response.probability)
                };
                console.log(responseData);

                createPredictionElementContent(responseData.class_name, responseData.probability, 2);
                setElementDisplay(predictionElement, 'initial');
                loadingIcon.stop();
            });
        });
    }


    window.addEventListener('load', function () {
        predictionElement = document.querySelector('#prediction');
        submitButton = document.querySelector('#upload-submit');
        setElementDisplay(predictionElement, 'none');

        // initialize load icon
        loadingIcon.init();

        // form onchange listener
        document.querySelector('#img-input').addEventListener('change', handleFormChange);

        // image upload
        handleImageUpload();
    });

    window.addEventListener('resize', function () {
        loadingIcon.sizeCanvas();
    });
}());

