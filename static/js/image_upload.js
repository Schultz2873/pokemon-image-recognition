// auto-executing function for encapsulation
(function () {
    let predictionElement = null;
    let imageInput = null;
    let uploadForm = null;
    let imagePreviewElement = null;
    let formClearButton = null;
    let submitButton = null;

    let allowedExtensions = [];

    let colors = {
        prediction: {
            success: '#33ce7a',
            probable: '#ffbd00',
            unlikely: '#c42348'
        }
    };

// loading icon assigned to auto-executing function for encapsulation
    /**
     * A poke ball loading icon using HTML Canvas.
     * @type {{init, stop, start, sizeCanvas}}
     */
    const loadingIcon = (function () {
        let canvas = null;
        let context = null;
        let pokeBall = null;
        let sparks = [];
        let maxSparks = 50;
        let isUseSparks = false;
        let lastSparkSpawn = null;
        let animationFrame = null;
        let isRunning = false;

        /**
         * Initializes variables, sets canvas size.
         */
        function init() {
            canvas = document.querySelector('#icon-canvas');
            context = canvas.getContext('2d');
            setElementDisplay(canvas, 'none');
            sizeCanvas();
            sparks = [];
        }

        /**
         * Creates the poke ball loading icon.
         */
        function createPokeBall() {
            let x = Math.floor(canvas.width / 2);
            let y = Math.floor(canvas.height / 2);
            let radius = Math.floor(canvas.width * .07);
            let rotation = toRadians(25);
            pokeBall = new PokeBall(context, x, y, radius, 0, 0, 0, rotation);
            pokeBall.isProportionalRotation = false;
            pokeBall.isProportionalSpeed = false;
        }

        /**
         * Deletes the poke ball loading icon.
         */
        function deletePokeBall() {
            pokeBall = null;
        }

        /**
         * Adds a single spark object.
         */
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

        /**
         * Removes all sparks.
         */
        function clearSparks() {
            sparks = [];
        }

        /**
         * Sets the canvas to a predetermined size suitable for the icon.
         */
        function sizeCanvas() {
            let size = Math.floor(Math.min(innerWidth, innerHeight) * .36);
            setCanvasSize(canvas, size, size);
        }

        /**
         * Determines if a spark may be spawned.
         * @returns {boolean} Returns true if number of sparks is less than the max allowed sparks, else false.
         */
        function isSparkSpawnReady() {
            return sparks.length < maxSparks;
        }

        /**
         * Updates sparks and the poke ball.
         */
        function update() {
            if (isUseSparks) {
                if (isSparkSpawnReady()) {
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

        /**
         * Animation loop.
         */
        function animate() {
            animationFrame = requestAnimationFrame(animate);
            context.clearRect(0, 0, canvas.width, canvas.height);
            update();
        }

        /**
         * Starts the poke ball icon animation.
         */
        function start() {
            if (!isRunning) {
                isRunning = true;
                createPokeBall();
                setElementDisplay(canvas, 'initial');
                animate();
            }
        }

        /**
         * Stops the poke ball animation.
         */
        function stop() {
            cancelAnimationFrame(animationFrame);
            isRunning = false;
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

    /**
     * Sets the passed in element's display value to the display parameter.
     * @param element An html element.
     * @param display Display type.
     */
    function setElementDisplay(element, display) {
        element.style.display = display;
    }

    /**
     * Removes all child nodes of the passed in html element.
     * @param element An html element.
     */
    function deleteChildren(element) {
        while (element.hasChildNodes()) {
            element.removeChild(element.childNodes[0]);
        }
    }

    /**
     * Formats a classification probability value.
     * @param probability A probability numerical value.
     * @param decimalPlaces The number of decimal places to format the probability value.
     * @returns {string} Returns the formatted probability value as a string.
     */
    function formatProbability(probability, decimalPlaces) {
        probability *= 100;
        return probability.toFixed(decimalPlaces);
    }

    /**
     * Sets the content of the panel that displays a prediction result.
     * @param className Predicted class.
     * @param probability Prediction probability.
     * @param decimalPlaces Number of decimal places to round probability.
     */
    function setPredictionElementContent(className, probability, decimalPlaces) {
        let classNameNode = document.createTextNode(className + ' ');
        let probabilityNode = document.createTextNode(formatProbability(probability, decimalPlaces));

        let color = colors.prediction.success;
        if (probability < .5) {
            color = colors.prediction.unlikely;
        } else if (probability < .8) {
            color = colors.prediction.probable;
        }

        let probabilitySpan = document.createElement('span');
        probabilitySpan.style.color = color;

        predictionElement.appendChild(classNameNode);
        predictionElement.appendChild(probabilitySpan);
        probabilitySpan.appendChild(probabilityNode);
        predictionElement.appendChild(document.createTextNode('%'));
    }

    /**
     * Sets element to display: none and removes all child nodes.
     * @param element An html element.
     */
    function clearElement(element) {
        setElementDisplay(element, 'none');
        deleteChildren(element);
    }

    /**
     * Handles when the image upload form changes. Hides/clears any current prediction results and displays the form's
     * image file in the preview window.
     */
    function handleFormChange() {
        clearElement(predictionElement);
        let reader = new FileReader();

        reader.addEventListener('load', function (event) {
            imagePreviewElement.src = event.target.result;
        });

        reader.readAsDataURL(this.files[0]);
    }

    /**
     * Sets allowed file extension values received server-side. Values assigned from hidden inputs.
     */
    function assignAllowedExtensions() {
        let elements = document.querySelector('#allowed-extensions').children;

        // if elements present
        if (elements.length > 0) {
            for (let i = 0; i < elements.length; i++) {
                allowedExtensions.push(elements[i].value.toLowerCase());
            }

        } else {
            // assign default values if no values found
            allowedExtensions = ['jpg', 'jpeg', 'png', 'svg', 'gif'];
        }
    }

    /**
     * Checks if the passed in file extension is usable for image upload.
     * @param filePath A file path.
     * @returns {boolean} Returns true if the file extension is found in the list of allowed extensions, else false.
     */
    function isAllowedExtension(filePath) {
        return allowedExtensions.includes(fileExtension(filePath.toLowerCase()));
    }

    /**
     * Handles clearing the upload form. Clears the form, removes the image preview source, and hides/clears any
     * current prediction results.
     */
    function handleFormClear() {
        uploadForm.reset();
        imagePreviewElement.src = '';
        clearElement(predictionElement);
    }

    /**
     * Handles image classification upload. Hides/clears any current prediction results, starts the loading icon, gets
     * prediction results via ajax, restores prediction element with ajax results, then stops loading icon.
     * @param event Event.
     */
    function handleImageUpload(event) {
        // prevent standard form submit
        event.preventDefault();

        // clear existing prediction display
        clearElement(predictionElement);

        // if file selected and extension allowed
        if (imageInput.value && isAllowedExtension(imageInput.value)) {
            loadingIcon.start();
            console.log('file input:', imageInput.files);

            let url = $(this).attr('action');
            let requestMethod = $(this).attr('method');
            let formData = new FormData();
            let file = imageInput.files[0];
            formData.append('file', file, file.name);

            console.log('formData:', formData);

            // ajax image upload
            $.ajax({
                url: url,
                type: requestMethod,
                dataType: 'json',
                data: formData,
                contentType: false,
                cache: false,
                processData: false
            }).fail(function (response) {
                console.log('ajax fail', response);
            }).done(function (response) {
                console.log('response:', response);

                // if successful, process prediction result
                if (response.status === 'success' && response.data) {
                    let data = response.data;

                    setPredictionElementContent(data.class_name, parseFloat(data.probability), 2);
                    setElementDisplay(predictionElement, 'initial');
                }

            }).always(function () {
                loadingIcon.stop();
            });
        }
    }

    window.addEventListener('load', function () {
        predictionElement = document.querySelector('#prediction');
        imageInput = document.querySelector('#img-input');
        uploadForm = document.querySelector('#upload-form');
        imagePreviewElement = document.querySelector('#img-preview');
        formClearButton = document.querySelector('#form-clear');
        submitButton = document.querySelector('#upload-submit');

        setElementDisplay(predictionElement, 'none');
        assignAllowedExtensions();

        // form clear listener
        formClearButton.addEventListener('click', handleFormClear);

        // form onchange listener
        imageInput.addEventListener('change', handleFormChange);

        // image upload
        uploadForm.addEventListener('submit', handleImageUpload);

        // initialize load icon
        loadingIcon.init();
    });

    window.addEventListener('resize', function () {
        loadingIcon.sizeCanvas();
    });
}());