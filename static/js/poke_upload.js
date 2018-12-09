let predictionElement = null;

function elementDisplay(element, display) {
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


window.addEventListener('load', function () {
    predictionElement = document.querySelector('#prediction');
    elementDisplay(predictionElement, 'none');

    // form onchange listener
    document.querySelector('#img-input').addEventListener('change', function () {

        elementDisplay(predictionElement, 'none');
        deleteChildren(predictionElement);

        let reader = new FileReader();

        reader.addEventListener('load', function (event) {
            document.querySelector('#img-preview').src = event.target.result;
        });

        reader.readAsDataURL(this.files[0]);
    });

    // image upload
    $("#upload-form").submit(function (event) {
        event.preventDefault(); //prevent default action
        let post_url = $(this).attr("action"); //get form action url
        let request_method = $(this).attr("method"); //get form GET/POST method
        let form_data = new FormData(this); //Creates new FormData object
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
            console.log();

            createPredictionElementContent(responseData.class_name, responseData.probability, 2);

            predictionElement.style.display = 'initial';
        });
    });
});