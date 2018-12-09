function formatProbability(probability, decimalPlaces) {
    probability *= 100;
    return probability.toFixed(decimalPlaces) + '%';
}

document.querySelector('#img-input').addEventListener('change', function () {
    let reader = new FileReader();

    reader.addEventListener('load', function (event) {
        document.querySelector('#img-preview').src = event.target.result;
    });

    reader.readAsDataURL(this.files[0]);
});

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
        console.log(formatProbability(responseData.probability, 2));
    });
});