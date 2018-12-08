// $(function () {
//     $('#upload-form').submit(function (event) {
//         event.preventDefault();
//         let formData = new FormData(this);
//         $.ajax({
//             url: '/upload',
//             type: 'POST',
//             data: $(this).serialize(),
//             success: function (response) {
//                 console.log(response);
//             },
//             error: function (error) {
//                 console.log(error);
//             }
//         });
//     });
// });

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
        console.log(response)
    });
});