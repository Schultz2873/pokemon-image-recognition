from flask import Flask, render_template, request, redirect, json, jsonify
from werkzeug.utils import secure_filename
from keras_train import PredictionData
import os

app = Flask(__name__)


def is_allowed_file_extension(filename):
    """
    Determines if the passed in file's extension matches allowed extensions.
    :param filename: A file name or path
    :return: Returns true if the extension is in the allowed list of extensions, else false.
    """
    period = '.'
    return period in filename and filename.rsplit(period, 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    print('request.files:', request.files)
    key = 'file'

    # dict keys
    status_key = 'status'
    num_files_key = 'num_files'
    data_key = 'data'
    class_name_key = 'class_name'
    probability_key = 'probability'
    messages_key = 'messages'

    # dict values
    status = 'success'
    data = None
    num_files = 0
    messages = []

    # if request has file
    if key in request.files:
        file = request.files.get(key)
        filename = file.filename

        num_files = len(request.files.getlist(key))
        print('files:', num_files)

        # if empty file name
        if filename == '':
            messages.append('Empty file name.')

        # else if invalid extension
        elif not is_allowed_file_extension(filename):
            messages.append('Invalid extension.')

        # else if invalid number of files
        elif num_files != 1:
            messages.append('Invalid number of files (' + str(num_files) + ').')

        # process uploaded file
        else:
            # dict for json data
            data = {class_name_key: None, probability_key: None}

            # change file name to secure name
            filename = secure_filename(file.filename)

            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            model_path = '../pokemon-repo/keras_model/2018-12-07 20-18-42.779946_100x100' \
                         + '_20-e_4-c2dl_32-f_0.4-d.h5'

            labels_directory = '../pokemon-repo/datasets/pokemon/validate'
            img_size = app.config['IMAGE_CONVERSION_SIZE']

            # get class prediction data from trained model
            prediction_data = PredictionData(model_path, image_path, labels_directory, img_size[0], img_size[1])

            # prediction results assigned to dict for json data
            data[class_name_key] = prediction_data.class_name.capitalize()
            data[probability_key] = str(prediction_data.probability)

            os.remove(image_path)

            print('prediction_data:', prediction_data.__str__())

    # create dictionary for json response
    response_dict = {status_key: status, num_files_key: num_files, data_key: data, messages_key: messages}

    return jsonify(response_dict)


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'svg'}
    app.config['UPLOAD_FOLDER'] = '../pokemon-repo/temp_img'
    app.config['IMAGE_CONVERSION_SIZE'] = 100, 100
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
    app.run(debug=True)
