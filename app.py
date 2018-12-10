from flask import Flask, render_template, request, redirect, json
from werkzeug.utils import secure_filename
from keras_train import PredictionData
import os

# from flask_bootstrap import Bootstrap

app = Flask(__name__)


# Bootstrap(app)

def is_allowed_file(filename, allowed_extensions):
    period = '.'
    return period in filename and filename.rsplit(period, 1)[1].lower() in allowed_extensions


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload():
    print('request.files:', request.files)
    key = 'file'

    if key in request.files:
        file = request.files.get(key)
        filename = file.filename

        # if valid file
        if filename != '' and is_allowed_file(filename, app.config['ALLOWED_EXTENSIONS']):
            filename = secure_filename(file.filename)

            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            model_path = '../pokemon-repo/keras_model/2018-12-07 20-18-42.779946_100x100' \
                         + '_20-e_4-c2dl_32-f_0.4-d.h5'

            labels_directory = '../pokemon-repo/datasets/pokemon/validate'
            img_size = app.config['IMAGE_CONVERSION_SIZE']
            prediction_data = PredictionData(model_path, image_path, labels_directory, img_size[0], img_size[1])

            os.remove(image_path)

            print('prediction_data:', prediction_data.__str__())
            return json.dumps({'status': 'OK', 'class_name': prediction_data.class_name.capitalize(),
                               'probability': str(prediction_data.probability)})

    # return redirect('/')


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'svg'}
    app.config['UPLOAD_FOLDER'] = '../pokemon-repo/temp_img'
    app.config['IMAGE_CONVERSION_SIZE'] = 100, 100
    app.run(debug=True)
