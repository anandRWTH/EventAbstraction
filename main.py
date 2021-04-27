from flask import Flask, flash, redirect, render_template, request, jsonify
import pandas as pd
import constants
import preprocess

app = Flask(__name__)

app.secret_key = "secret key"
app.config['LOG_FORMAT'] = ['csv']
app.config['STORAGE_PATH'] = "uploads/"


def allowed_log_file(filename):
    """
    checks if given file has correct type for log
    :param filename: name of the file to check
    :return: boolean: file has valid type
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['LOG_FORMAT']


@app.route('/', methods=['GET', 'POST'])
def process_file():
    if request.method == 'POST':
        if not request.files:
            flash('No file selected for uploading.')
            return redirect(request.url)
        csv_file = request.files['file']
        if csv_file.filename == '':
            flash('No file selected for uploading.')
            return redirect(request.url)
        if csv_file and allowed_log_file(csv_file.filename):
            filename = "og.csv"
            csv_file.save(app.config['STORAGE_PATH'] + filename)
            constants.og_data_file = app.config['STORAGE_PATH'] + filename
            csv_file = open(app.config['STORAGE_PATH'] + filename)
            df = pd.read_csv(csv_file)
            columns = list(df.head(0))
            print(columns)
            return render_template('columns.html', columns=columns)
        else:
            flash('The only allowed file type is csv.')
            return redirect(request.url)
    else:
        return render_template('upload.html')


@app.route('/map', methods=['POST'])
def map_columns():
    constants.start_timestamp = request.form.get('start_column')
    constants.id_column = request.form.get('id_column')
    constants.activities_column = request.form.get('activity_column')
    constants.date_column = request.form.get('date_column')
    tree_data = preprocess.build_hierarchical_json()

    """data = ast.literal_eval("{'null': 'og', 'og': ['og_c2_1', 'og_c2_0'], 'og_c2_1': ['og_c2_1_c2_0', 'og_c2_1_c2_1'], 'og_c2_0': ['og_c2_0_c2_1', 'og_c2_0_c2_0'], 'og_c2_0_c2_1': ['og_c2_0_c2_1_c2_1', 'og_c2_0_c2_1_c2_0'], 'og_c2_0_c2_0': ['og_c2_0_c2_0_c2_1', 'og_c2_0_c2_0_c2_0'], 'og_c2_0_c2_0_c2_0': ['og_c2_0_c2_0_c2_0_c2_1', 'og_c2_0_c2_0_c2_0_c2_0']}")

    tree_data = json.dumps(preprocess.get_nodes("og","null",data))
    print(tree_data)"""
    return render_template('tree.html', tree_data=tree_data)


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
