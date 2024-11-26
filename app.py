from flask import Flask, render_template, request, redirect, url_for
import os
import sys
import matplotlib.pyplot as plt
from segmentation import liver_demo  # Assuming liver_demo is in segmentation.py
from io import StringIO

app = Flask(__name__)

# Static folder setup
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'

# Ensure the results folder exists
if not os.path.exists(app.config['RESULTS_FOLDER']):
    os.makedirs(app.config['RESULTS_FOLDER'])

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for uploading files and running the liver segmentation model
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'nii_file' not in request.files or 'ground_file' not in request.files:
        return "No files in request."

    nii_file = request.files['nii_file']
    ground_file = request.files['ground_file']

    if nii_file and ground_file:
        # Save the files in the upload folder
        nii_filename = nii_file.filename
        ground_filename = ground_file.filename

        nii_filepath = os.path.join(app.config['UPLOAD_FOLDER'], nii_filename)
        ground_filepath = os.path.join(app.config['UPLOAD_FOLDER'], ground_filename)

        nii_file.save(nii_filepath)
        ground_file.save(ground_filepath)

        # Redirect stdout to capture printed results from liver_demo function
        captured_output = StringIO()
        sys.stdout = captured_output  # Redirect stdout to capture print statements

        outputs = []  # List to store all outputs

        try:
            # Run the liver segmentation model
            liver_demo(nii_filepath, ground_filepath)

            # Capture any text output
            text_output = captured_output.getvalue()
            if text_output:
                outputs.append({'type': 'text', 'content': text_output})
            
            # Capture image outputs
            for i in plt.get_fignums():
                fig = plt.figure(i)
                img_filename = f"segmentation_image_{i}.png"
                img_filepath = os.path.join(app.config['RESULTS_FOLDER'], img_filename)
                fig.savefig(img_filepath)
                outputs.append({'type': 'image', 'content': img_filename})
            plt.close('all')

        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__

        # Pass outputs to results page
        return render_template('results.html', outputs=outputs)

    return "Invalid file type."

# Route for displaying results
@app.route('/results')
def results():
    return "Results page should be accessed via file upload."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5006)
