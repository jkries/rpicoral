# -*- coding: utf-8 -*-
import os
from flask import Flask, render_template
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

import argparse
import re
from edgetpu.classification.engine import ClassificationEngine
from PIL import Image


app = Flask(__name__)
app.config['SECRET_KEY'] = 'AI rules'
#app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd()
app.config['UPLOADED_PHOTOS_DEST'] = 'myPhotos'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

#Coral config info
myModel = "../ai/mobilenet_v2_1.0_224_inat_insect_quant_edgetpu.tflite"
myLabel = "../ai/inat_insect_labels.txt"


# Function to read labels from text files.
def ReadLabelFile(file_path):
  """Reads labels from text file and store it in a dict.

  Each line in the file contains id and description separted by colon or space.
  Example: '0:cat' or '0 cat'.

  Args:
    file_path: String, path to the label file.

  Returns:
    Dict of (int, string) which maps label id to description.
  """
  with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
  ret = {}
  for line in lines:
    pair = re.split(r'[:\s]+', line.strip(), maxsplit=1)
    ret[int(pair[0])] = pair[1].strip()
  return ret


class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, u'Image only!'), FileRequired(u'File was empty!')])
    submit = SubmitField(u'Upload')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = photos.url(filename)
        imageSpot = os.path.basename(file_url)
        # Prepare labels.
        labels = ReadLabelFile(myLabel)
        # Initialize engine.
        engine = ClassificationEngine(myModel)
        # Run inference.
        img = Image.open(file_url)
        for result in engine.ClassifyWithImage(img, top_k=3):
            print('---------------------------')
            print(labels[result[0]])
            print('Score : ', result[1])
    else:
        file_url = None
        coralReply = ''
    return render_template('index.html', form=form, file_url=file_url, coralReply=coralReply)


if __name__ == '__main__':
    app.run()
