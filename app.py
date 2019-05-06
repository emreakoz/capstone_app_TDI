from flask import Flask, render_template, request
from werkzeug import secure_filename
from emotion_predictor import emotion_predictor

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        im_file = request.files["myfile"]
        im_file.save(secure_filename(im_file.filename))
        #path = '/Users/emre/apps/capstone_app/old_images/' + im_file.filename
        printout1, printout2 = emotion_predictor()
        
        return render_template('project.html', result1=printout1, 
                               result2 = printout2, result3 = im_file.filename)

if __name__ == '__main__':
    app.run(debug=True)