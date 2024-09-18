from flask import Flask, render_template
import subprocess
import threading

app = Flask(__name__)

# Run the bicep curl detection in a separate thread
def run_bicep_detection():
    subprocess.run(["python", "biceps.py"])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start_detection')
def start_detection():
    threading.Thread(target=run_bicep_detection).start()
    return "Bicep Curl Detection Started!"

if __name__ == '__main__':
    app.run(debug=True)
