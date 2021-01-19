from flask import render_template, request, flash, redirect, send_file
from werkzeug.utils import secure_filename
from pathlib import Path
from app import app
from app.utils.imageextraction import *

@app.route('/')
def home():
    uploaded_videos = []
    download_videos = []
    print(Path.cwd())
    current_path = Path.cwd() / "app" / "static" #/ "videos"
    folder_path = current_path.iterdir()
    get_stream("test")


    for file in folder_path:
        if file.is_file():
            uploaded_videos.append(file.name)


    current_path = Path.cwd() / "app" / "static" #/ "generatedvideos"
    folder_path = current_path.iterdir()

    for file in folder_path:
        if file.is_file():
            download_videos.append(file.name)
    
    return render_template("home.html", video_list=uploaded_videos, video_download_list=download_videos)

@app.route('/upload', methods=["POST"])
def upload():
    file = request.files["uploaded_file"]
    file_name = secure_filename(file.filename)
    file_name = Path(file_name)
    print(file_name.suffix)

    if(file_name.suffix != ".mp4" and file_name.suffix != ".mkv"):
        flash("Wrong File Format. Upload Failed")
        return home()

    current_path = Path.cwd()
    current_path = Path(current_path)
    file_path = current_path / "app" / "static" / "videos" / file_name
    file.save(file_path.__str__())
    # image_extraction(file_path.__str__())
    return home()

@app.route('/runmodel/<file_name>')
def run_model(file_name):
    image_path = image_extraction(file_name)
    print(image_path)
    return home()

@app.route('/downloadfile/<file_name>')
def download_file(file_name):
    current_path = Path.cwd() / "app" / "static" / "generatedvideos"
    file_path = current_path / file_name
    return send_file(file_path.__str__(), as_attachment=True)

@app.route('/playvideo/<file_name>')
def play_video(file_name):
    # current_path = Path.cwd() / "app" / "static"
    # file_path = current_path / file_name
    # print(file_path)
    # file_name = "videos" / Path(file_name)
    return render_template("play_video.html", file_name = file_name)

if __name__ == '__main__':
    app.run(debug=True)