import os
import tempfile
import shutil
import zipfile
from flask import Flask, render_template, request, send_file, jsonify, url_for, redirect
from werkzeug.utils import secure_filename
from main import create_mosaic

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['RESULT_FOLDER'] = tempfile.mkdtemp()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    temp_dir = tempfile.mkdtemp()
    try:
        # === Исходное изображение ===
        input_file = request.files['input_image']
        input_path = os.path.join(temp_dir, secure_filename(input_file.filename))
        input_file.save(input_path)

        # === Тайлы ===
        tiles_mode = request.form.get('tiles_mode')
        tiles_folder = os.path.join(temp_dir, 'tiles')
        os.makedirs(tiles_folder, exist_ok=True)

        if tiles_mode == 'zip':
            zip_file = request.files['tiles_zip']
            zip_path = os.path.join(temp_dir, secure_filename(zip_file.filename))
            zip_file.save(zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(tiles_folder)
        elif tiles_mode == 'images':
            tile_files = request.files.getlist('tiles_images')
            for file in tile_files:
                file.save(os.path.join(tiles_folder, secure_filename(file.filename)))
        elif tiles_mode == 'folder':
            tile_files = request.files.getlist('tiles_folder')
            for file in tile_files:
                file.save(os.path.join(tiles_folder, secure_filename(file.filename)))
        else:
            return jsonify({'error': 'Неверный способ выбора тайлов'}), 400

        # === Параметры ===
        grid = int(request.form.get('grid', 30))
        stride = request.form.get('stride')
        stride = int(stride) if stride else None
        rotate = 'rotate' in request.form
        max_usage = int(request.form.get('max_usage', 0))
        color_correction = float(request.form.get('color_correction', 0))
        blend = float(request.form.get('blend', 0))
        metric = request.form.get('metric', 'color')
        grad_weight = float(request.form.get('grad_weight', 0.5))
        seam_smoothing = float(request.form.get('seam_smoothing', 0))

        result_filename = "mosaic_preview.png"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

        # === Генерация мозаики ===
        create_mosaic(
            input_path,
            tiles_folder,
            result_path,
            grid,
            stride=stride,
            allow_rotate=rotate,
            max_usage=max_usage,
            color_correction_strength=color_correction,
            metric=metric,
            grad_weight=grad_weight,
            seam_smoothing=seam_smoothing,
            blend=blend
        )

        # Удаляем временные файлы тайлов
        shutil.rmtree(temp_dir, ignore_errors=True)

        # Переходим на страницу предпросмотра
        return redirect(url_for('preview_result', filename=result_filename))

    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return f"Ошибка: {e}", 500


@app.route('/preview/<filename>')
def preview_result(filename):
    return render_template('preview.html', image_url=url_for('serve_result', filename=filename))


@app.route('/result/<filename>')
def serve_result(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename))


@app.route('/download/<filename>')
def download_result(filename):
    return send_file(
        os.path.join(app.config['RESULT_FOLDER'], filename),
        as_attachment=True,
        download_name='mosaic.png'
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
