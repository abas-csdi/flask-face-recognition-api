from flask import Flask, request, jsonify
import face_recognition
import os
import pickle

app = Flask(__name__)
training_data_file = "training_data.pickle"


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def training_data_load():
    # Memuat data pelatihan yang ada jika ada
    if os.path.exists(training_data_file):
        with open(training_data_file, 'rb') as f:
            known_face_encodings, ids, filenames = pickle.load(f)
    else:
        known_face_encodings, ids, filenames = [], [], []

    return known_face_encodings, ids, filenames


def image_processing(image):
    # Memproses gambar yang diunggah
    image = face_recognition.load_image_file(image)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_encodings


def save_training_data(id, filename, face_encodings, known_face_encodings, ids, filenames):
    # Menyimpan encoding dan label pada data pelatihan
    for face_encoding in face_encodings:
        known_face_encodings.append(face_encoding)
        ids.append(id)
        filenames.append(filename)

    # Menyimpan data pelatihan ke dalam file
    with open(training_data_file, 'wb') as f:
        pickle.dump((known_face_encodings, ids, filenames), f)


def recognition_processing(face_encodings, known_face_encodings, ids, filenames):
    is_valid = False
    id = ""
    filename = ""

    # Melakukan face recognition pada gambar yang diunggah
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        if True in matches:
            first_match_index = matches.index(True)
            is_valid = True
            id = ids[first_match_index]
            filename = filenames[first_match_index]
            break
    return is_valid, id, filename


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# -> AI Zone


@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({
            'message': 'No image part in the request'
        }), 400
    image = request.files['image']
    if image.filename == '':
        return jsonify({
            'message': 'No image selected',
        }), 400

    known_face_encodings, ids, filenames = training_data_load()
    face_encodings = image_processing(image)
    if not face_encodings:
        return jsonify({
            'message': 'No face found in the uploaded image',
            'is_valid': False,
        }), 400
    if len(face_encodings) > 1:
        return jsonify({
            'message': 'Multiple faces found in the uploaded image',
            'is_valid': False,
        }), 400

    is_valid, id, filename = recognition_processing(face_encodings, known_face_encodings, ids, filenames)
    if is_valid:
        return jsonify({
            'is_valid': is_valid,
            'id': id,
            'filename': filename,
        }), 200
    else:
        return jsonify({
            'is_valid': False,
        }), 200


@app.route('/register', methods=['POST'])
def register():
    id = request.form.get('id')
    filename = request.form.get('filename')
    if id is None or filename is None:
        return jsonify({
            'message': 'Body is\'n complete!'
        }), 400
    if 'image' not in request.files:
        return jsonify({
            'message': 'No image part in the request'
        }), 400
    image = request.files['image']
    if image.filename == '':
        return jsonify({
            'message': 'No image selected'
        }), 400

    known_face_encodings, ids, filenames = training_data_load()
    face_encodings = image_processing(image)
    if not face_encodings:
        return jsonify({'message': 'No face found in the uploaded image'}), 400
    if len(face_encodings) > 1:
        return jsonify({'message': 'Multiple faces found in the uploaded image'}), 400

    is_valid, _, _ = recognition_processing(face_encodings, known_face_encodings, ids, filenames)
    if is_valid is True:
        return jsonify({
            'message': 'Face is already registered in the database!'
        }), 400
    save_training_data(id, filename, face_encodings, known_face_encodings, ids, filenames)

    return jsonify({
        'message': 'Training berhasil.'
    }), 201


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# -> Training Data Management


@app.route('/all_training_data', methods=['GET'])
def all_training_data():
    known_face_encodings, ids, filenames = training_data_load()
    training_data = [{'id': id, 'filename': filename} for id, filename in zip(ids, filenames)]
    return jsonify(training_data), 200


@app.route('/get_training_data/<id>', methods=['GET'])
def get_training_data(id):
    known_face_encodings, ids, filenames = training_data_load()
    result = [filename for filename, training_id in zip(filenames, ids) if training_id == id]
    return jsonify({
        'result': result
    }), 200


@app.route('/delete_training_data/<id>/<filename>', methods=['DELETE'])
def delete_training_data(id, filename):
    known_face_encodings, ids, filenames = training_data_load()
    # Membuat daftar indeks hasil training yang akan dihapus
    indices_to_delete = [index for index, (training_id, training_filename) in enumerate(zip(ids, filenames))
                         if training_id == id and training_filename == filename]
    # Menghapus hasil training dari daftar berdasarkan indeks
    for index in sorted(indices_to_delete, reverse=True):
        del known_face_encodings[index]
        del ids[index]
        del filenames[index]
    # Menyimpan ulang data pelatihan yang diperbarui ke dalam file
    with open(training_data_file, 'wb') as f:
        pickle.dump((known_face_encodings, ids, filenames), f)
    return jsonify({
        'message': 'Training data deleted successfully.'
    }), 200


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

if __name__ == '__main__':
    app.run()
