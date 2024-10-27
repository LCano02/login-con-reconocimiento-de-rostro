from flask import Flask, render_template, Response, redirect, url_for, request
import cv2
import os
import imutils
import numpy as np

app = Flask(__name__)

dataPath = 'C:/xampp/htdocs/facial/Data'  # Ruta para almacenar las imágenes capturadas
if not os.path.exists(dataPath):
    os.makedirs(dataPath)

personName = ''
personPath = ''

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/registro', methods=['POST', 'GET'])
def registro():
    global personName, personPath
    if request.method == 'POST':
        personName = request.form.get('nombre')
        personPath = os.path.join(dataPath, personName)
        
        if not os.path.exists(personPath):
            os.makedirs(personPath)
        return redirect(url_for('video_feed'))
    return render_template('registro.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)

        # Confirmar que personPath es válido antes de guardar
        if personPath and os.path.exists(personPath):
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                rostro = auxFrame[y:y + h, x:x + w]
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(personPath, f'rostro_{count}.jpg'), rostro)
                count += 1

            if count >= 300:
                break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/train', methods=['POST'])  # Asegúrate de que solo acepte POST
def train():
    peopleList = os.listdir(dataPath)
    labels = []
    facesData = []
    label = 0

    for nameDir in peopleList:
        personPath = os.path.join(dataPath, nameDir)

        for fileName in os.listdir(personPath):
            labels.append(label)
            facesData.append(cv2.imread(os.path.join(personPath, fileName), 0))
        
        label += 1

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(facesData, np.array(labels))
    face_recognizer.write('modeloLBPHFace.xml')

    return redirect(url_for('index'))

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/reconocer', methods=['POST'])
def reconocer():
    nombre_ingresado = request.form['nombre']  # Obtener el nombre ingresado
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('modeloLBPHFace.xml')

    cap = cv2.VideoCapture(0)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            rostro = auxFrame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            # Obtener el índice de la etiqueta
            label_index = result[0]
            confidence = result[1]

            if confidence < 70:  # Confianza aceptable
                # Obtener el nombre correspondiente a la etiqueta
                nombre_reconocido = os.listdir(dataPath)[label_index]
                
                if nombre_ingresado.lower() == nombre_reconocido.lower():  # Comparar sin distinción de mayúsculas
                    cap.release()
                    return redirect(url_for('bienvenida'))
                
            cap.release()
            return redirect(url_for('login'))

    cap.release()
    return redirect(url_for('login'))

@app.route('/bienvenida')
def bienvenida():
    return render_template('bienvenida.html')

if __name__ == '__main__':
    app.run(debug=True)
