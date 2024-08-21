import cv2
import mediapipe as mp
import sqlite3
import face_recognition
import numpy as np

def save_face_to_db(name, face_encoding):
    # Conectar ao banco de dados
    conn = sqlite3.connect('face_database.db')
    cursor = conn.cursor()

    # Armazenar a codificação no banco de dados
    cursor.execute('''
    INSERT INTO known_faces (name, encoding) VALUES (?, ?)
    ''', (name, face_encoding.tobytes()))

    # Salvar e fechar a conexão
    conn.commit()
    conn.close()

def load_known_faces():
    # Conectar ao banco de dados
    conn = sqlite3.connect('face_database.db')
    cursor = conn.cursor()

    # Recuperar todas as codificações de rostos conhecidos
    cursor.execute('SELECT name, encoding FROM known_faces')
    rows = cursor.fetchall()

    known_face_encodings = []
    known_face_names = []

    for row in rows:
        name = row[0]
        encoding = np.frombuffer(row[1], dtype=np.float64)
        known_face_encodings.append(encoding)
        known_face_names.append(name)

    conn.close()

    return known_face_encodings, known_face_names

def is_face_registered(face_encoding, known_face_encodings):
    # Comparar a nova face com as faces conhecidas
    distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    return np.any(distances < 0.6)  # 0.6 é um limiar comum para similaridade facial

def add_existing_face_to_db(image_path, name):
    # Carregar a imagem de rosto existente
    image = face_recognition.load_image_file(image_path)

    # Obter a codificação facial
    face_encodings = face_recognition.face_encodings(image)

    if not face_encodings:
        print(f"Nenhuma codificação facial encontrada na imagem {image_path}.")
        return

    face_encoding = face_encodings[0]

    # Conectar ao banco de dados (ou criar se não existir)
    conn = sqlite3.connect('face_database.db')
    cursor = conn.cursor()

    # Criar a tabela para armazenar os rostos conhecidos se não existir
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS known_faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        encoding BLOB NOT NULL
    )
    ''')
    conn.commit()

    # Verificar se a face já está registrada
    known_face_encodings, _ = load_known_faces()
    if is_face_registered(face_encoding, known_face_encodings):
        print("A pessoa já está cadastrada.")
    else:
        save_face_to_db(name, face_encoding)
        print("Pessoa cadastrada com sucesso!")

    conn.close()

# Adicionar um rosto existente ao banco de dados
add_existing_face_to_db('rosto.png', 'Nome da Pessoa')

# Inicializar a webcam
webcam = cv2.VideoCapture(0)

# Inicializar o reconhecedor de rostos do MediaPipe
face_detection = mp.solutions.face_detection.FaceDetection()

while True:
    # Ler o frame da webcam
    ret, frame = webcam.read()

    if not ret:
        break

    # Processar o frame para detecção de rostos
    result = face_detection.process(frame)

    # Verificar se foram detectados rostos
    if result.detections:
        for detection in result.detections:
            # Desenhar o retângulo ao redor do rosto
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, bbox, (0, 255, 0), 2)

            # Exibir uma mensagem para o usuário
            cv2.putText(frame, "Rosto Detectado! Pressione 's' para cadastrar", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Mostrar o frame com as detecções
            cv2.imshow("Rostos na Webcam", frame)

            # Aguardar pela entrada do usuário
            key = cv2.waitKey(1)

            # Se o usuário pressionar 's', a pessoa é cadastrada
            if key == ord('s'):
                # Solicitar o nome da pessoa
                name = input("Digite o nome da pessoa: ")
                print(f"Nome recebido: {name}")

                # Extrair a imagem do rosto usando o bounding box
                top, right, bottom, left = bbox
                face_image = frame[top:bottom, left:right]

                # Converter a imagem para RGB (se necessário)
                rgb_face_image = face_image[:, :, ::-1]

                # Obter a codificação facial
                face_encodings = face_recognition.face_encodings(rgb_face_image)

                if not face_encodings:
                    print("Nenhuma codificação facial encontrada.")
                    continue

                face_encoding = face_encodings[0]

                # Verificar se a face já está registrada
                known_face_encodings, known_face_names = load_known_faces()
                if is_face_registered(face_encoding, known_face_encodings):
                    print("A pessoa já está cadastrada.")
                else:
                    # Salvar a face no banco de dados
                    save_face_to_db(name, face_encoding)
                    known_face_encodings.append(face_encoding)  # Atualizar a lista de faces conhecidas
                    known_face_names.append(name)  # Atualizar a lista de nomes
                    print("Pessoa cadastrada!")

            # Se o usuário pressionar 'n', a pessoa não é cadastrada
            elif key == ord('n'):
                print("Pessoa não cadastrada!")

    # Quando 'ESC' é pressionado, sair do loop
    if cv2.waitKey(1) == 27:
        break

# Liberar recursos
webcam.release()
cv2.destroyAllWindows()