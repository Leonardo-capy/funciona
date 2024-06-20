import face_recognition # type: ignore
import cv2
import numpy as np
import cmake

# Carregar a imagem conhecida e codificar o rosto
known_image = face_recognition.load_image_file("known_face.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Inicializar a lista de rostos conhecidos e seus respectivos nomes
known_faces = [known_encoding]
known_names = ["Nome Conhecido"]

# Inicializar a webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capturar o frame da webcam
    ret, frame = video_capture.read()

    # Encontrar todos os rostos no frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Para cada rosto encontrado
    for face_encoding in face_encodings:
        # Verificar se o rosto é conhecido
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Desconhecido"

        # Se encontrou um rosto conhecido
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        # Desenhar um retângulo ao redor do rosto e mostrar o nome
        top, right, bottom, left = face_recognition.face_locations(frame)[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Se o rosto for desconhecido, abrir um prompt para colocar o nome
        if name == "Desconhecido":
            nome_desconhecido = input("Por favor, insira o nome para este rosto: ")
            known_faces.append(face_encoding)
            known_names.append(nome_desconhecido)

    # Mostrar o frame resultante
    cv2.imshow('Video', frame)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpar e fechar a janela
video_capture.release()
cv2.destroyAllWindows()
