from utils.aws_client import rekognition_client
from utils.alerts import send_sms_alert
from utils.face_comparison import compare_faces
import cv2
from datetime import datetime

MAX_OCCUPANCY = 5
entered_faces = {}
alert_triggered = False

def process_video_feed():
    global alert_triggered
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        response = rekognition_client.detect_faces(
            Image={'Bytes': frame_bytes}, Attributes=['ALL']
        )
        for face_detail in response['FaceDetails']:
            bounding_box = face_detail['BoundingBox']
            x = int(bounding_box['Left'] * frame.shape[1])
            y = int(bounding_box['Top'] * frame.shape[0])
            w = int(bounding_box['Width'] * frame.shape[1])
            h = int(bounding_box['Height'] * frame.shape[0])
            cropped_face = frame[y:y + h, x:x + w]
            _, face_buffer = cv2.imencode('.jpg', cropped_face)
            face_bytes = face_buffer.tobytes()
            is_new_face = True
            for person_id, stored_face in entered_faces.items():
                if compare_faces(stored_face, face_bytes):
                    is_new_face = False
                    break
            if is_new_face:
                entered_faces[datetime.now()] = face_bytes
        if len(entered_faces) > MAX_OCCUPANCY and not alert_triggered:
            send_sms_alert("Maximum room occupancy reached!")
            alert_triggered = True
        cv2.imshow('Room Monitoring', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video_feed()
