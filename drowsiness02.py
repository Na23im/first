import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils



def eye_aspect_ratio(eye):
    # Compute the EAR of an eye using the vertical and horizontal distances between landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize the dlib facial landmark detector and the video stream
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)

# Define the threshold EAR value for drowsiness
EYE_AR_THRESH = 0.2

# Define the number of consecutive frames for which the EAR must be below the threshold to trigger an alert
EYE_AR_CONSEC_FRAMES = 20

# Initialize counters for consecutive frames below the EAR threshold and total number of alerts
COUNTER = 0
ALERT_COUNT = 0

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale and detect faces using the dlib detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    # Loop over each detected face
    for face in faces:
        # Determine the facial landmarks for the face region using the dlib predictor
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye landmarks and compute the EAR for each eye
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Compute the average EAR for both eyes
        ear = (left_ear + right_ear) / 2.0

        # Draw the eyes and the EAR on the frame for visualization
        cv2.drawContours(frame, [left_eye], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye], -1, (0, 255, 0), 1)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # If the EAR is below the threshold, increment the counter for consecutive frames below the threshold
        if ear < EYE_AR_THRESH:
            cv2.putText(frame, "Somnolence", (frame.shape[1] - 300, frame.shape[0] - 20),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            COUNTER += 1
        else:
            COUNTER = 0

        # If the number of consecutive frames below the threshold exceeds the threshold, trigger an alert and reset the counter
        if COUNTER >= EYE_AR_CONSEC_FRAMES:
            ALERT_COUNT += 1
            #sound_alarm()
            COUNTER = 0
        #if ALERT_COUNT == 1:


            #insert GSM sending SMS script

            #ALERT_COUNT=0

    

    # Display the frame
    cv2.imshow("Drowsiness Detection", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()