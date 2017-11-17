from face_swap import *
from cartoon_filter import *
import cv2

def avg(frame1, frame2):
    if frame1.shape != frame2.shape:
        return None
    add = np.add(frame1.astype(np.float), frame2.astype(np.float)) 
    return np.dot(add, 0.5).astype(np.uint8)

def video_swap():
    video = cv2.VideoCapture('../videos/thang.mp4')
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('../videos/output_swap.avi', fourcc, 20.0, (width, height))

    andrew = cv2.imread('../videos/andrew.jpeg')
    ho = cv2.imread('../videos/ho.jpg')
    kim = cv2.imread('../videos/kim.jpg')
    hero = cv2.imread('../videos/hero.jpg')

    detected = True
    swap = 0
   
    num_frame = 0
    count = 0
    while True:
        ret, frame = video.read()
        if ret == True:
            num_frame += 1 
            print (num_frame)

            face = get_landmarks(frame)
            if face is None:
                detected = False
            else:
                if not detected:
                    swap += 1
                    count = 0
                detected = True

            if not detected or swap == 0 or swap == 5:
                writer.write(frame)
                continue
            
            if swap == 1:
                output_frame = face_swap(andrew, frame)
            elif swap == 2:
                output_frame = face_swap(ho, frame)
            elif swap == 3:
                output_frame = face_swap(kim, frame)
            elif swap == 4:
                output_frame = face_swap(hero, frame)
            else:
                continue

            if output_frame is not None:
                writer.write(output_frame)
            else:
                writer.write(frame)
        else:
            break

    video.release()
    writer.release()
    cv2.destroyAllWindows()
        
    return 0


def video_append():
    video1 = cv2.VideoCapture('../videos/hieu2.mp4')
    video2 = cv2.VideoCapture('../videos/swap.avi')
    video3 = cv2.VideoCapture('../videos/emotion.avi')
    width = int(video2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('../videos/output.avi', fourcc, 20.0, (width, height))


    num_frame = 0
    for i in range(241):
        ret, frame = video2.read()
        if ret == True:
            num_frame += 1
            print(num_frame)
            shape = frame.shape
            writer.write(frame)
        else:
            break

    for i in range(481):
        ret, frame = video1.read()
    while True:
        ret, frame = video1.read()
        if ret == True:
            res = cv2.resize(frame, (shape[1], shape[0]), interpolation = cv2.INTER_CUBIC)
            print(res.shape)
            writer.write(res)
        else:
            break

    while True:
        ret, frame = video3.read()
        if ret == True:
            writer.write(frame)
        else:
            break

    video1.release()
    video2.release()
    video3.release()
    writer.release()


def video_emotion():
    emotions = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sadness", "surprise"] #Emotion list
    emoticons = _load_emoticons(emotions)

    video = cv2.VideoCapture('../videos/emotion.mp4')
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('../videos/output_emotion.avi', fourcc, 20.0, (width, height))

    num_frame = 0
    while True:
        ret, frame = video.read()
        if ret == True:
            num_frame += 1

            prediction = predict_emotion(frame)
            icon = emoticons[prediction]
            mode = emotions[prediction]
            output = cartoonize(frame, mode)
            if output is not None:
                writer.write(output)
            else:
                writer.write(frame)
        else:
            break

    video.release()
    writer.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    video_append()
