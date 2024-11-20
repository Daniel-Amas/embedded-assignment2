import cv2
import time

def detect_objects(frame, cascade, color, object_name, counters):
    objects = cascade.detectMultiScale(frame, 1.1, 1)
    
    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
       
    
    counters[object_name] += len(objects)

def main():
    cap = cv2.VideoCapture('video.mp4')
    
    car_cascade = cv2.CascadeClassifier('cars.xml')
    bikes_cascade = cv2.CascadeClassifier('bikes.xml')
    pedestrian_cascade = cv2.CascadeClassifier('pedestrian.xml')
    bus_cascade = cv2.CascadeClassifier('bus.xml')

    print("Generated classification objects through online XML files")

    counters = {
        'car': 0,
        'bike': 0,
        'pedestrian': 0,
        'bus': 0
    }

    num_frames = 0
    total_execution_time = 0.0 

    for _ in range(500):
        ret, frame = cap.read()
        if not ret:
            break

        num_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        start = time.time()

        detect_objects(frame, car_cascade, (0, 0, 255), 'car', counters)
        detect_objects(frame, bikes_cascade, (0, 255, 0), 'bike', counters)
        detect_objects(frame, pedestrian_cascade, (255, 0, 0), 'pedestrian', counters)
        detect_objects(frame, bus_cascade, (255, 255, 255), 'bus', counters)

        end = time.time()

        frame_execution_time = end - start
        total_execution_time += frame_execution_time  
        print(f"Inference time: {frame_execution_time}")

        cv2.imshow('video2', frame)

        if cv2.waitKey(33) == 27:
            break

    cv2.destroyAllWindows()

    print(f"Total inferences: {num_frames}")
    print(f"Cars: {counters['car']} Bikes: {counters['bike']} Pedestrian: {counters['pedestrian']} Bus: {counters['bus']}")

    print(f"Total execution time: {total_execution_time} seconds")
    print(f"Average execution time per frame: {total_execution_time / num_frames} seconds")

if __name__ == "__main__":
    main()