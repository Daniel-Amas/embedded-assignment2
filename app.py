import cv2
import time

def detect_objects(frame, cascade, color, object_name, counters):
    # Detect objects using the given cascade
    objects = cascade.detectMultiScale(frame, 1.1, 1)
    
    # Draw rectangles around detected objects
    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    # Update the object count
    counters[object_name] += len(objects)

def main():
    # Use an absolute path for the video file
    video_path = 'videoplayback.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file at {video_path}")
        return

    # Load Haar cascades for different object types
    car_cascade = cv2.CascadeClassifier('cars.xml')
    bikes_cascade = cv2.CascadeClassifier('bikes.xml')
    pedestrian_cascade = cv2.CascadeClassifier('pedestrian.xml')
    bus_cascade = cv2.CascadeClassifier('bus.xml')

    print("Generated classification objects using provided XML files.")

    # Object counters
    counters = {
        'car': 0,
        'bike': 0,
        'pedestrian': 0,
        'bus': 0
    }

    num_frames = 0
    total_execution_time = 0.0 

    # Initialize the video writer for saving output
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Cannot read first frame from video.")
        return
    
    # Get the frame width and height
    frame_width, frame_height = int(frame.shape[1]), int(frame.shape[0])
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

    print("Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("End of video or invalid frame.")
            break

        num_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Measure the inference time for each frame
        start = time.time()

        detect_objects(frame, car_cascade, (0, 0, 255), 'car', counters)
        detect_objects(frame, bikes_cascade, (0, 255, 0), 'bike', counters)
        detect_objects(frame, pedestrian_cascade, (255, 0, 0), 'pedestrian', counters)
        detect_objects(frame, bus_cascade, (255, 255, 255), 'bus', counters)

        end = time.time()
        frame_execution_time = end - start
        total_execution_time += frame_execution_time
        print(f"Inference time for frame {num_frames}: {frame_execution_time:.4f} seconds")

        # Write the processed frame to the output file
        out.write(frame)

    # Release resources
    cap.release()
    out.release()

    print("Processing complete. Results:")
    print(f"Total frames processed: {num_frames}")
    print(f"Cars: {counters['car']} Bikes: {counters['bike']} Pedestrian: {counters['pedestrian']} Bus: {counters['bus']}")
    print(f"Total execution time: {total_execution_time:.2f} seconds")
    print(f"Average execution time per frame: {total_execution_time / num_frames:.4f} seconds")

if __name__ == "__main__":
    main()
