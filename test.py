import cv2
from ultralytics import YOLO


def run_inference(video_path):
    try:
        # Load your custom-trained YOLOv8 model
        model = YOLO(r"id_detector.pt")
        print("YOLOv8 model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file at '{video_path}'")
        return

    print("Starting video processing... Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model(frame, conf=0.45, iou=0.4)

            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", cv2.resize(annotated_frame, (600, 800)))
            # cv2.imshow("YOLOv8 Inference", annotated_frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            print("End of video reached.")
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing finished and resources have been released.")


if __name__ == "__main__":
    VID_PATH = r"C:\Users\jays3\Documents\CODE\Internal_hacakathon\test\test_clips\test_clip_1.mp4"
    run_inference(VID_PATH)
