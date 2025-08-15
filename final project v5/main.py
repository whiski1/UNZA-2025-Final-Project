from imports import *
CAM_INDEX = 0
CAP_WIDTH  = 640
CAP_HEIGHT = 480
MAX_WORKERS = 2
SHOW_FPS = True

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise SystemExit("Could not open camera")

    hand_detector = HandBoundingBox()
    pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    fps_hist = deque(maxlen=30)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.time()

            # Run gaze and hands in parallel
            fut_hands = pool.submit(hand_detector.detect_and_draw, frame.copy())
            frame = fut_hands.result()

            coordinates = pool.submit(hand_detector.coordinates, frame.copy()).result()
            print(coordinates)
            

            fps_hist.append(1.0 / max(time.time() - t0, 1e-6))
            if SHOW_FPS:
                cv2.putText(frame, f"FPS: {sum(fps_hist)/len(fps_hist):.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)

            cv2.imshow("Gaze & Hands", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        # gaze_detector.close()
        hand_detector.close()
        pool.shutdown()

if __name__ == "__main__":
    main()
