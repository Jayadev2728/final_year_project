"""
find_camera_index.py
----------------------
Shows a live preview for each working camera index, one at a time, so
you can visually confirm which index is your phone (Iriun) vs your
laptop's built-in webcam.

For each index: a window opens showing the live feed.
  - Press N to move to the next index
  - Press Q to stop and print the index you want to use
"""

import cv2

for i in range(4):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Index {i}: not available")
        cap.release()
        continue

    print(f"\nIndex {i}: showing preview. Press N for next index, Q to stop here and use this one.")
    chosen = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Index {i}: opened but couldn't read a frame, skipping.")
            break

        cv2.putText(frame, f"Camera index {i}  (N = next, Q = use this one)",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Camera Preview", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            break
        if key == ord('q'):
            chosen = True
            break

    cap.release()
    cv2.destroyAllWindows()

    if chosen:
        print(f"\n>>> Use index {i} in main.py:")
        print(f">>> cap = cv2.VideoCapture({i}, cv2.CAP_DSHOW)")
        break
else:
    print("\nFinished checking all indices without choosing one.")
