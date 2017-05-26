import sys
import cv2
import numpy as np

DEFAULT_VIDEO_PATH = './../resources/gh-ps2-gameplay-trimmed.mp4'

hsv_green = np.array([25, 183, 27])
hsv_red = np.ndarray([211, 65, 56])
hsv_yellow = np.ndarray([194, 203, 42])
hsv_blue = np.ndarray([33, 129, 213])
hsv_orange = np.ndarray([184, 95, 18])


def load_video(filepath: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture()
    cap.open(filepath)
    return cap


def crop_fretboard(gameplay_image: np.ndarray) -> np.ndarray:
    rows = gameplay_image.shape[0]
    cols = gameplay_image.shape[1]

    start_row = int(0.6 * rows)
    end_row = int(0.8 * rows)

    start_col = int(0.25 * cols)
    end_col = int(0.75 * cols)

    cropped_grayscale = gameplay_image[start_row:end_row, start_col:end_col]

    return cropped_grayscale


def apply_note_thresholds(fretboard: np.ndarray):
    hsv = cv2.cvtColor(fretboard, cv2.COLOR_BGR2HSV)

    # green_mask = cv2.inRange(hsv, hsv_green)

    return hsv


def main():
    cap = load_video(DEFAULT_VIDEO_PATH)
    while cap.grab():
        _, frame = cap.retrieve()

        fretboard = crop_fretboard(frame)
        notes = apply_note_thresholds(fretboard)

        cv2.imshow('Guitar Hero 2', notes)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return 0

if __name__ == '__main__':
    sys.exit(main())
