import sys
import cv2
import numpy as np

DEFAULT_VIDEO_PATH = './../resources/gh-ps2-gameplay-trimmed.mp4'

hsv_green = 25  # np.array([25, 183, 27])
hsv_red = 211  # np.array([211, 65, 56])
hsv_yellow = 194  # np.array([194, 203, 42])
hsv_blue = 33  # np.array([33, 129, 213])
hsv_orange = 184  # np.array([184, 95, 18])

hue_offset = 10


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


def apply_note_threshold(fretboard: np.ndarray, note_hue: int) -> np.ndarray:
    hsv = cv2.cvtColor(fretboard, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, np.array([note_hue - hue_offset, 100, 100]), np.array([note_hue + hue_offset, 255, 255]))
    res = cv2.bitwise_and(fretboard, fretboard, mask=mask)

    return res


def main():
    cap = load_video(DEFAULT_VIDEO_PATH)
    while cap.grab():
        _, frame = cap.retrieve()

        fretboard = crop_fretboard(frame)
        notes = apply_note_threshold(fretboard, hsv_green)

        cv2.imshow('Guitar Hero 2', notes)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return 0

if __name__ == '__main__':
    sys.exit(main())
