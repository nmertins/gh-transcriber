import sys
import cv2
import numpy as np

DEFAULT_VIDEO_PATH = './../resources/gh-ps2-gameplay-trimmed.mp4'
SLIDER_WINDOW_NAME = 'HSV Controls'
HUE_SLIDER_NAME = 'Hue'
HUE_OFFSET_SLIDER_NAME = 'Hue Offset'
SATURATION_LOWER_SLIDER_NAME = 'Saturation Lower'
SATURATION_UPPER_SLIDER_NAME = 'Saturation Upper'
BRIGHTNESS_LOWER_SLIDER_NAME = 'Brightness Lower'
BRIGHTNESS_UPPER_SLIDER_NAME = 'Brightness Upper'


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


def apply_note_threshold(fretboard: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(fretboard, cv2.COLOR_BGR2HSV)

    np.clip(lower_bound, 0, 255, out=lower_bound)
    np.clip(upper_bound, 0, 255, out=upper_bound)

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    res = cv2.bitwise_and(fretboard, fretboard, mask=mask)

    return res


def nothing(val):
    pass


def setup_slider_controls():
    cv2.namedWindow(SLIDER_WINDOW_NAME)
    cv2.resizeWindow(SLIDER_WINDOW_NAME, 600, 200)
    cv2.createTrackbar(HUE_SLIDER_NAME, SLIDER_WINDOW_NAME, 0, 255, nothing)
    cv2.createTrackbar(HUE_OFFSET_SLIDER_NAME, SLIDER_WINDOW_NAME, 10, 20, nothing)
    cv2.createTrackbar(SATURATION_LOWER_SLIDER_NAME, SLIDER_WINDOW_NAME, 100, 255, nothing)
    cv2.createTrackbar(SATURATION_UPPER_SLIDER_NAME, SLIDER_WINDOW_NAME, 255, 255, nothing)
    cv2.createTrackbar(BRIGHTNESS_LOWER_SLIDER_NAME, SLIDER_WINDOW_NAME, 100, 255, nothing)
    cv2.createTrackbar(BRIGHTNESS_UPPER_SLIDER_NAME, SLIDER_WINDOW_NAME, 255, 255, nothing)


def read_sliders():
    hue = cv2.getTrackbarPos(HUE_SLIDER_NAME, SLIDER_WINDOW_NAME)
    hue_offset = cv2.getTrackbarPos(HUE_OFFSET_SLIDER_NAME, SLIDER_WINDOW_NAME)
    saturation_lower = cv2.getTrackbarPos(SATURATION_LOWER_SLIDER_NAME, SLIDER_WINDOW_NAME)
    saturation_upper = cv2.getTrackbarPos(SATURATION_UPPER_SLIDER_NAME, SLIDER_WINDOW_NAME)
    brightness_lower = cv2.getTrackbarPos(BRIGHTNESS_LOWER_SLIDER_NAME, SLIDER_WINDOW_NAME)
    brightness_upper = cv2.getTrackbarPos(BRIGHTNESS_UPPER_SLIDER_NAME, SLIDER_WINDOW_NAME)

    lower_bound = np.array([hue - hue_offset,
                            saturation_lower,
                            brightness_lower])
    upper_bound = np.array([hue + hue_offset,
                            saturation_upper,
                            brightness_upper])

    return lower_bound, upper_bound


def main():
    setup_slider_controls()
    cap = load_video(DEFAULT_VIDEO_PATH)
    while cap.grab():
        _, frame = cap.retrieve()

        fretboard = crop_fretboard(frame)

        lower_bound, upper_bound = read_sliders()

        notes = apply_note_threshold(fretboard, lower_bound, upper_bound)

        cv2.imshow('Guitar Hero 2', notes)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return 0

if __name__ == '__main__':
    sys.exit(main())
