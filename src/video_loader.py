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

HUE_INDEX = 0
HUE_OFFSET_INDEX = 1
SATURATION_LOWER_INDEX = 2
SATURATION_UPPER_INDEX = 3
BRIGHTNESS_LOWER_INDEX = 4
BRIGHTNESS_UPPER_INDEX = 5


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

    cropped_video = gameplay_image[start_row:end_row, start_col:end_col]

    return cropped_video


def apply_note_threshold(fretboard: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(fretboard, cv2.COLOR_BGR2HSV)

    np.clip(lower_bound, 0, 255, out=lower_bound)
    np.clip(upper_bound, 0, 255, out=upper_bound)

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    res = cv2.bitwise_and(fretboard, fretboard, mask=mask)

    return res


def nothing(val: int) -> None:
    pass


def setup_slider_controls() -> None:
    cv2.namedWindow(SLIDER_WINDOW_NAME)
    cv2.resizeWindow(SLIDER_WINDOW_NAME, 600, 200)
    cv2.createTrackbar(HUE_SLIDER_NAME, SLIDER_WINDOW_NAME, 0, 255, nothing)
    cv2.createTrackbar(HUE_OFFSET_SLIDER_NAME, SLIDER_WINDOW_NAME, 10, 20, nothing)
    cv2.createTrackbar(SATURATION_LOWER_SLIDER_NAME, SLIDER_WINDOW_NAME, 100, 255, nothing)
    cv2.createTrackbar(SATURATION_UPPER_SLIDER_NAME, SLIDER_WINDOW_NAME, 255, 255, nothing)
    cv2.createTrackbar(BRIGHTNESS_LOWER_SLIDER_NAME, SLIDER_WINDOW_NAME, 100, 255, nothing)
    cv2.createTrackbar(BRIGHTNESS_UPPER_SLIDER_NAME, SLIDER_WINDOW_NAME, 255, 255, nothing)


def read_slider_values() -> tuple:
    hue = cv2.getTrackbarPos(HUE_SLIDER_NAME, SLIDER_WINDOW_NAME)
    hue_offset = cv2.getTrackbarPos(HUE_OFFSET_SLIDER_NAME, SLIDER_WINDOW_NAME)
    saturation_lower = cv2.getTrackbarPos(SATURATION_LOWER_SLIDER_NAME, SLIDER_WINDOW_NAME)
    saturation_upper = cv2.getTrackbarPos(SATURATION_UPPER_SLIDER_NAME, SLIDER_WINDOW_NAME)
    brightness_lower = cv2.getTrackbarPos(BRIGHTNESS_LOWER_SLIDER_NAME, SLIDER_WINDOW_NAME)
    brightness_upper = cv2.getTrackbarPos(BRIGHTNESS_UPPER_SLIDER_NAME, SLIDER_WINDOW_NAME)

    return hue, hue_offset, saturation_lower, saturation_upper, brightness_lower, brightness_upper


def get_bounds() -> tuple:
    slider_values = read_slider_values()
    hue = slider_values[HUE_INDEX]
    hue_offset = slider_values[HUE_OFFSET_INDEX]
    saturation_lower = slider_values[SATURATION_LOWER_INDEX]
    saturation_upper = slider_values[SATURATION_UPPER_INDEX]
    brightness_lower = slider_values[BRIGHTNESS_LOWER_INDEX]
    brightness_upper = slider_values[BRIGHTNESS_UPPER_INDEX]

    lower_bound = np.array([hue - hue_offset,
                            saturation_lower,
                            brightness_lower])
    upper_bound = np.array([hue + hue_offset,
                            saturation_upper,
                            brightness_upper])

    return lower_bound, upper_bound


def print_variable_values() -> None:
    slider_values = read_slider_values()
    hue = slider_values[HUE_INDEX]
    hue_offset = slider_values[HUE_OFFSET_INDEX]
    saturation_lower = slider_values[SATURATION_LOWER_INDEX]
    saturation_upper = slider_values[SATURATION_UPPER_INDEX]
    brightness_lower = slider_values[BRIGHTNESS_LOWER_INDEX]
    brightness_upper = slider_values[BRIGHTNESS_UPPER_INDEX]

    print('Hue: ', hue)
    print('Hue Offset: ', hue_offset)
    print('Saturation Lower Bound: ', saturation_lower)
    print('Saturation Upper Bound: ', saturation_upper)
    print('Brightness Lower Bound: ', brightness_lower)
    print('Brightness Upper Bound: ', brightness_upper)


def main() -> int:
    setup_slider_controls()
    cap = load_video(DEFAULT_VIDEO_PATH)
    grab_new_frame = True
    while True:
        if grab_new_frame:
            cap.grab()
            _, frame = cap.retrieve()

        fretboard = crop_fretboard(frame)

        lower_bound, upper_bound = get_bounds()
        notes = apply_note_threshold(fretboard, lower_bound, upper_bound)

        cv2.imshow('HSV View', notes)
        cv2.imshow('Guitar Hero 2', fretboard)

        wait_key = cv2.waitKey(33) & 0xFF

        if wait_key == ord('p'):
            print_variable_values()

        if wait_key == ord(' '):
            grab_new_frame = not grab_new_frame

        if wait_key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return 0

if __name__ == '__main__':
    sys.exit(main())
