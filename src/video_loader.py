import sys
import cv2
import numpy as np

DEFAULT_VIDEO_PATH = './../resources/gh-ps2-gameplay-trimmed.mp4'


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


def apply_note_threshold(fretboard: np.ndarray, hue: int, hue_offset: int, saturation_lower: int, saturation_upper: int,
                         brightness_lower: int, brightness_upper: int) -> np.ndarray:
    hsv = cv2.cvtColor(fretboard, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([hue - hue_offset,
                            saturation_lower,
                            brightness_lower])
    np.clip(lower_bound, 0, 255, out=lower_bound)
    upper_bound = np.array([hue + hue_offset,
                            saturation_upper,
                            brightness_upper])
    np.clip(upper_bound, 0, 255, out=upper_bound)

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    res = cv2.bitwise_and(fretboard, fretboard, mask=mask)

    return res


def nothing(val):
    pass


def setup_slider_controls():
    cv2.namedWindow('HSV Controls')
    cv2.createTrackbar('Hue', 'HSV Controls', 0, 255, nothing)
    cv2.createTrackbar('Hue Offset', 'HSV Controls', 10, 20, nothing)
    cv2.createTrackbar('Saturation Lower', 'HSV Controls', 100, 255, nothing)
    cv2.createTrackbar('Saturation Upper', 'HSV Controls', 255, 255, nothing)
    cv2.createTrackbar('Brightness Lower', 'HSV Controls', 100, 255, nothing)
    cv2.createTrackbar('Brightness Upper', 'HSV Controls', 255, 255, nothing)


def main():
    setup_slider_controls()
    cap = load_video(DEFAULT_VIDEO_PATH)
    while cap.grab():
        _, frame = cap.retrieve()

        fretboard = crop_fretboard(frame)
        hue = cv2.getTrackbarPos('Hue', 'HSV Controls')
        hue_offset = cv2.getTrackbarPos('Hue Offset', 'HSV Controls')
        saturation_lower = cv2.getTrackbarPos('Saturation Lower', 'HSV Controls')
        saturation_upper = cv2.getTrackbarPos('Saturation Upper', 'HSV Controls')
        brightness_lower = cv2.getTrackbarPos('Brightness Lower', 'HSV Controls')
        brightness_upper = cv2.getTrackbarPos('Brightness Upper', 'HSV Controls')
        notes = apply_note_threshold(fretboard, hue, hue_offset,
                                     saturation_lower, saturation_upper,
                                     brightness_lower, brightness_upper)

        cv2.imshow('Guitar Hero 2', notes)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return 0

if __name__ == '__main__':
    sys.exit(main())
