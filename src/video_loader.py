import cv2
import sys

DEFAULT_VIDEO_PATH = './../resources/gh-ps2-gameplay-trimmed.mp4'


def load_video(filepath: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture()
    cap.open(filepath)
    return cap


def main():
    cap = load_video(DEFAULT_VIDEO_PATH)
    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', gray)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return 0

if __name__ == '__main__':
    sys.exit(main())
