# coding:utf-8
import cv2
import os
import time

def cam():
    cap = cv2.VideoCapture(0)
    before = None
    while True:
        if not cap.isOpened():
            return
        ret, frame = cap.read()
        frame = cv2.resize(frame, (int(frame.shape[1]/4), int(frame.shape[0]/4)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 前フレームを保存
        if before is None:
            before = gray.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray, before, 0.6)
        mdframe = cv2.absdiff(gray, cv2.convertScaleAbs(before))
        thresh = cv2.threshold(mdframe, 3, 255, cv2.THRESH_BINARY)[1]
        image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        target = contours[0]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if max_area < area and area < 10000 and area > 2000:
                max_area = area
                target = cnt

        if max_area <= 2000:
            area_frame = frame
            cv2.putText(area_frame, 'not detected', (0, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow("frame", frame)
        else:
            cv2.putText(area_frame, 'detected!!!!', (0, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3, cv2.LINE_AA)
            x, y, w, h = cv2.boundingRect(target)
            area_frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("frame", frame)
            print("d")
            # time.sleep(3)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyWindow('frame')


if __name__ == '__main__':
    cam()