import time
import cv2
cap = cv2.VideoCapture(0)
cnt = 0
#cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    if ret:
        if ret == True:
            print time_stamp
            #cv2.imshow('frame',img)
            cv2.imwrite('frames/' + str(cnt) +'.jpg', img) #make frames dir
            cnt += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

cap.release()
cv2.destroyAllWindows()

