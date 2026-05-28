"""
This file is for testing results. 

Visualizing the pixels refereced from the splats in the image.

"""

import cv2

img = cv2.imread(r"C:\Users\Valeria\OneDrive - Rice University\calibration_viser\kinect_3cam_dataset\images\cam0.png")

x = 930
y = 843

cv2.circle(img, (x, y), 10, (0, 0, 255), -1)

cv2.putText(
    img,
    f"({x}, {y})",
    (x + 15, y - 15),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 0, 255),
    2
)

cv2.imshow("debug", img)
cv2.waitKey(0)
cv2.destroyAllWindows()