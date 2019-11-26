import cv2
import numpy as np
import redis
import base64
import io
from PIL import Image
import json

r = redis.StrictRedis()
origin = cv2.imread('./cute.jpg')

def pil_enco(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img, 'RGB')
    buf = io.BytesIO()
    buf.flush()
    image.save(buf, format='JPEG')
    value = buf.getvalue()
    buf.flush()
    enco = base64.b64encode(value)
    
    return enco


def pil_deco(img):
    deco = base64.b64decode(img)
    image = Image.open(io.BytesIO(deco))
    
    return image
    
    
def cv2_enco(img):
    image = img.tobytes()
    enco = base64.b64encode(image)
    
    return enco


def cv2_deco(img):
    deco = base64.b64decode(img)
    sha = origin.shape
    dty = origin.dtype
    image = np.frombuffer(deco, dtype=dty).reshape(sha)
    
    return image
    
    
#pil_img = pil_enco(origin)
cv2_img = cv2_enco(origin)

cv2_frame = []
cv2_frame.append({"image":cv2_img})

cv2_dict = {'cv2_test' : cv2_frame}
cv2_json = json.dumps(cv2_dict)

r.set('cv2', cv2_json)

cv2_data = r.get('cv2')
cv2_data = json.loads(str(cv2_data))
key = cv2_data.keys()[0]
cv2_result = cv2_data[key][0]

cv2_img = cv2_result["image"]

#pil_img = pil_deco(pil_img)
cv2_img = cv2_deco(cv2_img)


for i in range(origin.shape[0]):
    for j in range(origin.shape[1]):
        if origin[i][j][0] != cv2_img[i][j][0] and origin[i][j][1] != cv2_img[i][j][1] and origin[i][j][2] != cv2_img[i][j][2]:
            print origin[i][j], cv2_img[i][j]

print 'same'

#print origin == cv2_img
#print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
#pil_img = np.array(pil_img)

#print origin == pil_img

