# **Model of Recognition and Classification of products with the purpose of automatic payment for Distributors**
  
  

## Table of Contents  
* [About](#about)
/* * [Problem-Solving](#problem-Solving)*/
* [How to](#how-to)
* [Architecture](#architecture)
* [Result](#result)
* [References](#references)
  


## About

object detection model과 classification model로 실시간 상품 인식/위치 파악을 한 뒤 어떤 상품인지 분류하는 프로그램입니다.  


위 프로그램이 결제 서비스와 연동되었을 때 대형 유통 기업, 도매업에 큰 이득을 불러올 수 있습니다.  
  
    

**1. 결제 과정의 간소화**
> 상품에서 바코드를 찾아 따로 찍을 필요 없이 단순히 카메라가 달린 계산대 위에 올려 놓기만 하면 됩니다.  

**2. 인건비 감소**
> 카운터에 항상 직원이 상주해 있지 않아도 되기 때문에, 비효율적인 인력 배치를 줄일 수 잇습니다.  

**3. 모바일 결제와 결합 가능**
> 빠르고 안전한 모바일 결제와의 결합으로 이미지 인식 자동계산 프로그램의 장점을 더욱 높일 수 있습니다.  
  
  
## Problem-Solving
SSD만으로 detection과 classification을 모두 진행했을 때 CNN 모델에 비해,  
1. 커버가능한 class의 수 적음
2. class가 늘어날 수록 비교적 정확도 낮음
  
**상품 인식 용도의 SSD + 인식된 상품을 분류하는 CNN** 형태의 엔진을 구성함으로 위의 문제 해결가능  
SSD와 CNN 모델의 연결고리로 Redis 사용  
![grab-landing-page](https://github.com/rib-president/Hide-on-bush/blob/master/sample/checkout_solution.JPG)





## How to
### 1. Data Preprocessing  
각기 다른 조명 환경을 지닌 여러 매장에 모두 적용시키는 것을 목적으로 Augmentation
* rotate
* brightness
* contrast
* RGB balance
  

``` python
def rotate_image(self, image):
    img = image
    rotatednum = random.choice([0, 90, 180, 270]) # choice rotation angle randomly
    rotatedImg = img.rotate(rotatednum, expand=True) # roate image
    return rotatedImg
```
  
``` python
def brightness(self, image, filter_num):
    input_im = image
    min_, max_ = filter_num
    num = round(random.uniform(min_, max_), 1)
    bright = ImageEnhance.Brightness(input_im)
    output_im = bright.enhance(num)
    return output_im
```
  
``` python
def contrast(self, image, filter_num):
    img = image
    min_, max_ = filter_num
    num = round(random.uniform(min_, max_), 1)
    enhancer = ImageEnhance.Contrast(img)
    cont = enhancer.enhance(num)
    return cont
```
  

### 2. Object Detection  
상품의 좌표정보가 있는 xml파일과 이미지를 이용하여 Caffe SSD 학습
  
  
![grab-landing-page](https://github.com/rib-president/Hide-on-bush/blob/master/sample/ssd_xml.JPG)
  
  
  
### 3. Classification  
* inception model로 feature extract
  
``` python
# First decode the JPEG image, resize it, and rescale the pixel values.
resized_input_values = sess.run(decoded_image_tensor,
                                  {image_data_tensor: image_data})
# Then run it through the recognition network.
bottleneck_values = sess.run(bottleneck_tensor,
                               {resized_input_tensor: resized_input_values})
bottleneck_values = np.squeeze(bottleneck_values)
```  
  
  

* mlp 학습
  
``` python
# mlp model
model = Sequential()
model.add(Dense(dense_num, kernel_initializer='uniform', activation='relu', input_dim=2048))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1], kernel_initializer='uniform', activation='softmax'))
model.compile(optimizer=optimizer_name, loss="categorical_crossentropy", metrics=["accuracy"])

# train
hist=model.fit(X_train, y_train, nb_epoch=epoch_size, batch_size=512, validation_data=(X_test, y_test), callbacks=callbacks_list)
```
  
  
### 4. Integration
Object Detection으로 인식된 상품의 좌표를 classfication model로 전달하기 위해 redis 사용
  
ex)
``` python
import redis
import json

#
obd = redis.StrictRedis(port=6380)
#
image_dict = {'imageData': cropped_imageData, 'index': index, 'width': w, 'height': h}
image_json = json.dumps(image_dict)
#
obd.rpush('obd', image_json)
```



  

## Architecture  


|Stages | Preprocessing | Object Detection |Classification|Integration|
|:-----:|:-------------:|:----------------:|:---------------------:|:---------:|
|**ML Models** | | SSD | Inception, MLP| |
|**Libraries** | OpenCV | Caffe | Tensorflow, Keras|redis, json|
|**Language** | Python | Python | Python| Python|
  


## Result
![grab-landing-page](https://github.com/rib-president/Hide-on-bush/blob/master/sample/output.gif)
#### <center>realtime inference</center>
  
  

## References
Data preprocessing: [OpenCV](http://opencv.org)  
Object Detection: [Caffe](https://github.com/wupeng78/weiliu89-caffe)   
Classification: [Inception](https://www.tensorflow.org/hub/tutorials/image_retraining)
