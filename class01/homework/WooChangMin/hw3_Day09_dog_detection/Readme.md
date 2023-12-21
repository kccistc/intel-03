# 코드

```shell

def crop_images(bgr_image, resized_image, boxes, threshold=0.6) -> np.ndarray:
    """
    Use bounding boxes from detection model to find the absolute car position
    
    :param: bgr_image: raw image
    :param: resized_image: resized image
    :param: boxes: detection model returns rectangle position
    :param: threshold: confidence threshold
    :returns: car_position: car's absolute position
    """
    # Fetch image shapes to calculate ratio
    (real_y, real_x), (resized_y, resized_x) = bgr_image.shape[:2], resized_image.shape[:2]
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

    # Find the boxes ratio
    boxes = boxes[:, 1:]
    #print(boxes.shape)#2번째 값부터만 저장하니까기존 (2, 7) ->  (2, 5)로 변화됨. 이제 image_id, label이 사라지고 conf값이 0번 index에 들어가있을거임.

    # Store the vehicle's position
    car_position = []
    # Iterate through non-zero boxes
    for box in boxes:
        # Pick confidence factor from last place in array
        conf = box[1]
        if conf > threshold:
            # Convert float to int and multiply corner position of each box by x and y ratio
            # In case that bounding box is found at the top of the image, 
            # upper box  bar should be positioned a little bit lower to make it visible on image 
            (x_min, y_min, x_max, y_max) = [
                int(max(corner_position * ratio_y * resized_y, 10)) if idx % 2 
                else int(corner_position * ratio_x * resized_x)
                for idx, corner_position in enumerate(box[2:])
            ]
            
            car_position.append([x_min, y_min, x_max, y_max])
            
    return car_position
    
```
    
# 변화사항

```shell
boxes = boxes[:, 1:]#1번 라인부터 값을 저장하니 box에는 label,conf,x_min,.....이런식으로 값이 저장될거임.
고로 boxes에 접근하는 코드 부분들을 다  수정해주면 됨.
conf = box[1]
for idx, corner_position in enumerate(box[2:])
```









==============================DOG Classification and detection====================================
YoLo 활용해서 detection 하였음.

실행결과는 result.jpg 참고


    
