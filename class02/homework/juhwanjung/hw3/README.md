# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx) juhwan@juhwan-500TFA-500SFA:~/otx_hw/classification-task$ ds_count ./splitted_dataset 2
./splitted_dataset:	207
./splitted_dataset/val:	42
./splitted_dataset/val/Rook:	21
./splitted_dataset/val/Knight:	21
./splitted_dataset/train:	165
./splitted_dataset/train/Rook:	81
./splitted_dataset/train/Knight:	84
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S|1.0|50.19|0:01:36.252919|32|0.01||
|EfficientNet-B0|0.9761904907226563|FPS : 163.56|0:00:28.698423|32|0.01||
|DeiT-Tiny|1.0|55.31|0:00:25.920379|32|0.01||
|MobileNet-V3-large-1x|0.9286|222.94|0:00:18.812921|32|0.01||

## FPS 측정 방법
'''
import time

# --------------------------- Step 6. Create infer request and do inference synchronously -----------------------------
    log.info('Starting inference in synchronous mode')
    start_time = time.time()
    results = compiled_model.infer_new_request({0: input_tensor})

# --------------------------- Step 7. Process output ------------------------------------------------------------------
    predictions = next(iter(results.values()))
    during_time = time.time() - start_time
    fps = 1 / during_time
    log.info(f"FPS : {fps:.2f}")
'''
