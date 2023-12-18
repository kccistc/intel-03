from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
import requests


base_artifacts_dir = Path("./artifacts").expanduser()
base_data_dir = Path("./data").expanduser()

model_name = "v3-small_224_1.0_float"
model_xml_name = f'{model_name}.xml'
model_bin_name = f'{model_name}.bin'

model_xml_path = base_artifacts_dir / model_xml_name
model_bin_path = base_artifacts_dir / model_bin_name

base_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/mobelinet-v3-tf/FP32/'


def download_file(url, destination):
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination, 'wb') as file:
            file.write(response.content)
            print("XML 다운완료")
    else:
        print("failed to dwonload file")


if not model_xml_path.exists():
    download_file(base_url + model_xml_name,model_xml_path)
    download_file(base_url + model_xml_name,model_bin_path)
else:
    print(f'{model_name} already downloaded to {base_artifacts_dir}')
    
core = ov.Core()
model = core.read_model(model=model_xml_path)
compiled_model = core.compile_model(model=model, device_name="CPU")

output_layer = compiled_model.output(0)


# The MobileNet model expects images in RGB format.
image = cv2.cvtColor(cv2.imread("data/gos.jpeg"), code=cv2.COLOR_BGR2RGB)

# Resize to MobileNet image shape.
input_image = cv2.resize(src=image, dsize=(224, 224))

# Reshape to model input shape.
input_image = np.expand_dims(input_image, 0)
plt.imshow(image)
plt.show()

result_infer = compiled_model([input_image])[output_layer]
result_index = np.argmax(result_infer)

imagenet_name = "imagenet_2012.txt"
imagenet_path = base_data_dir / imagenet_name

imagenet_filename = download_file(
    "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/datasets/imagenet/imagenet_2012.txt",
   imagenet_path
)
if imagenet_path:
    with open(imagenet_path, 'r') as file:
        imagenet_classes = file.read().splitlines()
        imagenet_result = ['background'] + imagenet_classes
        print(imagenet_result[result_index])
              

