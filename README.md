# Mask Detector to Stop Airborne Disease Spread

## Inspiration
- COVID-19 was significant due to its global impact as a highly contagious respiratory illness caused by the SARS-CoV-2 virus, leading to widespread illness, significant loss of life, strain on healthcare systems, and profound socio-economic disruptions.
- Masks are useful against airborne diseases as they act as a barrier, preventing the spread of respiratory droplets that may contain infectious agents, thereby reducing the risk of transmission from infected individuals to others in close proximity.
- Accurately checking for mask usage in public places, such as airports, can help enforce mask requirements.
- IoU, or Intersection over Union, is a metric used in object detection to evaluate the accuracy of the model's predictions by measuring the overlap between the predicted and ground truth bounding boxes of objects in an image. Higher IoU values indicate better alignment between predicted and actual object locations.

## Data
- The dataset containes 1,292 images of people, and corresponding label files.
- Some are with masks, and some are without.
- The label files specify whether the person is wearing a mask, and the bounding box of the face in the image.

## Approach
- I created a multitask neural network using the TensorFlow functional API.
- During training, the neural network learns to identify a person's face and whether they're wearing a mask.

## Clone the Repo
1. git clone https://github.com/ynusinovich/machine-learning-zoomcamp-dtc-2

## Create and Activate Virtual Environment
1. cd project2
2. conda create --name mask python=3.10.13
3. conda activate mask
4. conda install pipenv
5. pipenv install
6. pipenv install --dev (if running notebook or deploying to AWS Elastic Beanstalk)
7. pipenv shell

## Run EDA, Data Splitting, and Preliminary Model Training Notebook
1. sudo apt install graphviz
- If you can't install graphviz, you can skip the cell that visualizes the neural network.
- Run all cells.

## Run Model Training
1. python3 training.py
- Feel free to modify the hyperparameters or the neural network structure.
- Look at results.csv to pick an optimal model.

## Change Model Path
- You will have to modify the saved_model_path variable in the predict function of inference.py, as well as in the Dockerfile, to your best model.
- I have temporarily loaded my best model to Google Drive, so if you run any of the code below before training your own model, first download the model from the following link:
- https://drive.google.com/file/d/1UFuIH1GVldfDolrAcsQdbKeeMuDgSGc3/view?usp=drive_link
- I chose a best model based on high accuracy and high IoU on the test dataset.

## Run Predictions in Terminal
1. python3 inference.py --data '{"url": "https://image.cnbcfm.com/api/v1/image/106467352-1585602933667virus-medical-flu-mask-health-protection-woman-young-outdoor-sick-pollution-protective-danger-face_t20_o07dbe.jpg"}'
- You can modify the input image URL in the command above.

## Run App in Terminal
1. python3 inference_app.py
2. python3 inference_app_test.py
- You can modify the input image URL in inference_app_test.py.

## Run App with Docker
1. docker build -t mask .
2. docker run -it -p 9696:9696 mask:latest
3. python3 inference_app_test.py
- You can modify the input image URL in inference_app_test.py.

## Deploy to AWS Elastic Beanstalk
1. eb init -p docker -r us-east-1 mask
2. eb create mask
3. python3 inference_app_test_aws_eb.py
...
4. eb terminate mask
- You can modify the input image URL in inference_app_test_aws_eb.py.