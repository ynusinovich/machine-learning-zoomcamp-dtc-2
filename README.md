# Mask Detector to Stop Airborne Disease Spread

## Inspiration
- COVID-19 was significant due to its global impact as a highly contagious respiratory illness caused by the SARS-CoV-2 virus, leading to widespread illness, significant loss of life, strain on healthcare systems, and profound socio-economic disruptions.
- Masks are useful against airborne diseases as they act as a barrier, preventing the spread of respiratory droplets that may contain infectious agents, thereby reducing the risk of transmission from infected individuals to others in close proximity.
- Accurately checking for mask usage in public places, such as airports, can help enforce mask requirements.
- IoU, or Intersection over Union, is a metric used in object detection to evaluate the accuracy of the model's predictions by measuring the overlap between the predicted and ground truth bounding boxes of objects in an image. Higher IoU values indicate better alignment between predicted and actual object locations.

## Credit and References
- Starter code credit: https://medium.com/@doleron/building-your-own-object-detector-from-scratch-bfeadfaddad8
- Data source: https://www.kaggle.com/datasets/techzizou/labeled-mask-dataset-yolo-darknet
- Performance tips: https://www.tensorflow.org/datasets/performances
- Better performance with the tf.data API tips: https://www.tensorflow.org/guide/data_performance
- IoU for object detection: https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
- Deployment in ECS Fargate: https://aws.plainenglish.io/deploying-a-docker-container-in-aws-using-fargate-5a19a140b018.

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

## Run Data Splitting and Preliminary Model Training Notebook
1. sudo apt install graphviz
- If you can't install graphviz, you can skip the cell that visualizes the neural network.
- Run all cells.

## Run EDA notebook
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
- If you would like to visualize the box coordinates that are returned, try running check_inference_box.ipynb, but don't forget to modify the box coordinates and image URL accordingly.
- Please note, the code returns the coordinates of the box in the resized image. If you prefer to return the coordinates in the original image, please modify the code accordingly.

## Run App in Terminal
1. python3 inference_app.py
2. python3 inference_app_test.py
- You can modify the input image URL in inference_app_test.py.

## Run App with Docker
1. docker build -t mask .
2. docker run -it -p 9696:9696 mask:latest
3. python3 inference_app_test.py
- You can modify the input image URL in inference_app_test.py.

## Deploy to ECS Fargate
1. docker build -t mask .
- Create a repository named mask in Amazon ECR. Modify the following commands based on the repository's account ID and region.
2. docker tag mask:latest 614132154255.dkr.ecr.us-east-1.amazonaws.com/mask:latest
3. aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 614132154255.dkr.ecr.us-east-1.amazonaws.com
4. docker push 614132154255.dkr.ecr.us-east-1.amazonaws.com/mask:latest
- Create an Amazon ECS cluster, define a task definition, and create an ECS service to deploy the Docker container. If you need help with this, follow steps 4-6 from the last link in the "Credit and References" section above.
- Don't forget to create an inbound rule for your security group that allows all traffic on port 9696.
- If you want to try my API on ECS Fargate, you can run inference_app_test_aws_eb.py.
- You can modify the input image URL in inference_app_test_aws_eb.py.