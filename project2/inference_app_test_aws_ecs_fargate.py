import requests

# Replace this with the public IP address of your ECS Fargate service
host = "35.175.129.233"
port = 9696
url = f'http://{host}:{port}/inference_app'

image = {"url": "https://image.cnbcfm.com/api/v1/image/106467352-1585602933667virus-medical-flu-mask-health-protection-woman-young-outdoor-sick-pollution-protective-danger-face_t20_o07dbe.jpg"}

response = requests.post(url, json=image).json()
print(response)