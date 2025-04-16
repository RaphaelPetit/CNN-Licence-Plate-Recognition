from watch_camera import watch_camera
from frame_handler_for_prediction import frame_handler_for_prediction
from check_gpu import check_gpu
import json

check_gpu()

isLocal = True

source = None
username = None
password = None

if isLocal:
    source = 0
else:
    # Configurações da câmera
    PORT = '554'               # Porta RTSP (geralmente 554)

    with open('./config/cameras.json', 'r') as file:
        cameras_dict = json.load(file)

    cameraConfig = cameras_dict['cancela6']

    host = cameraConfig['host']
    path = cameraConfig['path']
    username = cameraConfig['username']
    password = cameraConfig['password']

    source = f"rtsp://{host}:{PORT}{path}"
    print(f"URL da câmera: {source}")

rtsp_user = username
rtsp_password = password

watch_camera(source,
             use_gstreamer=False,
             rtsp_user=username,
             rtsp_password=password,
             frame_handler=frame_handler_for_prediction)
