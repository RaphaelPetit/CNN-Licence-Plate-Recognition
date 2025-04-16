from predict_license_plate import predict_license_plate
import torch

def frame_handler_for_prediction(frame):
    """
    Processa o frame e retorna os dados de predição.
    """
    # Obter predições e dados
    detections = predict_license_plate(frame)

    # Formatar dados para a visualização
    if detections:
        # Usar a primeira detecção (mais confiável)
        detection = detections[0]
        prediction_data = {
            'frame_info': {
                'height': frame.shape[0],
                'width': frame.shape[1]
            },
            'detected_boxes': [{
                'coords': detection['bbox'],
                'confidence': detection['confidence']
            }],
            # 'plate_text': detection['text'],
            'plate_text': ''.join([obj['text'] for obj in detections]),
            'confidence': detection['confidence'],
            'device_info': {
                'type': 'gpu' if torch.cuda.is_available() else 'cpu',
                'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
            },
            'execution_times': {
                'detection': detection['detection_time'],
                'ocr': detection['ocr_time'],
                'total': detection['detection_time'] + detection['ocr_time']
            },
            'found_plate': True,
            'cropped_plate': detection['plate_image']
        }
    else:
        # Se não houver detecções, criar dados vazios
        prediction_data = {
            'frame_info': {
                'height': frame.shape[0],
                'width': frame.shape[1]
            },
            'detected_boxes': [],
            'plate_text': "Sem placa identificada",
            'confidence': 0.0,
            'device_info': {
                'type': 'gpu' if torch.cuda.is_available() else 'cpu',
                'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
            },
            'execution_times': {
                'detection': 0.0,
                'ocr': 0.0,
                'total': 0.0
            },
            'found_plate': False,
            'cropped_plate': None
        }
    
    return prediction_data
