import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt  # Removendo matplotlib para evitar conflitos
import cv2
import tensorflow as tf
from tensorflow import keras
import easyocr
from ultralytics import YOLO
import time
import torch  # Importar torch explicitamente para configurações de GPU
import platform
import multiprocessing
import subprocess
import datetime
import re # Importar o módulo regex

# Detectar informações do sistema
print("\n===== INFORMAÇÕES DO SISTEMA =====")
print(f"Sistema Operacional: {platform.system()} {platform.release()}")
print(f"Processador: {platform.processor()}")
print(f"Número de CPUs: {multiprocessing.cpu_count()}")

# Verificar hardware gráfico disponível
try:
    gpu_info = subprocess.check_output('lspci | grep -E "VGA|3D|Display"', shell=True).decode('utf-8').strip()
    print("Hardware gráfico detectado:")
    for line in gpu_info.split('\n'):
        print(f"  {line}")

    # Verificar especificamente se existe GPU NVIDIA
    if "NVIDIA" in gpu_info:
        print("\nDetectada GPU NVIDIA que pode suportar CUDA!")

        # Verificar se nvidia-smi está disponível (drivers instalados)
        try:
            nvidia_smi = subprocess.check_output('which nvidia-smi', shell=True).decode('utf-8').strip()
            print(f"Driver NVIDIA encontrado: {nvidia_smi}")
        except:
            print("\n⚠️ AVISO: GPU NVIDIA detectada, mas drivers não estão instalados corretamente!")
            print("Para usar aceleração GPU, instale os drivers NVIDIA:")
            print("  - Em Ubuntu/Debian: sudo apt install nvidia-driver-XXX cuda-toolkit-XX.X")
            print("  - Em Arch Linux: sudo pacman -S nvidia nvidia-utils cuda")
            print("  - Em outras distribuições, consulte a documentação específica.")
            print("Após instalar, reinicie o sistema.")
except Exception as e:
    print(f"Não foi possível detectar informações detalhadas do hardware gráfico: {e}")
print("====================================\n")

# Configurar número de threads para otimizar CPU
try:
    # Definir número de threads para OpenCV
    cv2.setNumThreads(multiprocessing.cpu_count())
    print(f"OpenCV configurado para usar {multiprocessing.cpu_count()} threads")

    # Configurar paralelismo para TensorFlow
    tf.config.threading.set_intra_op_parallelism_threads(multiprocessing.cpu_count())
    tf.config.threading.set_inter_op_parallelism_threads(2)
    print(f"TensorFlow configurado para paralelismo: {multiprocessing.cpu_count()} intra-op, 2 inter-op threads")
except Exception as e:
    print(f"Erro ao configurar otimizações para CPU: {e}")

# Configurar TensorFlow para usar GPU se disponível
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"TensorFlow detectou {len(gpus)} GPU(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memória GPU configurada para crescimento dinâmico")
        # Usar GPU para TensorFlow
        tf.config.set_visible_devices(gpus, 'GPU')
    else:
        print("TensorFlow: Nenhuma GPU compatível detectada, usando CPU")
except Exception as e:
    print(f"Erro ao configurar GPU para TensorFlow: {e}")

# Verificar se o PyTorch pode usar CUDA
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"PyTorch usando GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memória GPU disponível: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # Configurar PyTorch para usar a GPU
        torch.cuda.set_device(0)
    else:
        # Verificar se existe GPU NVIDIA mas sem drivers
        nvidia_gpu_exists = False
        try:
            gpu_info = subprocess.check_output('lspci | grep -i nvidia', shell=True).decode('utf-8').strip()
            if gpu_info:
                nvidia_gpu_exists = True
                print("\n⚠️ AVISO: GPU NVIDIA detectada, mas PyTorch não consegue utilizá-la.")
                print("Verifique se os pacotes CUDA e cuDNN estão instalados corretamente.")
                print("Para instalação no Python, execute:")
                print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        except:
            pass

        if not nvidia_gpu_exists:
            # Otimizações para CPU
            torch.set_num_threads(multiprocessing.cpu_count())
            print(f"PyTorch usando CPU com {torch.get_num_threads()} threads")

        device = torch.device("cpu")
except Exception as e:
    # Otimizações para CPU em caso de erro
    torch.set_num_threads(multiprocessing.cpu_count())
    device = torch.device("cpu")
    print(f"Erro ao verificar CUDA para PyTorch: {e}")
    print(f"PyTorch usando CPU com {torch.get_num_threads()} threads")

model_path = './license_plate_detector.pt'

# Inicializar o modelo YOLO apenas uma vez (economiza tempo)
print("Carregando modelo YOLO...")
start_time = time.time()
model = YOLO(model_path)

# Forçar o modelo a usar a GPU se disponível
if device.type == "cuda":
    model.to(device)
    print("Modelo YOLO movido para GPU com sucesso")
else:
    print("Modelo YOLO usando CPU")

print(f"Modelo YOLO carregado em {time.time() - start_time:.2f} segundos")

# Inicializar o leitor OCR com suporte para inglês e português
print("Carregando modelo OCR...")
start_time = time.time()
try:
    # Forçar o uso de GPU para o EasyOCR
    reader = easyocr.Reader(['en', 'pt'], gpu=True if device.type == "cuda" else False)
    if device.type == "cuda":
        print("EasyOCR usando GPU")
    else:
        print("EasyOCR usando CPU")
except Exception as e:
    # Fallback para CPU se houver erro
    reader = easyocr.Reader(['en', 'pt'], gpu=False)
    print(f"EasyOCR usando CPU devido a erro: {e}")
print(f"Modelo OCR carregado em {time.time() - start_time:.2f} segundos")

# Variável global para manter a última placa detectada (efeito de persistência)
last_plate_text = "Sem placa identificada"
last_confidence = 0.0
no_detection_count = 0
max_no_detection = 10  # Quantos frames sem detecção antes de resetar

# Variáveis para armazenar tempos de execução para análise de performance
execution_times = {
    'total': [],
    'detection': [],
    'ocr': [],
    'rendering': []
}

def predict_license_plate(frame):
    # Inicia o timer para medição de performance
    start_time = datetime.datetime.now()
    
    # Define o padrão regex para placas Mercosul (LLLNLNN) e placas antigas (LLLNNNN)
    plate_pattern = re.compile(r'^[A-Z]{3}[0-9][A-Z0-9][0-9]{2}$')
    
    # Realiza a detecção
    results = model(frame)
    
    # Calcula o tempo de detecção
    detection_time = (datetime.datetime.now() - start_time).total_seconds()
    
    # Inicia o timer para OCR
    ocr_start_time = datetime.datetime.now()
    
    # Processa os resultados
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            try:
                # Obtém as coordenadas da caixa delimitadora
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Obtém a confiança da detecção YOLO
                yolo_confidence = float(box.conf[0].cpu().numpy())
                
                # Obtém a classe
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                # Se for uma placa (assumindo que o modelo detecta 'license_plate')
                if class_name == 'license_plate': # Certifique-se que 'license_plate' é a classe correta
                    # Recorta a região da placa
                    plate_region = frame[y1:y2, x1:x2]
                    
                    # Realiza OCR na região da placa
                    # O readtext retorna uma lista de tuplas: (bbox, text, confidence)
                    allowed_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    ocr_results = reader.readtext(plate_region, allowlist=allowed_chars)
                    
                    # Calcula o tempo total de OCR (pode incluir múltiplas tentativas)
                    ocr_time = (datetime.datetime.now() - ocr_start_time).total_seconds()
                    
                    best_plate_text = "" # Guarda o melhor texto de placa validado
                    best_ocr_confidence = 0.0 # Guarda a confiança OCR do melhor texto

                    # Valida os resultados do OCR contra o padrão LLLNLNN e LLLNNNN
                    for (_, text, ocr_conf) in ocr_results:
                        # Limpa o texto: maiúsculas, remove não alfanuméricos
                        cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                        
                        # Verifica se corresponde ao padrão
                        match = plate_pattern.fullmatch(cleaned_text)
                        
                        # Se correspondeu e tem maior confiança que o melhor anterior
                        if match and ocr_conf > best_ocr_confidence:
                            best_plate_text = cleaned_text
                            best_ocr_confidence = ocr_conf
                    
                    # Adiciona a detecção aos resultados APENAS se um texto válido foi encontrado
                    if best_plate_text:
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': yolo_confidence, # Confiança da detecção YOLO
                            'ocr_confidence': best_ocr_confidence, # Confiança do OCR
                            'class': class_name,
                            'text': best_plate_text, # Texto validado
                            'detection_time': detection_time,
                            'ocr_time': ocr_time,
                            'plate_image': plate_region
                        })
            except Exception as e:
                # Log detalhado do erro
                print(f"Erro ao processar detecção: Classe={class_name}, Coords=({x1},{y1},{x2},{y2}), Erro: {e}")
                continue
    
    # Retorna a lista de detecções validadas
    return detections
