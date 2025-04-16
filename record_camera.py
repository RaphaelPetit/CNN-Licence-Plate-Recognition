import cv2
import time
import os
import sys

def record_camera(source, record_time, output_filename="webcam.avi", window_name="Recording", codec="MJPG", use_gstreamer=False, rtsp_user=None, rtsp_password=None):
    """
    Grava um vídeo a partir de uma fonte de câmera e salva todos os frames na resolução máxima disponível.

    Parâmetros:
      - source: URL da câmera (ex.: "rtsp://192.168.2.137:554/axis-media/media.amp")
      - record_time: tempo de gravação em segundos
      - output_filename: nome do arquivo de saída (padrão: "webcam.avi")
      - window_name: nome da janela onde o vídeo será exibido durante a gravação (padrão: "Recording")
      - codec: codec de vídeo a ser utilizado (padrão: "MJPG"). Exemplos: "mp4v", "avc1", "XVID", "MJPG"
      - use_gstreamer: se True, utiliza pipeline GStreamer para capturar o stream
      - rtsp_user: (opcional) usuário para autenticação RTSP
      - rtsp_password: (opcional) senha para autenticação RTSP

    Durante a gravação, os frames são exibidos e salvos. Para interromper antes do tempo, pressione 'q'.
    """
    # Certificar que todas as janelas anteriores estão fechadas
    cv2.destroyAllWindows()
    
    # Se rtsp_user e rtsp_password foram fornecidos, mas não estão na URL,
    # tentar criar uma URL com credenciais embutidas
    if rtsp_user and rtsp_password and "://" in source and "@" not in source:
        # Extrair o protocolo e o restante da URL
        protocol, rest = source.split("://", 1)
        # Inserir as credenciais
        source = f"{protocol}://{rtsp_user}:{rtsp_password}@{rest}"
    
    print("Abrindo fonte de vídeo:", source)
    
    cap = None
    out = None
    success = False
    
    try:
        if use_gstreamer:
            # Configurando uma pipeline GStreamer para RTSP
            # Usando protocolos=4 para forçar o uso de TCP que é mais estável
            gst_pipeline = (
                f'rtspsrc location="{source}" latency=200 protocols=4 ! '
                'rtpjitterbuffer ! decodebin ! videoconvert ! '
                'appsink sync=false'
            )
            print(f"Usando GStreamer com a seguinte pipeline: {gst_pipeline}")
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        else:
            # Usando a captura padrão do OpenCV
            print("Usando OpenCV para conexão direta")
            cap = cv2.VideoCapture(source)
            
        if not cap.isOpened():
            print(f"Não foi possível acessar a câmera: {source}")
            print("Tente usar uma URL com as credenciais embutidas: rtsp://usuário:senha@host:porta/caminho")
            return False

        # Captura o primeiro frame para definir a resolução
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Erro ao capturar o primeiro frame")
            return False

        height, width = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
            print("FPS inválido obtido. Usando valor padrão de 30 FPS.")

        print(f"Resolução obtida: {width}x{height}")
        print(f"FPS obtido: {fps}")

        # Configura o VideoWriter com o codec especificado e formato AVI
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        if not out.isOpened():
            print("Não foi possível abrir o VideoWriter para gravar o vídeo")
            return False

        print(f"VideoWriter configurado com codec {codec} e arquivo {output_filename}")
        start_time = time.time()
        frame_count = 0

        print(f"Iniciando gravação por {record_time} segundos...")
        print("Pressione 'q' para encerrar a gravação antecipadamente")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Erro ao capturar o frame. Frame count:", frame_count)
                break

            out.write(frame)
            frame_count += 1
            cv2.imshow(window_name, frame)

            # Debug: imprime informações a cada 50 frames
            if frame_count % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Frame {frame_count}: shape = {frame.shape}, tempo decorrido: {elapsed:.2f} segundos")

            if time.time() - start_time >= record_time:
                print("Tempo de gravação atingido")
                break

            # Encerra a gravação quando a tecla 'q' é pressionada
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Gravação interrompida pelo usuário")
                break

        file_size = os.path.getsize(output_filename) if os.path.exists(output_filename) else 0
        print(f"Total de frames gravados: {frame_count}")
        print(f"Gravação finalizada e salva em {output_filename}")
        print(f"Tamanho do arquivo: {file_size} bytes")
        
        success = True
        
    except Exception as e:
        print(f"Erro durante a gravação: {e}")
        
    finally:
        # Garantir que tudo seja liberado, mesmo em caso de erro
        print("Limpando recursos...")
        if cap is not None:
            cap.release()
            print("Câmera liberada")
        if out is not None:
            out.release()
            print("Gravador de vídeo liberado")
        
        # Força a destruição de todas as janelas criadas pelo OpenCV
        print("Fechando todas as janelas...")
        cv2.destroyAllWindows()
        # Aguarde um pouco para garantir que as janelas realmente fechem
        cv2.waitKey(500)
        # Tentativa adicional para forçar o fechamento de janelas
        for i in range(5):
            cv2.waitKey(1)
        
        print("Todos os recursos foram liberados")
        
    return success
