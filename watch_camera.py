import cv2
import time
from add_prediction_to_frame import add_prediction_to_frame

def watch_camera(
        source,
        window_name="Camera",
        use_gstreamer=False,
        rtsp_user=None,
        rtsp_password=None,
        frame_handler=None,
        frame_handler_period=0.1
):
    """
    Inicia a captura de vídeo a partir de uma fonte RTSP e exibe em uma janela.

    Parâmetros:
      - source: URL RTSP sem credenciais embutidas (ex.: "rtsp://192.168.2.137:554/axis-media/media.amp")
      - window_name: nome da janela onde o vídeo será exibido (padrão: "Camera")
      - use_gstreamer: se True, utiliza uma pipeline GStreamer para capturar o stream
      - rtsp_user: (opcional) usuário de autenticação RTSP
      - rtsp_password: (opcional) senha de autenticação RTSP
      - frame_handler: (opcional) função para processar os frames
      - frame_handler_period: (opcional) período em segundos para chamar o frame_handler

    A função captura e exibe os frames da fonte especificada. Para encerrar, pressione a tecla 'q'.
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

    print(f"Conectando à câmera: {source}")

    cap = None
    success = False

    # Variáveis para medição de tempo
    frame_count = 0
    handler_count = 0

    # Variáveis para armazenar a última predição
    last_prediction_data = None
    last_prediction_time = 0

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

        # Exibe informações sobre o stream
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Conexão estabelecida! Resolução: {width:.0f}x{height:.0f}, FPS: {fps:.2f}")

        # Tenta capturar o primeiro frame para confirmar
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar o primeiro frame")
            return False

        print("Exibindo stream (pressione 'q' para sair)...")

        # Configurar a janela para exibição
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        has_frame_handler = frame_handler is not None

        # Se temos um frame handler, processamos o primeiro frame imediatamente
        # para configurar a janela com o tamanho correto
        if has_frame_handler:
            initial_prediction = frame_handler(frame)
            if initial_prediction is not None:
                # Obter as dimensões do frame processado (que inclui a barra lateral)
                processed_frame = add_prediction_to_frame(frame, initial_prediction)
                h, w = processed_frame.shape[:2]
                # Redimensionar a janela para acomodar o frame processado
                cv2.resizeWindow(window_name, w, h)
                last_prediction_data = initial_prediction
                last_prediction_time = time.time()

        # Posicionar a janela
        cv2.moveWindow(window_name, 50, 50)

        frame_handler_time_sum = 0
        frame_handler_count = 0

        loop_time_sum = 0
        loop_count = 0

        while cap.isOpened():
            # Iniciar medição do tempo do loop
            loop_start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Erro ao capturar frame")
                break

            display_frame = frame.copy()  # Vamos usar uma cópia para exibição

            # Se houver um frame_handler e o tempo certo passou
            if has_frame_handler:
                current_time = time.time()
                elapsed_time = current_time - last_prediction_time

                if elapsed_time >= frame_handler_period:
                    # Iniciar medição do tempo do frame_handler
                    handler_start_time = time.time()

                    # Processa o frame e pega o resultado
                    prediction_data = frame_handler(frame)

                    # Calcular o tempo do frame_handler
                    handler_time = time.time() - handler_start_time
                    frame_handler_time_sum += handler_time
                    frame_handler_count += 1
                    handler_count += 1

                    if prediction_data is not None:
                        last_prediction_data = prediction_data
                        last_prediction_time = current_time  # Atualizar o tempo da última predição

                    # Calcular o tempo total do loop
                    loop_time = time.time() - loop_start_time
                    loop_time_sum += loop_time
                    loop_count += 1

                # Atualizar o frame com a última predição
                if last_prediction_data is not None:
                    display_frame = add_prediction_to_frame(frame, last_prediction_data)

            # Exibe o frame (original ou processado)
            cv2.imshow(window_name, display_frame)
            frame_count += 1

            # Encerra a exibição quando a tecla 'q' é pressionada
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Usuário encerrou a exibição")
                break

        print(f"Total de frames exibidos: {frame_count}")
        print(f"Total de frames processados pelo handler: {handler_count}")

        # Mostrar estatísticas finais
        if frame_handler_time_sum > 0:
            avg_handler = frame_handler_time_sum / frame_handler_count
            avg_loop = loop_time_sum / loop_count
            print("\n===== ESTATÍSTICAS FINAIS =====")
            print(f"Tempo médio do frame_handler: {avg_handler:.3f}s")
            print(f"Tempo médio do loop: {avg_loop:.3f}s")
            print(f"FPS efetivo médio: {1.0 / avg_loop:.1f}")
            print("==============================\n")

        success = True

    except Exception as e:
        print(f"Erro durante a exibição: {e}")
    finally:
        # Garantir que tudo seja liberado, mesmo em caso de erro
        print("Limpando recursos...")
        if cap is not None:
            cap.release()
            print("Câmera liberada")

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

