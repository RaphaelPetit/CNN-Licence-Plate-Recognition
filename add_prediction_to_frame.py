import cv2
import numpy as np
from datetime import datetime

def add_prediction_to_frame(frame, prediction_data):
    """
    Adiciona as informações de predição ao frame.

    Args:
        frame: Frame original da câmera
        prediction_data: Dicionário com dados da predição retornado pelo frame_handler

    Returns:
        Frame modificado com as informações de predição
    """
    # Criar barra lateral
    height = prediction_data['frame_info']['height']
    width = prediction_data['frame_info']['width']
    sidebar_width = 400 # Restaurar largura original da barra lateral

    # Criar frame com espaço para barra lateral
    new_width = width + sidebar_width
    output_frame = np.zeros((height, new_width, 3), dtype=np.uint8)
    output_frame[0:height, 0:width] = frame

    # Criar barra lateral com gradiente
    sidebar = np.zeros((height, sidebar_width, 3), dtype=np.uint8)
    for i in range(sidebar_width):
        factor = 1 - (i / sidebar_width) * 0.2 # Slightly less intense gradient
        sidebar[:, i] = (40 * factor, 42 * factor, 54 * factor)
    output_frame[0:height, width:new_width] = sidebar

    # Adicionar borda separadora
    cv2.line(output_frame, (width, 0), (width, height), (80, 80, 100), 2)

    # Desenhar boxes de detecção no frame principal
    for box in prediction_data['detected_boxes']:
        x1, y1, x2, y2 = box['coords']
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # --- Layout da Barra Lateral (com espaçamento ajustado) ---
    current_y = 30  # Starting Y position
    line_height_s = 20 # Small line height
    line_height_m = 25 # Medium line height
    line_height_l = 30 # Large line height
    padding = 15 # General padding
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_size_s = 0.6
    text_size_m = 0.7
    text_color_light = (220, 220, 240)
    text_color_white = (255, 255, 255)
    box_color_bg = (50, 50, 60)
    box_color_border = (80, 80, 100)

    # Título
    cv2.putText(output_frame, "Reconhecimento de Placas",
               (width + padding, current_y),
               cv2.FONT_HERSHEY_DUPLEX, 0.9, text_color_white, 1, cv2.LINE_AA)
    current_y += line_height_l + 5

    # Linha divisória
    cv2.line(output_frame, (width + padding, current_y), (new_width - padding, current_y),
             (120, 120, 150), 1, cv2.LINE_AA)
    current_y += line_height_m

    # --- Seção Texto da Placa ---
    cv2.putText(output_frame, "Texto detectado:",
               (width + padding, current_y),
               text_font, text_size_m, text_color_light, 1, cv2.LINE_AA)
    current_y += line_height_m

    # Caixa com texto da placa
    is_plate_detected = prediction_data['plate_text'] != "Sem placa identificada"
    plate_bg_color = (60, 70, 80) if is_plate_detected else (80, 40, 40)
    plate_text_color = (120, 255, 120) if is_plate_detected else text_color_light
    plate_box_height = 45

    cv2.rectangle(output_frame,
                 (width + padding, current_y),
                 (new_width - padding, current_y + plate_box_height),
                 plate_bg_color, -1)
    cv2.rectangle(output_frame,
                 (width + padding, current_y),
                 (new_width - padding, current_y + plate_box_height),
                 (100, 100, 120), 1, cv2.LINE_AA)

    # Ajustar tamanho da fonte da placa
    plate_font_scale = 0.8
    plate_text = prediction_data['plate_text']
    if len(plate_text) > 10: plate_font_scale = 0.7
    if len(plate_text) > 15: plate_font_scale = 0.6

    # Centralizar texto da placa
    plate_text_size = cv2.getTextSize(plate_text, text_font, plate_font_scale, 2)[0]
    plate_text_x = width + (sidebar_width - plate_text_size[0]) // 2
    plate_text_y = current_y + (plate_box_height + plate_text_size[1]) // 2

    # Desenhar texto da placa
    cv2.putText(output_frame, plate_text,
               (plate_text_x + 1, plate_text_y + 1),
               text_font, plate_font_scale, (20, 20, 20), 2, cv2.LINE_AA) # Shadow
    cv2.putText(output_frame, plate_text,
               (plate_text_x, plate_text_y),
               text_font, plate_font_scale, plate_text_color, 2, cv2.LINE_AA)
    current_y += plate_box_height + line_height_m

    # --- Seção Confiança ---
    confidence = prediction_data['confidence']
    if confidence >= 0: # Show even if 0
        cv2.putText(output_frame, f"Confianca: {confidence:.2f}",
                   (width + padding, current_y),
                   text_font, text_size_m, text_color_light, 1, cv2.LINE_AA)
        current_y += line_height_m

        bar_width_max = sidebar_width - (2 * padding)
        bar_width = int(bar_width_max * confidence)
        bar_height = 12

        # Background bar
        cv2.rectangle(output_frame,
                     (width + padding, current_y),
                     (width + padding + bar_width_max, current_y + bar_height),
                     (60, 60, 70), -1)

        # Confidence bar
        if bar_width > 0:
            bar_color = (100, 255, 100) if confidence > 0.7 else \
                       (100, 255, 255) if confidence > 0.4 else \
                       (100, 100, 255)
            cv2.rectangle(output_frame,
                         (width + padding, current_y),
                         (width + padding + bar_width, current_y + bar_height),
                         bar_color, -1)

        # Border
        cv2.rectangle(output_frame,
                     (width + padding, current_y),
                     (width + padding + bar_width_max, current_y + bar_height),
                     (120, 120, 150), 1, cv2.LINE_AA)
        current_y += bar_height + line_height_m

    # --- Seção Imagem da Placa ---
    plate_image = prediction_data.get('cropped_plate')
    if prediction_data.get('found_plate') and plate_image is not None:
        try:
            cv2.putText(output_frame, "Placa Capturada:",
                       (width + padding, current_y),
                       text_font, text_size_m, text_color_light, 1, cv2.LINE_AA)
            current_y += line_height_m

            # Calculate display size for the plate image
            max_plate_width = sidebar_width - (2 * padding)
            max_plate_height = 60 # Slightly smaller max height
            plate_h, plate_w = plate_image.shape[:2]

            if plate_h > 0 and plate_w > 0: # Check dimensions are valid
                # Resize while maintaining aspect ratio
                scale = min(max_plate_width / plate_w, max_plate_height / plate_h)
                display_width = int(plate_w * scale)
                display_height = int(plate_h * scale)

                if display_width > 0 and display_height > 0:
                    resized_plate = cv2.resize(
                        plate_image,
                        (display_width, display_height),
                        interpolation=cv2.INTER_AREA
                    )

                    # Calculate position to center the plate image
                    plate_preview_x = width + (sidebar_width - display_width) // 2
                    plate_preview_y = current_y

                    # Check if it fits vertically before drawing
                    if plate_preview_y + display_height < height - 80: # Ensure space for stats & footer
                        # Draw background rectangle
                        cv2.rectangle(output_frame,
                                     (plate_preview_x - 5, plate_preview_y - 5),
                                     (plate_preview_x + display_width + 5, plate_preview_y + display_height + 5),
                                     (70, 70, 90), -1)

                        # Place the resized plate image
                        output_frame[plate_preview_y : plate_preview_y + display_height,
                                    plate_preview_x : plate_preview_x + display_width] = resized_plate

                        # Draw border around the plate image
                        cv2.rectangle(output_frame,
                                     (plate_preview_x - 2, plate_preview_y - 2),
                                     (plate_preview_x + display_width + 2, plate_preview_y + display_height + 2),
                                     text_color_white, 1, cv2.LINE_AA)

                        current_y += display_height + line_height_m # Update Y position after drawing
                    else:
                       # Not enough space, skip drawing plate image but still increment Y slightly
                       current_y += line_height_s
                else:
                    # Invalid display dimensions, increment Y slightly
                    current_y += line_height_s
            else:
                 # Invalid plate dimensions, increment Y slightly
                current_y += line_height_s
        except Exception as e:
            print(f"Erro ao exibir imagem da placa: {e}")
            current_y += line_height_m # Increment Y even if error occurs

    # --- Seção Estatísticas --- (Restored and integrated)
    stats_box_height = 75 # Increased height for more info
    stats_y = current_y # Start stats after plate image section

    # Check if stats box fits
    if stats_y + stats_box_height < height - 40: # Leave space for footer
        # Draw stats box background and border
        cv2.rectangle(output_frame,
                     (width + padding, stats_y),
                     (new_width - padding, stats_y + stats_box_height),
                     box_color_bg, -1)
        cv2.rectangle(output_frame,
                     (width + padding, stats_y),
                     (new_width - padding, stats_y + stats_box_height),
                     box_color_border, 1, cv2.LINE_AA)

        stats_text_y = stats_y + line_height_s # Start text inside the box

        # Device Info
        device_txt = f"Dispositivo: {prediction_data['device_info']['type'].upper()}"
        cv2.putText(output_frame, device_txt,
                   (width + padding + 5, stats_text_y),
                   text_font, text_size_s, text_color_light, 1, cv2.LINE_AA)
        stats_text_y += line_height_s

        # Detection Time
        det_time = prediction_data['execution_times']['detection']
        det_txt = f"Identificacao: {det_time:.3f}s"
        cv2.putText(output_frame, det_txt,
                   (width + padding + 5, stats_text_y),
                   text_font, text_size_s, text_color_light, 1, cv2.LINE_AA)
        stats_text_y += line_height_s

        # OCR Time
        ocr_time = prediction_data['execution_times']['ocr']
        ocr_txt = f"OCR: {ocr_time:.3f}s"
        cv2.putText(output_frame, ocr_txt,
                   (width + padding + 5, stats_text_y),
                   text_font, text_size_s, text_color_light, 1, cv2.LINE_AA)
        # stats_text_y += line_height_s # REMOVE increment here

        # Detected Objects Count (align to the right on the same line as OCR)
        det_count = len(prediction_data['detected_boxes'])
        count_txt = f"Deteccoes: {det_count}"
        (w_count, h_count), _ = cv2.getTextSize(count_txt, text_font, text_size_s, 1)
        count_x = new_width - padding - w_count - 5 # Calculate right-aligned X
        cv2.putText(output_frame, count_txt,
                   (count_x, stats_text_y), # Use the SAME Y as OCR text
                   text_font, text_size_s, text_color_light, 1, cv2.LINE_AA)

        # NOW increment Y after drawing both texts on the same line
        stats_text_y += line_height_s

        current_y = stats_y + stats_box_height + padding # Update Y below stats box
    else:
        # Not enough space for stats box, just update Y
        current_y += line_height_s

    # --- Footer --- (Positioned near bottom, dynamically)
    footer_y = height - 20 # Position 20px from bottom
    current_time = datetime.now().strftime("%H:%M:%S")
    footer_text = f"Hora: {current_time}"
    cv2.putText(output_frame, footer_text,
               (width + padding, footer_y),
               text_font, text_size_s, text_color_light, 1, cv2.LINE_AA)

    # Add quit message to footer
    q_text = "| Pressione 'q' para sair"
    (w_f, h_f), _ = cv2.getTextSize(footer_text, text_font, text_size_s, 1)
    q_pos_x = width + padding + w_f + 10
    cv2.putText(output_frame, q_text,
               (q_pos_x, footer_y),
               text_font, text_size_s, text_color_light, 1, cv2.LINE_AA)

    return output_frame
