import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import setproctitle
import cv2
from datetime import datetime
import numpy as np

import hailo
from hailo_rpi_common import (
    get_default_parser,
    QUEUE,
    get_caps_from_pad,
    get_numpy_from_buffer,
    GStreamerApp,
    app_callback_class,
)

# -----------------------------------------------------------------------------------------------
# Clase para rastreo de objetos
# -----------------------------------------------------------------------------------------------

class TrackedObject:
    def __init__(self, object_id, label, initial_center):
        self.object_id = object_id
        self.label = label
        self.initial_center = initial_center
        self.current_center = initial_center
        self.history = [initial_center]  # Historial de posiciones
        self.frames_since_seen = 0  # Contador de frames sin ser visto
        self.lost_frames = 0  # Contador de frames desde que fue perdido

    def update(self, new_center):
        self.current_center = new_center
        self.history.append(new_center)
        self.frames_since_seen = 0  # Reiniciar contador
        self.lost_frames = 0  # Reiniciar contador de perdido

    def increment_frames_since_seen(self):
        self.frames_since_seen += 1

    def get_distance(self):
        x1, y1 = self.initial_center
        x2, y2 = self.current_center
        return ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

    def get_slope(self):
        x1, y1 = self.initial_center
        x2, y2 = self.current_center
        if x2 - x1 == 0:
            return float('inf')  # Evitar división por cero
        return (y2 - y1) / (x2 - x1)


# -----------------------------------------------------------------------------------------------
# Clase definida por el usuario para ser utilizada en la función de callback
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42
        self.detected_classes = set()
        self.tracked_objects = {}  # Diccionario de objetos rastreados
        self.next_object_id = 1    # Para asignar IDs únicos
        self.distance_threshold = 50  # Umbral de distancia en píxeles
        self.frames_since_seen_threshold = 100  # Umbral para considerar un objeto como perdido
        self.lost_objects = {}  # Diccionario de objetos perdidos
        self.lost_objects_max_age = 150  # Máximo de frames para mantener objetos perdidos
        self.first_frame_saved = False  # Bandera para controlar si el primer frame ha sido guardado

    def new_function(self):
        return "El significado de la vida es: "



# -----------------------------------------------------------------------------------------------
# Función de callback para detección de objetos
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        print("No hay buffer")
        return Gst.PadProbeReturn.OK

    user_data.increment()

    formato, ancho, alto = get_caps_from_pad(pad)

    frame = None
    if user_data.use_frame and formato is not None and ancho is not None and alto is not None:
        frame = get_numpy_from_buffer(buffer, formato, ancho, alto)

        # Guardar el primer frame en un PNG
        if not user_data.first_frame_saved:
            # Convertir el frame de RGB a BGR para OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite('primer_frame.png', frame_bgr)
            user_data.first_frame_saved = True
            print("Primer frame guardado como 'primer_frame.png'")

    """# Redimensionar el frame a 640x360
        frame_resized = cv2.resize(frame, (640, 360))

        # Crear una imagen negra de 640x640
        frame_black = np.zeros((640, 640, 3), dtype=np.uint8)

        # Calcular el desplazamiento vertical para centrar el frame
        y_offset = (640 - 360) // 2  # (tamaño total - altura del frame redimensionado) / 2

        # Colocar el frame redimensionado en el centro de la imagen negra
        frame_black[y_offset:y_offset + 360, 0:640] = frame_resized

        # Reemplazar 'frame' por la imagen negra con el frame centrado
        frame = frame_black"""

    roi = hailo.get_roi_from_buffer(buffer)
    detecciones = roi.get_objects_typed(hailo.HAILO_DETECTION)

    detecciones_actuales = []
    etiquetas_interes = ['car', 'truck', 'bus', 'train']  # Etiquetas a tratar como 'car'

    for deteccion in detecciones:
        etiqueta = deteccion.get_label()
        bbox = deteccion.get_bbox()

        x1_norm = bbox.xmin()
        y1_norm = bbox.ymin()
        x2_norm = bbox.xmax()
        y2_norm = bbox.ymax()
        x1 = int(x1_norm * ancho)
        y1 = int(y1_norm * alto)
        x2 = int(x2_norm * ancho)
        y2 = int(y2_norm * alto)

        confianza = deteccion.get_confidence()
        if confianza > 0.7:
            if etiqueta in etiquetas_interes:
                etiqueta_original = etiqueta  # Guardar la etiqueta original
                etiqueta = 'car'  # Convertir a 'car'
                # Calcular el centro del bounding box
                centro_actual = ((x1 + x2) // 2, (y1 + y2) // 2)
                detecciones_actuales.append((etiqueta, etiqueta_original, centro_actual, (x1, y1, x2, y2), confianza))
            else:
                # Ignorar otras etiquetas o procesarlas de otra manera
                pass

    if not detecciones_actuales:
        # No hay detecciones de interés en este frame
        return Gst.PadProbeReturn.OK

    # Lista para IDs de objetos actualizados en esta iteración
    ids_actualizados = set()

    # Procesar las detecciones actuales
    for etiqueta, etiqueta_original, centro_actual, bbox_coords, confianza in detecciones_actuales:
        objeto_emparejado = None
        distancia_minima = float('inf')
        is_lost_object = False  # Indica si el objeto emparejado es de lost_objects

        # Emparejar con objetos rastreados
        for obj_id, objeto_rastreado in user_data.tracked_objects.items():
            if etiqueta == objeto_rastreado.label:
                # Calcular distancia euclidiana entre centros
                dist = ((centro_actual[0] - objeto_rastreado.current_center[0])**2 +
                        (centro_actual[1] - objeto_rastreado.current_center[1])**2) ** 0.5
                if dist < user_data.distance_threshold and dist < distancia_minima:
                    distancia_minima = dist
                    objeto_emparejado = objeto_rastreado

        # Si no se encontró en objetos rastreados, buscar en objetos perdidos
        if objeto_emparejado is None:
            for obj_id, lost_object in user_data.lost_objects.items():
                if etiqueta == lost_object.label:
                    dist = ((centro_actual[0] - lost_object.current_center[0])**2 +
                            (centro_actual[1] - lost_object.current_center[1])**2) ** 0.5
                    if dist < user_data.distance_threshold and dist < distancia_minima:
                        distancia_minima = dist
                        objeto_emparejado = lost_object
                        is_lost_object = True
                        matched_lost_object_id = obj_id

        if objeto_emparejado:
            if is_lost_object:
                # Remover de lost_objects y agregar de nuevo a tracked_objects
                user_data.lost_objects.pop(matched_lost_object_id)
                user_data.tracked_objects[matched_lost_object_id] = objeto_emparejado
                print(f"Objeto ID {matched_lost_object_id} recuperado.")
            # Actualizar objeto rastreado existente
            objeto_emparejado.update(centro_actual)
            ids_actualizados.add(objeto_emparejado.object_id)

            # Calcular distancia y pendiente
            distancia_total = objeto_emparejado.get_distance()
            pendiente = objeto_emparejado.get_slope()

            print(f"ID: {objeto_emparejado.object_id}, Clase: {etiqueta_original}, "
                f"Distancia: {distancia_total:.2f}, Pendiente: {pendiente:.2f}, "
                f"Centro inicial: {objeto_emparejado.initial_center}, "
                f"Centro actual: {objeto_emparejado.current_center}")


            # Dibujar en el frame
            if user_data.use_frame:
                x1, y1, x2, y2 = bbox_coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {objeto_emparejado.object_id} {etiqueta_original} {confianza:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            # Crear nuevo objeto rastreado
            object_id = user_data.next_object_id
            user_data.next_object_id += 1

            nuevo_objeto = TrackedObject(object_id, etiqueta, centro_actual)
            user_data.tracked_objects[object_id] = nuevo_objeto
            ids_actualizados.add(object_id)

            print(f"Nuevo objeto detectado: ID {object_id}, Clase: {etiqueta_original}")

            # Dibujar en el frame
            if user_data.use_frame:
                x1, y1, x2, y2 = bbox_coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {object_id} {etiqueta_original} {confianza:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Incrementar contador de frames sin ver para objetos no actualizados
    ids_para_eliminar = []
    for obj_id, objeto_rastreado in user_data.tracked_objects.items():
        if obj_id not in ids_actualizados:
            objeto_rastreado.increment_frames_since_seen()
            if objeto_rastreado.frames_since_seen > user_data.frames_since_seen_threshold:
                ids_para_eliminar.append(obj_id)

    # Mover objetos perdidos a lost_objects
    for obj_id in ids_para_eliminar:
        lost_object = user_data.tracked_objects.pop(obj_id)
        lost_object.lost_frames = 0  # Inicializar contador
        user_data.lost_objects[obj_id] = lost_object
        print(f"Objeto ID {obj_id} perdido temporalmente.")

    # Incrementar contador de lost_frames y eliminar objetos perdidos antiguos
    ids_to_remove_from_lost = []
    for obj_id, lost_object in user_data.lost_objects.items():
        lost_object.lost_frames += 1
        if lost_object.lost_frames > user_data.lost_objects_max_age:
            ids_to_remove_from_lost.append(obj_id)

    for obj_id in ids_to_remove_from_lost:
        del user_data.lost_objects[obj_id]
        print(f"Objeto ID {obj_id} eliminado permanentemente de objetos perdidos.")

    if user_data.use_frame:
        # Mostrar estadísticas personalizadas
        cv2.putText(frame, f"Detecciones: {len(detecciones_actuales)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Puedes añadir más información si lo deseas

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    return Gst.PadProbeReturn.OK


# -----------------------------------------------------------------------------------------------
# Aplicación GStreamer para Hailo con YOLOv6n
# -----------------------------------------------------------------------------------------------

class GStreamerDetectionApp(GStreamerApp):
    def __init__(self, args, user_data):
        super().__init__(args, user_data)
        self.args = args  # Almacena los argumentos
        self.batch_size = 2
        self.network_width = 640
        self.network_height = 640
        self.network_format = "RGB"
        nms_score_threshold = 0.7
        nms_iou_threshold = 0.45

        self.default_postprocess_so = os.path.join(self.current_path, '../resources/libyolo_hailortpp_post.so')

        if args.hef_path is not None:
            self.hef_path = args.hef_path
        else:
            self.hef_path = os.path.join(self.current_path, '../resources/yolov6n.hef')

        self.app_callback = app_callback

        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        setproctitle.setproctitle("Aplicación de Detección Hailo")
        self.create_pipeline()

    def get_pipeline_string(self):
        if self.args.video_file is not None:
            source_element = (
                f"filesrc location={self.args.video_file} ! decodebin ! videoconvert ! "
                "videoscale ! video/x-raw, width=640, height=640, format=RGB ! "
            )
        else:
            source_element = (
                f"rtspsrc location=rtsp://admin:panamet0@192.168.0.84:554/Streaming/Channels/102 ! "
                "rtph264depay ! avdec_h264 ! videoconvert ! "
                "videoscale ! video/x-raw, width=640, height=640, format=RGB ! "
            )

        pipeline_string = (
            source_element
            + f"hailonet hef-path={self.hef_path} batch-size={self.batch_size} {self.thresholds_str} ! "
            + f"hailofilter so-path={self.default_postprocess_so} ! "
            + "identity name=identity_callback ! "
            + "hailooverlay ! videoconvert ! "
            + "fpsdisplaysink name=hailo_display video-sink=xvimagesink sync=false"
        )
        print(pipeline_string)
        return pipeline_string

if __name__ == "__main__":
    user_data = user_app_callback_class()
    parser = get_default_parser()
    parser.add_argument("--network", default="yolov6n", choices=['yolov6n'], help="Red a utilizar")
    parser.add_argument("--hef-path", default=None, help="Ruta al archivo HEF")
    parser.add_argument("--video-file", help="Ruta al archivo de video para usar como entrada")  # Nuevo argumento
    args = parser.parse_args()
    app = GStreamerDetectionApp(args, user_data)
    app.run()
