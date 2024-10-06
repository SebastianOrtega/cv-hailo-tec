import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import setproctitle
import cv2

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
# Clase definida por el usuario para ser utilizada en la función de callback
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42
        self.detected_classes = set()

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

    roi = hailo.get_roi_from_buffer(buffer)
    detecciones = roi.get_objects_typed(hailo.HAILO_DETECTION)

    conteo_detecciones = 0
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
        deteccion_auto = (x1, y1, x2 - x1, y2 - y1)
        #x1, y1, x2, y2 = int(bbox.xmin()), int(bbox.ymin()), int(bbox.xmax()), int(bbox.ymax())
        
        confianza = deteccion.get_confidence()
        if confianza > 0.7:
            # Verificar si la etiqueta es nueva
            if etiqueta not in user_data.detected_classes:
                user_data.detected_classes.add(etiqueta)
                print(f"Nueva clase detectada: {etiqueta},  caja: {deteccion_auto},  confianza: {confianza*100:.2f}%")

            if user_data.use_frame:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{etiqueta} {confianza:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            conteo_detecciones += 1
            #print("Detection",conteo_detecciones)

    if user_data.use_frame:
        # Renderizado de estadísticas personalizadas
        texto_fps = f"FPS: {user_data.get_fps()}"
        texto_dropped = f"Frames perdidos: {user_data.get_dropped()}"
        texto_rendered = f"Frames renderizados: {user_data.get_rendered()}"

        cv2.putText(frame, f"Detecciones: {conteo_detecciones}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, texto_fps, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, texto_dropped, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, texto_rendered, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame) 

    return Gst.PadProbeReturn.OK


# -----------------------------------------------------------------------------------------------
# Aplicación GStreamer para Hailo con YOLOv6n
# -----------------------------------------------------------------------------------------------
# ... código anterior ...

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
