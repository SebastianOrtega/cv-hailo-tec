from flask import Flask, request, jsonify

app = Flask(__name__)

# Variable global para mantener el conteo de lecturas
contador_lecturas = 0

@app.route('/', methods=['POST'])
def recibir_json():
    global contador_lecturas  # Declarar que usaremos la variable global
    if request.is_json:
        datos = request.get_json()
        
        # Verificar si los datos son una lista
        if isinstance(datos, list):
            for item in datos:
                contador_lecturas += 1  # Incrementar el contador por cada lectura
                print(f"--- Lectura {contador_lecturas} ---")
                for key, value in item.items():
                    print(f"{key}: {value}")
                print("\n")  # Línea en blanco para separar lecturas
        else:
            contador_lecturas += 1  # Incrementar el contador por cada lectura
            print(f"--- Lectura {contador_lecturas} ---")
            for key, value in datos.items():
                print(f"{key}: {value}")
            print("\n")  # Línea en blanco después de la lectura
        
        respuesta = {
            "mensaje": "JSON recibido y procesado correctamente",
            "datos": datos
        }
        return jsonify(respuesta), 200
    else:
        return jsonify({"error": "El cuerpo de la solicitud debe ser JSON"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
