import numpy as np
import cv2
import pandas as pd

cap = cv2.VideoCapture('recording_20240928_133154.mp4')
frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = int(width)
height = int(height)
print(frames_count, fps, width, height)
# Creamos un datas frame de pandas con el mismo numero de filas que el count de los frames
df = pd.DataFrame(index=range(int(frames_count)))
df.index.name = "Frames"

framenumber = 0  # Conteo de los frames
carscrossedup = 0  # Conteo de entradas
carscrosseddown = 0  # Conteo de salidas
carids = []  # Lista en blanco con los id de los vehiculos
caridscrossed = []  # Lista en blanco de los vehiculos que cruzaron
totalcars = 0  # Contador total de vehiculos

fgbg = cv2.createBackgroundSubtractorMOG2()  # Creamos la sustraccion de fondo

# Informacion para comenzar a guardar la informacion de video 
ret, frame = cap.read()  # Importamos la imagen
ratio = 1  # resize ratio
image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
width2, height2, channels = image.shape
video = cv2.VideoWriter('Resultados.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (height2, width2), 1)

while True:

    ret, frame = cap.read()  # Importamos la imagen

    if ret:  # if hay un frame continuamos con el codigo
        image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertimos la imagen en blanco y negro
        fgmask = fgbg.apply(gray)  # Usamos la sustraccion de fondo

        # Aplica diferentes umbrales a fgmask para intentar aislar los vehiculos
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel para aplicar transformaciones morfologicas
        closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        dilation = cv2.dilate(opening, kernel)
        retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # Eliminar sombras
        # Crear contornos
        contours, hierarchy = cv2.findContours(bins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Usamos convex hull para crear un poligono alrededor de los contornos
        hull = [cv2.convexHull(c) for c in contours]
        # Dibujamos los contornos
        cv2.drawContours(image, hull, -1, (0, 255, 0), 2)
        # Linea superior para no asignar id si el objeto esta antes de esta
        lineypos = 80
        cv2.line(image, (0, lineypos), (width, lineypos), (255, 0, 0), 1)
        # Linea inferior para contar
        lineypos2 = 190
        cv2.line(image, (0, lineypos2), (width, lineypos2), (0, 255, 0), 1)
        # Area minima del contorno para el conteo
        minarea = 5000
        # Area maxima para el conteo
        maxarea = 50000

        # Vectores X y Y de la ubicacion de los contornos y centroides en el frame actual
        cxx = np.zeros(len(contours))
        cyy = np.zeros(len(contours))

        for i in range(len(contours)):  # Ciclos para todos los contornos en el frame actual

            if hierarchy[0, i, 3] == -1:  # Usar la jerarquia para contar solo los contornos padre

                area = cv2.contourArea(contours[i])  # area del contorno

                if minarea < area < maxarea:  # Umbral de area para el contorno
                    # Calculando los centroides de los contornos
                    cnt = contours[i]
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    if cy > lineypos:  # Filtrar contornos que estan arriba de la linea (Y comienza en la parte superior)

                        # Obtiene los ountos delimitadores del contorno para crear un rectangulo
                        # X, Y es la esquina superior derecha y W, H es ancho y alto
                        x, y, w, h = cv2.boundingRect(cnt)

                        # Crea los rectangulos alrededor del contorno
                        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                        # Imprime el texto del centroide para volver a compararlo despues
                        cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    .3, (0, 0, 255), 1)

                        cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1,
                                       line_type=cv2.LINE_AA)

                        # Agrega los centroides que pasaron los ctiterios anteriores a la lista de centroides
                        cxx[i] = cx
                        cyy[i] = cy

        # Eliminamos los centroides que no pasaron las validaciones
        cxx = cxx[cxx != 0]
        cyy = cyy[cyy != 0]

        # Lista vacia para comprobar los indices de centroide que se agregaron al dataframe
        minx_index2 = []
        miny_index2 = []

        # Radio maximo permitido para considerar el mismo centroide entre frames
        maxrad = 25

        # En la siguiente seccion realiza un seguimiento de los centroides y los asigna a los antiguos id o  a los nuevos

        if len(cxx):  # Si hay centroides en el área especificada

            if not carids:  # Si carids esta vacio

                for i in range(len(cxx)):  # Recorre todos los centroides

                    carids.append(i)  # Agrega un ID de coche a la lista vacía carids
                    df[str(carids[i])] = ""  # Agrega una columna al dataframe correspondiente a un carid

                    # Asigna los valores de centroide al frame actual (fila) y al carid (columna) 
                    df.at[int(framenumber), str(carids[i])] = [cxx[i], cyy[i]]

                    totalcars = carids[i] + 1  # Agrega un conteo al total de autos

            else:  # Si carids ya existe

                dx = np.zeros((len(cxx), len(carids)))  # Nuevos arreglos para calcular deltas
                dy = np.zeros((len(cyy), len(carids)))  # Nuevos arreglos para calcular deltas

                for i in range(len(cxx)):  # loops por todos los centroides

                    for j in range(len(carids)):  # loops por todos los carids

                        # Adquiere el centroide del fotograma anterior para un ID de vehículo específico
                        oldcxcy = df.iloc[int(framenumber - 1)][str(carids[j])]

                        # Adquiere el centroide del fotograma actual que no necesariamente coincide con el centroide del fotograma anterior
                        curcxcy = np.array([cxx[i], cyy[i]])

                        if not oldcxcy:  # Verifica si el centroide antiguo está vacío en caso de que un automóvil salga de la pantalla y aparezca un nuevo automóvi

                            continue  # Continua al siguiente carid

                        else:  # Calcular las diferencias de los centroides para comparar más tarde con la posición del fotograma actual

                            dx[i, j] = oldcxcy[0] - curcxcy[0]
                            dy[i, j] = oldcxcy[1] - curcxcy[1]

                for j in range(len(carids)):  # Recorre todos los IDs de los vehículos actuales

                    sumsum = np.abs(dx[:, j]) + np.abs(dy[:, j])  # Suma los deltas con respecto a los IDs de los vehículos

                    # Encuentra qué índice tuvo el carid con la diferencia mínima y este es el índice verdadero
                    correctindextrue = np.argmin(np.abs(sumsum))
                    minx_index = correctindextrue
                    miny_index = correctindextrue

                    # Aquiere los valores delta de los deltas mínimos para verificar si están dentro del radio más adelante
                    mindx = dx[minx_index, j]
                    mindy = dy[miny_index, j]

                    if mindx == 0 and mindy == 0 and np.all(dx[:, j] == 0) and np.all(dy[:, j] == 0):
                        # Verifica si el valor mínimo es 0 y comprueba si todos los deltas son cero, ya que este es un conjunto vacío
                        # Delta podría ser cero si el centróide no se movió.

                        continue  # Continua al siguiente carid

                    else:

                        # Si los valores delta son menores que el radio máximo, entonces añade ese centróide al carid específico
                        if np.abs(mindx) < maxrad and np.abs(mindy) < maxrad:

                            # Agrega el centróide al carid existente correspondiente
                            df.at[int(framenumber), str(carids[j])] = [cxx[minx_index], cyy[miny_index]]
                            minx_index2.append(minx_index)  # appends all the indices that were added to previous carids
                            miny_index2.append(miny_index)

                for i in range(len(cxx)):  # Recorre todos los centróides

                    # Si el centróide no está en la lista de minindex, entonces se necesita agregar otro coche
                    if i not in minx_index2 and miny_index2:

                        df[str(totalcars)] = ""  # Crear otra columna con el total de coches
                        totalcars = totalcars + 1  # Añade otro coche al total del conteo
                        t = totalcars - 1  # t es un marcador de posición para el total de coches
                        carids.append(t)  # Agregar a la lista de identificadores de coches
                        df.at[int(framenumber), str(t)] = [cxx[i], cyy[i]]  # Agregar el centroide al nuevo identificador de coche

                    elif curcxcy[0] and not oldcxcy and not minx_index2 and not miny_index2:
                        # Verifica si el centroide actual existe, pero el centroide anterior no
                        # Nuevo coche para ser agregado en caso de que minx_index2 esté vacío

                        df[str(totalcars)] = ""  # Crear otra columna con el total de coches
                        totalcars = totalcars + 1  # Añade otro coche al total del conteo
                        t = totalcars - 1  # t es un marcador de posición para el total de coches
                        carids.append(t)  # Agregar a la lista de identificadores de coches
                        df.at[int(framenumber), str(t)] = [cxx[i], cyy[i]]  # Agregar el centroide al nuevo identificador de coche

        # La sección a continuación etiqueta los centroides en la pantalla

        currentcars = 0  # Autos actuales en pantalla
        currentcarsindex = []  # Autos actuales en pantalla index

        for i in range(len(carids)):  # Itera a través de todos los carids

            if df.at[int(framenumber), str(carids[i])] != '':
                # Verifica el fotograma actual para ver qué car IDs están activos
                # verificando si el centroid existe en el fotograma actual para un cierto car ID

                currentcars = currentcars + 1 
                currentcarsindex.append(i)  # Agrega los IDs de los vehículos a los vehículos actuales en pantalla

        for i in range(currentcars):  # Itera a través de todos los IDs de vehículos actuales en pantalla

            # Obtiene el centroide de un cierto ID de vehículo para el cuadro actual
            curcent = df.iloc[int(framenumber)][str(carids[currentcarsindex[i]])]

            # Obtiene el centroide de un cierto ID de vehículo para el cuadro anterior
            oldcent = df.iloc[int(framenumber - 1)][str(carids[currentcarsindex[i]])]

            if curcent:  # Si hay un centroide actual

                # Texto en pantalla para el centroide actual
                cv2.putText(image, "Centroid" + str(curcent[0]) + "," + str(curcent[1]),
                            (int(curcent[0]), int(curcent[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                cv2.putText(image, "ID:" + str(carids[currentcarsindex[i]]), (int(curcent[0]), int(curcent[1] - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                cv2.drawMarker(image, (int(curcent[0]), int(curcent[1])), (0, 0, 255), cv2.MARKER_STAR, markerSize=5,
                               thickness=1, line_type=cv2.LINE_AA)

                if oldcent:  # Verifica si el centroide antiguo existe
                    # Agrega un cuadro de radio desde el centroide anterior al centroide actual para visualización
                    xstart = oldcent[0] - maxrad
                    ystart = oldcent[1] - maxrad
                    xwidth = oldcent[0] + maxrad
                    yheight = oldcent[1] + maxrad
                    cv2.rectangle(image, (int(xstart), int(ystart)), (int(xwidth), int(yheight)), (0, 125, 0), 1)

                    # Verifica si el centroide antiguo está en o por debajo de la línea y el centroide actual está en o por encima de la línea
                    # para contar los coches y coches que no han sido contados aún
                    if oldcent[1] >= lineypos2 and curcent[1] <= lineypos2 and carids[
                        currentcarsindex[i]] not in caridscrossed:

                        carscrossedup = carscrossedup + 1
                        cv2.line(image, (0, lineypos2), (width, lineypos2), (0, 0, 255), 5)
                        caridscrossed.append(
                            currentcarsindex[i])  # Agrega el ID del coche a la lista de coches contados para evitar el conteo duplicado

                    # Verifica si el antiguo centroide está en o por encima de la línea y el centroide actual está en o por debajo de la línea
                    # para contar los coches y coches que no han sido contados aún
                    elif oldcent[1] <= lineypos2 and curcent[1] >= lineypos2 and carids[
                        currentcarsindex[i]] not in caridscrossed:

                        carscrosseddown = carscrosseddown + 1
                        cv2.line(image, (0, lineypos2), (width, lineypos2), (0, 0, 125), 5)
                        caridscrossed.append(currentcarsindex[i])

        # Contador de la parte superior izquierda
        cv2.rectangle(image, (0, 0), (180, 50), (0, 0, 0), -1)  # background 

        cv2.putText(image, "Vehiculos en area: " + str(currentcars), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

        cv2.putText(image, "Entradas: " + str(carscrossedup), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

        cv2.putText(image, "Salidas: " + str(carscrosseddown), (0, 45), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

        # Mostramos las imagenes y las transformaciones
        cv2.imshow("countours", image)
        cv2.moveWindow("countours", 0, 0)

        cv2.imshow("fgmask", fgmask)
        cv2.moveWindow("fgmask", int(width * ratio), 0)

        cv2.imshow("closing", closing)
        cv2.moveWindow("closing", width, 0)

        cv2.imshow("opening", opening)
        cv2.moveWindow("opening", 0, int(height * ratio))

        cv2.imshow("dilation", dilation)
        cv2.moveWindow("dilation", int(width * ratio), int(height * ratio))

        cv2.imshow("binary", bins)
        cv2.moveWindow("binary", width, int(height * ratio))

        video.write(image)  # guardamos la iamagen actual al video

        # Sumamos uno al contador de frames
        framenumber = framenumber + 1

        k = cv2.waitKey(int(10/fps)) & 0xff  # int(1000/fps) seria la velocidad normal por lo que waitkey esta en ms
        if k == 27:
            break

    else:  # Si el video termino termina el loop

        break
cap.release()
cv2.destroyAllWindows()
# Guardamos el dataframe de pandas para analisis y deteccion de errores
df.to_csv('Resultados.csv', sep=',')
