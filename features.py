#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import numpy as np
import argparse
import cv2
import os
from primesense import openni2
from collections import Counter


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    gp = p.add_mutually_exclusive_group(required=True)
    gp.add_argument('-f', action='store_false', dest='cls', help='extract features from .oni video')
    gp.add_argument('-k', action='store_true', dest='cls', help='classifies people in .oni video')
    p.add_argument('video_path', help='path file .oni')
    args = p.parse_args()

    videoid = os.path.basename(args.video_path[:-4])  # nome del video
    framecount = 0
    pid = 0  # ID persona corrente (per classificazione)
    newid = False  # flag nuovo ID (per classificazione)
    peoplefeats = []  # vettore di features (per classificazione)
    altezze = []
    vteste = []
    vspalle = []
    vhsv_testa = []
    vhsv_spalle = []

    # inizializzazione di OpenNI e apertura degli stream video
    openni2.initialize()
    dev = openni2.Device.open_file(args.video_path)
    dev.set_depth_color_sync_enabled(True)
    depth_stream = dev.create_depth_stream()
    color_stream = dev.create_color_stream()
    depth_stream.start()
    color_stream.start()

    while framecount < depth_stream.get_number_of_frames() and framecount < color_stream.get_number_of_frames():
        dframe = depth_stream.read_frame()
        cframe = color_stream.read_frame()
        framecount += 1

        # conversione di tipo/formato per OpenCV
        depth_array = np.ndarray((dframe.height, dframe.width), dtype=np.uint16, buffer=dframe.get_buffer_as_uint16())
        color_array = cv2.cvtColor(np.ndarray((cframe.height, cframe.width, 3), dtype=np.uint8,
                                              buffer=cframe.get_buffer_as_uint8()), cv2.COLOR_RGB2HSV)

        # ALTEZZA ======================================================================================================

        # ci salviamo il background (primo frame) per sottrarlo a quello corrente
        if framecount == 1:
            background = depth_array.copy()
            mask_b = cv2.inRange(background, 0, 0)  # maschera dei pixel nulli del background
        foreground = cv2.absdiff(depth_array, background)

        mask_f = cv2.bitwise_or(mask_b, cv2.inRange(depth_array, 0, 0))  # maschera pixel nulli bg + pixel nulli deptharray
        mask_p = cv2.inRange(foreground, 150, 2500)  # maschera della persona (pixel tra 150 e 2500)
        cont, _ = cv2.findContours(mask_p.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        mask_p = np.zeros(mask_p.shape, np.uint8)

        # disegno solo il contorno con area maggiore
        if len(cont) > 0:
            cv2.drawContours(mask_p, cont, np.argmax([cv2.contourArea(x) for x in cont]), (255, 255, 255),
                             cv2.cv.CV_FILLED)
        mask_p = cv2.bitwise_and(mask_p, cv2.bitwise_not(mask_f))  # tolgo i pixel nulli dalla maschera del contorno

        _, hmax, _, _ = cv2.minMaxLoc(foreground, mask_p)  # ci accertiamo che l'altezza max trovata sia dentro la maschera

        # TESTA + SPALLE ===============================================================================================

        if hmax > 1500 and np.mean(mask_p[235:245, 315:325]) == 255:  # solo quando si passa al centro dell'immagine
            newid = args.cls  # nuova persona (solo se sto classificando)

            altezze.append(hmax)  # la consideriamo per la media
            mask_t = cv2.bitwise_and(mask_p, cv2.inRange(foreground, hmax - 150, hmax))  # testa fino a 15cm dal max
            mask_s = cv2.bitwise_and(mask_p, cv2.inRange(foreground, hmax - 500, hmax - 150))  # spalle tra 15 e 50cm dal max
            cont_t, _ = cv2.findContours(mask_t.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cont_s, _ = cv2.findContours(mask_s.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(cont_s) > 0:
                spalle = np.argmax([cv2.contourArea(x) for x in cont_s])
                if len(cont_s[spalle]) >= 5 and cv2.contourArea(cont_s[spalle]) > 5000:  # area spalle 5000 (soglia arbitraria)
                    vspalle.append(np.sqrt(4 * cv2.contourArea(cont_s[spalle]) * np.pi))  # circ. equivalente delle spalle

                    # erodo la maschera per eliminare le porzioni di pavimento che capitano nel frame a colori
                    hist_spalle = cv2.calcHist([color_array], [0, 1, 2], cv2.erode(mask_s, np.ones((10, 10), np.uint8)),
                                               [18, 4, 4], [0, 180, 0, 255, 0, 255])  # istogramma HSV spalle
                    vhsv_spalle.append(np.unravel_index(np.argmax(hist_spalle), hist_spalle.shape))  # colore dominante delle spalle

            if len(cont_t) > 0:
                testa = np.argmax([cv2.contourArea(x) for x in cont_t])
                if len(cont_t[testa]) >= 5 and cv2.contourArea(cont_t[testa]) > 2500:  # area testa 2500 (soglia arbitraria)
                    vteste.append(np.sqrt(4 * cv2.contourArea(cont_t[testa]) * np.pi))  # circ. equivalente della testa

                    # erodo la maschera per eliminare le porzioni di pavimento che capitano nel frame a colori
                    hist_testa = cv2.calcHist([color_array], [0, 1, 2], cv2.erode(mask_t, np.ones((10, 10), np.uint8)),
                                              [18, 4, 4], [0, 180, 0, 255, 0, 255])  # istogramma HSV testa
                    vhsv_testa.append(np.unravel_index(np.argmax(hist_testa), hist_testa.shape))  # colore dominante della testa

        else:
            if newid:  # se classifichiamo teniamo traccia delle singole persone
                pid += 1
                newid = False
                peoplefeats.append([pid, np.mean(altezze)/2500 if len(altezze) else 0, np.mean(vteste)/500 if len(vteste) else 0,
                                    np.amax(vspalle)/500 if len(vspalle) else 0,
                                    Counter(vhsv_testa).most_common(1)[0][0][0]/17. if len(vhsv_testa) else 0,
                                    Counter(vhsv_testa).most_common(1)[0][0][1]/3. if len(vhsv_testa) else 0,
                                    Counter(vhsv_testa).most_common(1)[0][0][2]/3. if len(vhsv_testa) else 0,
                                    Counter(vhsv_spalle).most_common(1)[0][0][0]/17. if len(vhsv_spalle) else 0,
                                    Counter(vhsv_spalle).most_common(1)[0][0][1]/3. if len(vhsv_spalle) else 0,
                                    Counter(vhsv_spalle).most_common(1)[0][0][2]/3. if len(vhsv_spalle) else 0])
                altezze = []
                vteste = []
                vspalle = []  # resettiamo tutto per una nuova persona
                vhsv_testa = []
                vhsv_spalle = []

        # FINE FEATURES ================================================================================================

    depth_stream.stop()
    color_stream.stop()

    # salvo le features in un file csv
    if not args.cls:
        with open('features_id.csv', 'a') as features:
            features.write(str(videoid) + ';')
            features.write(str(np.mean(altezze)/2500 if len(altezze) else 0) + ';')
            features.write(str(np.mean(vteste)/500 if len(vteste) else 0) + ';')
            features.write(str(np.amax(vspalle)/500 if len(vspalle) else 0) + ';')
            if len(vhsv_testa):
                features.write(str(Counter(vhsv_testa).most_common(1)[0][0][0]/17.) + ';')  # H testa
                features.write(str(Counter(vhsv_testa).most_common(1)[0][0][1]/3.) + ';')  # S testa
                features.write(str(Counter(vhsv_testa).most_common(1)[0][0][2]/3.) + ';')  # V testa
            else:
                features.write('0;0;0;')
            if len(vhsv_spalle):
                features.write(str(Counter(vhsv_spalle).most_common(1)[0][0][0]/17.) + ';')  # H spalle
                features.write(str(Counter(vhsv_spalle).most_common(1)[0][0][1]/3.) + ';')  # S spalle
                features.write(str(Counter(vhsv_spalle).most_common(1)[0][0][2]/3.) + '\n')  # V spalle
            else:
                features.write('0;0;0\n')

    else:  # classifichiamo con knn
        assert os.path.exists('features_id.csv') and os.path.getsize('features_id.csv'), \
            "features_id non esiste o è vuoto"
        traindata = np.loadtxt('features_id.csv', dtype=np.float32, delimiter=';')

        knn = cv2.KNearest()
        knn.train(traindata[:, 1:], np.matrix(traindata[:, 0]))
        _, results, _, dist = knn.find_nearest(np.matrix(peoplefeats, dtype=np.float32)[:, 1:], 1)
        for i in range(len(results)):
            print "person: {!s} -> class: {!s}, distance: {!s}".format(
                np.matrix(peoplefeats, dtype=np.float32)[:, 0][i], results[i], dist[i])

    openni2.unload()
    cv2.destroyAllWindows()

if __name__ == '__main__':
        main()
