# -*- coding: utf-8 -*-
"""
Código baseado na Master Class de Visão computacional de Carlos Melo, Sigmoidal.
"""

import cv2
import numpy as np
import os

#constantes do modelo Yolo
CONFIDENCE_MIN  = 0.4
NMS_THRESHOLD = 0.2
MODEL_BASE_PATH = 'yolo-coco'  ## Utilizar o seu modelo, criar uma pasta com este nome e colocar os arquivos
                                ## de peso, nomes e configs.
    
VIDEO_BASE_PATH = 'vigilancia.mp4' #colocar o endereço do vídeo


#extrair os nomes das classes a partir do arquivo salvo
print('[+] Extraindo os nomes das classes')

with open(os.path.sep.join([MODEL_BASE_PATH,'coco.names'])) as f:
    labels = f.read().strip().split('\n')
    
    
    #gerar cores para cada label
    np.random.seed(42) #ter sempre as mesmas cores
    colors = np.random.randint(0,255,size=(len(labels),3),dtype = 'uint8')


#carregar o modelo treinado YOLO (c/ COCO dataset)
print('[+] Carregando o modelo Yolo Treinado com o dataset COCO')
net = cv2.dnn.readNetFromDarknet(
    os.path.sep.join([MODEL_BASE_PATH,'yolov3.cfg']),
    os.path.sep.join([MODEL_BASE_PATH,'yolov3.weights']))

#extrair layer não conectadas da arquitetura Yolo
ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]


vs = cv2.VideoCapture(VIDEO_BASE_PATH)
out = cv2.VideoWriter('colocar o endereço onde você quer salvar o vídeo', -1, 10.0, (1280,720))

print('[+]Iniciando a Captura')
while True:
    
            ret,frame = vs.read()
            if ret is False:
                break
            frame = cv2.resize(frame,(1280,720))
    
    
            (h,w) = frame.shape[:2]
            #construir um container blob e fazer uma passagem (forward) na modelo Yolo
            blob = cv2.dnn.blobFromImage(frame,1/255.0,(416,416),swapRB=True,crop=False)
            net.setInput(blob)
            layers_output = net.forward(ln)
            
            #criar listas com boxes, nível de confiança e ids das classes:
            boxes = []
            confidences = []
            class_ids  = []
            
            for output in layers_output:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    #filtrar pelo threshold da confiança
                    if confidence>CONFIDENCE_MIN and class_id in [0,1,2,3]:
                        box = detection[0:4] * np.array([w,h,w,h])
                        (center_x,center_y,width,height) = box.astype('int')
                        
                        x = int(center_x - (width/2))
                        y = int(center_y - (height/2))
                        
                        boxes.append([x,y,int(width),int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            
            #elimar os ruídos e redundâncias aplicando a no-max supression
            new_ids = cv2.dnn.NMSBoxes(boxes,confidences,CONFIDENCE_MIN,NMS_THRESHOLD)
            
           
            if len(new_ids) > 0 :
                for i in new_ids.flatten():
                    (x,y) = (boxes[i][0], boxes[i][1])
                    (w,h) = (boxes[i][2], boxes[i][3])
                    
                    
                    #plotar retângulos e textos pré-definidos na frame
                    text = '{}: {:.4f}'.format(labels[class_ids[i]],confidences[i])
                    text2 = 'Video de Exemplo'
                    color_picked = [int(c) for c in colors[class_ids[i]]]
                    
                    
                   
                    cv2.rectangle(frame, (x,y), (x+w,y+h), color_picked)
                    cv2.putText(frame,text,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color_picked,2)
                    cv2.putText(frame,text2,(400,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
            out.write(frame)
            cv2.imshow('frame',frame)
    
            key = cv2.waitKey(1) & 0xFF
    
            if key == ord('q'):
                break
    
    
cv2.destroyAllWindows()
vs.release()
out.release()
