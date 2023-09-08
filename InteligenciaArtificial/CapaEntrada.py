import cv2 as cv
import os
import imutils
modelo='FotosNahuel'
ruta1='D:/Documentos/Documentos/Proyectos Python/Recursos/reconocimientofacial1/data'
rutacompleta = ruta1 + '/'+ modelo
if not os.path.exists(rutacompleta):
    os.makedirs(rutacompleta)



camara=cv.VideoCapture(0) #'D:/Documentos/Documentos/Proyectos Python/Recursos/reconocimientofacial1/Recursos/Modelo.mp4'
ruidos=cv.CascadeClassifier('D:/Documentos/Documentos/Proyectos Python/Recursos/entrenamientos opencv ruidos/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
id=0               
while True:
    respuesta,captura=camara.read()
    if respuesta==False:break
    captura=imutils.resize(captura,width=640)

    grises=cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    idcaptura=captura.copy()

    cara=ruidos.detectMultiScale(grises,1.3,5)

    for(x,y,e1,e2) in cara:
        cv.rectangle(captura, (x,y), (x+e1,y+e2), (0,255,0),2)
        rostrocapturado=idcaptura[y:y+e2,x:x+e1]
        rostrocapturado=cv.resize(rostrocapturado, (160,160),interpolation=cv.INTER_CUBIC)
        cv.imwrite(rutacompleta+'/imagen_{}.jpg'.format(id), rostrocapturado)
        id=id+1
    
    cv.imshow("Resultado rostro", captura)

    if id==350:
        break
camara.release()
cv.destroyAllWindows()
print("Finalizo la captura")