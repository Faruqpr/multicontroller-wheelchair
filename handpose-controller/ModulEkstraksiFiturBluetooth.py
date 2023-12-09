
import copy
import cv2
import mediapipe as mp
import numpy as np
import math 
import os
import datetime
from keras.utils import load_img
from keras.models import load_model
from google.protobuf.json_format import MessageToDict
import time 
from keras import models

import bluetooth
# ESP32's Bluetooth MAC address
esp32_mac_address = "EC:62:60:9B:E4:92"  # Replace with your ESP32's MAC address

# Establish a Bluetooth connection
sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
sock.connect((esp32_mac_address, 1))  # Channel 1 is commonly used for SPP (Serial Port Profile)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cPinv = 2
cAng = 3
cLrinv=4
cZ =5
cOrientasi = 6
FrameRate = 5 
cap=[]
imsize=(640, 480)
hands = mp_hands.Hands( model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5) 

def GetFileName():
        x = datetime.datetime.now()
        s = x.strftime('%Y-%m-%d-%H%M%S%f')
        return s
def CreateDir(path):
    ls = [];
    head_tail = os.path.split(path)
    ls.append(path)
    while len(head_tail[1])>0:
        head_tail = os.path.split(path)
        path = head_tail[0]
        ls.append(path)
        head_tail = os.path.split(path)   
    for i in range(len(ls)-2,-1,-1):
        sf =ls[i]
        isExist = os.path.exists(sf)
        if not isExist:
            os.makedirs(sf)
         
def NormRow(v):
    r=np.sum(np.abs(v)**2,axis=-1)**(1./2)
    
    return r
    
    
#==================================================================                   
def ExtrakLandmark(hands,img):
    br,kl ,w= img.shape
    image =copy.copy(img) 
    #image.flags.writeable = False
    image2 = copy.deepcopy(image)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = hands.process(image)

    # Draw the hand annotations on the image.
    #image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lhd = []
    if results.multi_handedness:
        for hd in results.multi_handedness:
            handedness_dict = MessageToDict(hd)
            handedness= handedness_dict['classification'][0]["index"]
            lhd.append(handedness)
    lHandLandmark = []
            
    if results.multi_hand_landmarks:
      
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        lRes = np.zeros([21,3])
        for i in range(21):
            lRes[i,0]=hand_landmarks.landmark[i].x*kl;
            lRes[i,1]=hand_landmarks.landmark[i].y*br;
            lRes[i,2]=hand_landmarks.landmark[i].z*kl;
            
        lHandLandmark.append(lRes)
    return [lHandLandmark,image,image2,lhd]


def EkstrakFitur(lData):
    image = lData[1]
    image2 =lData[2]
    lhd = lData[3]
    lHandLandmark =lData[0]
    lFeature = []
    va=np.array([[0,1,2],
                 [1,2,3],
                 [2,3,4],
                 [0,5,6],
                 [5,6,7],
                 [6,7,8],
                 [0,9,10],
                 [9,10,11],
                 [10,11,12],
                 [0,13,14],
                 [13,14,15],
                 [14,15,16],
                 [0,17,18],
                 [17,18,19],
                 [18,19,20],
                 [4,0,8],
                 [8,0,12],
                 [12,0,16],
                 [16,0,20]])
    va = np.array(va, dtype=np.uint32)

    lrinv =[]
    lAng= []
    lpinv=[]
    limd = [] 
    lo = [] 
    for lRes in lHandLandmark:
        
        
        
        #Normalisasi
        xmin = int(lRes[:,0].min() )
        ymin = int(lRes[:,1].min()) 
        xmax = int(lRes[:,0].max())  
        ymax = int(lRes[:,1].max()) 
        tx = int((xmin+xmax)/2)        
        ty = int((ymin+ymax)/2)        
        
        dx = (xmax -xmin)
        dy = (ymax - ymin)
        dd = dx 
        
        if dy>dd:
            dd = dy
        d2 = int(dd/2)
        xmin = tx - d2 
        xmax = tx + d2
        ymin = ty - d2 
        ymax = ty + d2 
        if xmin<0:
            xmin = 0
            
        if xmax>image.shape[1]:
            xmax = image.shape[1]
        if ymin<0:
            ymin = 0
            
        if ymax>image.shape[0]:
            ymax = image.shape[0]
        
        dx = (xmax -xmin)
        dy = ymax - ymin
        print(dx,dy)
        imd = np.zeros((dy,dx,3)).astype("uint8")
       
        imd[:,:,:]  = image2[ymin:ymax,xmin:xmax,:]
        limd.append(copy.deepcopy(imd))
        
        p0 = lRes[0:1,:]
        p1 = lRes[5:6,:]
        p2 = lRes[17:18,:]
        
        vx = p1 - p0
        vx=vx/np.linalg.norm(vx)
        
        vd = p2 - p0
        vd=vd/np.linalg.norm(vd)
        
        
        vdx = p1-p2
        vdx =vdx/np.linalg.norm(vdx) 
        
        #Ekstrak fitur orientasi 
      
        #o1 = np.arctan2( vx[0,1] ,vx[0,0])/np.pi 
        #o2 = np.arctan2( vd[0,1] ,vd[0,0])/np.pi 
        #o3 = np.arctan2( vdx[0,1] ,vdx[0,0])/np.pi 
        
        #lo.append(np.array([o1,o2,o3]))
        lo.append(np.array([vx[0,0],vx[0,1],vd[0,0],vd[0,1] ,vdx[0,0],vdx[0,1]] ))
        
        vx = vx/np.linalg.norm(vx);
        vz =np.cross(vx,vd)
        vz = vz/np.linalg.norm(vz);
        vy = np.cross(vz,vx)
        m = np.zeros([4,4])
        m[0:3,0]=vx
        m[0:3,1]=vy
        m[0:3,2]=vz
        m[0:3,3]=p0
        m[3,3]=1
        
        mi = np.linalg.inv(m)
        pd =np.ones([21,4])
        pd[:,0:3]=lRes
        pinv =np.matmul(mi,pd.transpose()).transpose()
        pinv = pinv[:,0:3]

        #Ekstrak fitur koordinat ternormalisasi
        pinv=pinv/pinv[5,0]
        lpinv.append(pinv)
        #Ekstrak fitur jarak titik koordinat ternormalisasi
        r = NormRow(pinv)
        lrinv.append(r)
        
        #Ekstrak fitur angular
        v1 =lRes[va[:,0]]-lRes[va[:,1]]
        v2 =lRes[va[:,2]]-lRes[va[:,1]]
        v1[:,2]=0 
        v2[:,2]=0 
        
        v3 =np.cross(v1,v2)
        
        s = NormRow(v3)
        c = np.sum(v1*v2,axis=1)
        sd = np.arctan2(s,c)/math.pi 
        lAng.append(sd)
    
    lFeature = {"lHandLanmark":lHandLandmark,"lpinv":lpinv,"lrinv":lrinv ,"lAng":lAng,"image": image,"image2":image2,"limd":limd,"lhd":lhd,"lo":lo}
    return lFeature

def SaveFeature(sf,lFeature):
    lHandLandmark = lFeature["lHandLanmark"]
    lpinv = lFeature["lpinv"]
    lrinv = lFeature["lrinv"]
    lAng  = lFeature["lAng"] 
    lo  = lFeature["lo"] 
    
    image = lFeature["image"]
    image2 = lFeature["image2"]
    
    imd = lFeature["limd"]
    

    ndt = len(lHandLandmark)
    nda = np.array([ndt])
    sfFile = sf+"\\"+GetFileName()

    np.savetxt(sfFile+'.num', nda, delimiter=',') 
    cv2.imwrite(sfFile+'.jpg', image)
    cv2.imwrite(sfFile+'.jpeg', image2)
    
    
    
    
    
    for i in range( ndt): 
        sf1=sfFile+"_"+str(i) 
        np.savetxt(sf1+".HandLandmark", lHandLandmark[i], delimiter=',') 
        np.savetxt(sf1+".pinv", lpinv[i], delimiter=',') 
        np.savetxt(sf1+".rinv", lrinv[i], delimiter=',')         
        np.savetxt(sf1+".ang", lAng[i], delimiter=',') 
        np.savetxt(sf1+".orientasi", lo[i], delimiter=',') 
        np.savetxt(sf1+".pinvz", lpinv[i][:,2:3], delimiter=',') 
        np.savetxt(sf1+".or", lo[i], delimiter=',') 
        
        cv2.imwrite(sf1+'.png', imd[i])
        
        
        

def LoadFeature(sfFile):
    
    lHandLandmark = []
    lpinv = []
    lrinv = []
    lAng  = [] 
    lz    = []
    lo =[]


    nda = np.loadtxt(sfFile+'.num', delimiter=',') 
    ndt =np.int32(nda)
    #cv2.imwrite(sfFile+'.jpg', image)
    for i in range( ndt): 
        sf1=sfFile+"_"+str(i) 
         
        d=np.loadtxt(sf1+".HandLandmark", delimiter=',')
        lHandLandmark.append(d)
        d =np.loadtxt(sf1+".pinv", delimiter=',') 
        lpinv.append(d.flatten())
        d =np.loadtxt(sf1+".rinv",  delimiter=',')     
        lrinv.append(d.flatten())
        d= np.loadtxt(sf1+".ang",  delimiter=',') 
        lAng.append(d.flatten())
        d= np.loadtxt(sf1+".pinvz",  delimiter=',') 
        lz.append(d.flatten())
        d= np.loadtxt(sf1+".or",  delimiter=',') 
        lo.append(d.flatten())

    
        
    return [ndt, lHandLandmark, lpinv, lrinv, lAng,lz,lo]



    
def GetDirList(path): 
    ls  =[]
    if os.path.exists(path):
        for x in os.listdir(path):
           
            sf = os.path.join(path, x)
            
            if os.path.isdir(sf):
                ls.append(sf)
    return ls 


def GetDataList(path,Ext =".jpg" ): 
    ls  =[]
    if os.path.exists(path):
        for x in os.listdir(path):
            
            sf = os.path.join(path, x)
            
            
            
            if sf.endswith(Ext):
                print(sf)

                [f,xt]=os.path.splitext(sf)          
                ls.append(f)
                
                
    return ls 

def LoadDataSet(SDirektoriData,Kelas,sExt=".jpg"):

    #SDirektoriData ="c:\\temp\\Data"
    #Kelas =["Kanan" ,"Kiri"]
    
    nkelas = len(Kelas)
    print(nkelas)
    lb = np.identity(nkelas)
    print(lb)
    cFile = 0
     
    lLb=[]
    lndt=[]
    lpinv=[]
    lrinv=[]
    lAng=[]
    lz=[]
    lLb=[]
    lcFile = []
    lo = [] 
    lImd =[] 
    
    c =0
    
    for ic  in range(nkelas):
        kl = Kelas[ic]
        sd =  SDirektoriData+"\\"+kl
        
        l= GetDirList(sd)
    
        for  i in range(len(l)) : 
            f=l[i]
            ll = GetDataList(f,sExt)
            cFile = cFile+1
            for sfFiture in ll:
                fit = LoadFeature(sfFiture)
                ndt = fit[0]
                if ndt==0:
    #                lndt.append(ndt) 
     #               lcFile.append(cFile)
      #              lLb.append(lb[ic,:])
       #             lz.append(np.zeros((21)))
        #            lpinv.append(np.zeros((63)))
         #           lAng.append(np.zeros((19)))
          #          lrinv.append(np.zeros((21)))
           #         lo.append(np.zeros((3)))
            #        
                    c =c+1
                    
                    
                else:
                    for ihand in range(ndt):
                        lndt.append(ndt) 
                        lcFile.append(cFile)
                        lLb.append(lb[ic,:])
                        #print(lb[ic,:])
                        lpinv.append(fit[2][ihand].flatten())
                        lrinv.append(fit[3][ihand].flatten())
                        lAng.append(fit[4][ihand].flatten())
                        lz.append(fit[5][ihand].flatten()) 
                        lo.append(fit[6][ihand].flatten()) 
                        
                        c =c+1
                        #print(c)
                        
                    
    lcFile = np.array(lcFile)
    lpinv =np.array(lpinv)      
    lz = np.array(lz)
    lAng = np.array(lAng) 
    lrinv = np.array(lrinv)        
    lndt = np.array(lndt) 
    lLb = np.array(lLb)
    
    lo = np.array(lo)
    
    return  [lndt,lcFile, lpinv, lAng, lrinv,lz,lo, lLb]



def CreateDataSet(sDirektoriData,sKelas,NoKamera,FrameRate):
       

    sDirektoriKelas = sDirektoriData+"\\"+sKelas+"\\"+GetFileName()
    CreateDir(sDirektoriKelas )
    hands = mp_hands.Hands( model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5) 
    # For webcam input:
    cap = cv2.VideoCapture(NoKamera)
    TimeStart = time.time() 
    
    while cap.isOpened():
      success, image = cap.read()
    
      if not success:
        print("Ignoring empty camera frame.")
        continue
    
      image = cv2.resize(image, imsize)
      lData= ExtrakLandmark(hands,image)
      
      lFeature  =EkstrakFitur(lData)
      TimeNow = time.time() 
      if TimeNow-TimeStart>1/FrameRate:
          SaveFeature(sDirektoriKelas,lFeature)
          TimeStart =TimeNow
      
      cv2.imshow('MediaPipe Hands', cv2.flip(lData[1], 1))
      if cv2.waitKey(5) & 0xFF == 27:
        break
    cap.release()
    cv2.destroyAllWindows()


def SelectFitur(L,NoFitur=[cAng]):
    nd = L[0]>0
    L[0]=L[0][nd]
    L[1]=L[1][nd]
    L[2]=L[2][nd]
    L[3]=L[3][nd]
    L[4]=L[4][nd]
    L[5]=L[5][nd]
    L[6]=L[6][nd]
    
    F=[]
    if len(NoFitur)>0:
        F=L[NoFitur[0]]
        
        for i in range(1,len(NoFitur)):
            F =np.concatenate((F,L[NoFitur[i]]),axis=1)
    else:
        NoFitur=(2,3,4,5)
        F=L[NoFitur[0]]
        for i in range(1,len(NoFitur)):
            F =np.concatenate((F,L[NoFitur[i]]),axis=1)
    return  F,L[7]


def PilihFitur(L,NoFitur):
    F=L[NoFitur[0]]
    
    print(len(L))
    
    for j in range(1,len(NoFitur)):
        F =np.concatenate((F,L[NoFitur[j]]),axis=1)
        
    return F


def GetKelasNumber(P,th):
    P = np.array(P)    
    b,k = P.shape
    res = -1
    
    maxdt =np.max(P)
    if maxdt>th:
        for i in range(k):
            if P[0,i]==maxdt:
                res = i
                break
    return res
    
        
def DrawText(img,sText,x,y):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    posf = (x,y)
    fontScale              = 5
    fontColor              = (255,255,255)
    thickness              = 1
    lineType               = 2
    print("Masuk")
    cv2.putText(img,sText, 
        posf, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
    return copy.deepcopy(img)
    
    

def DeteksiTiapFrameFrame(model,image,Fitur):
    
    lData= ExtrakLandmark(hands,image)
    lFeature  =EkstrakFitur(lData)
    F=np.zeros((1,1))
    img = cv2.flip(copy.deepcopy(lData[1]),1) 
    LPredict  =[]
    
    
    for i in range(len(lFeature["lHandLanmark"])):
        lpinv=np.array([lFeature["lpinv"][i].flatten()])
        lrinv=np.array([lFeature["lrinv"][i].flatten()])
        lAng=np.array([lFeature["lAng"][i].flatten()])
        HandNum = lFeature["lhd"][i]
        lz =np.array([lFeature["lpinv"][i][:,2:3].flatten()]) 
        lo=np.array([lFeature["lo"][i].flatten()])
        
        L = [[],[], lpinv, lAng, lrinv,lz,lo] 
        F = PilihFitur(L,Fitur)
        predictionsPose=model.predict(F,verbose=0)
        print(predictionsPose)
        PoseNum= GetKelasNumber(predictionsPose,0.5)
        LPredict.append([HandNum,PoseNum])
    return LPredict,img,
    

def Deteksi2(NoKamera,NamaModel,Kelas,Fitur=[cAng], imsize = (640,480)):
    KiKa =["Kiri","Kanan"]
    cap = cv2.VideoCapture(NoKamera)
    model =load_model(NamaModel)
    TimeStart = time.time() 
    TimeNow = time.time() +2/FrameRate
    

    while cap.isOpened():
      success, image = cap.read()
    
      if not success:
        print("Ignoring empty camera frame.")
        continue
      
      if TimeNow-TimeStart>1/FrameRate :
         TimeStart =TimeNow
         image = cv2.resize(image, imsize)
         L,img = DeteksiTiapFrameFrame(model,image,Fitur)
         
         for fit in L:
            HandNum = fit[0]
            PoseNum =fit[1]
            cv2.putText(img, KiKa[HandNum], (0, 30+int(HandNum*30)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
            cv2.putText(img, Kelas[PoseNum], (100, 30+int(HandNum*30)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
     
            # Send ke ESP32
            if Kelas[PoseNum] == 'Kiri':
                #command = 'A'
                #s.sendall(command.encode())
                sock.send('A\n')
            elif Kelas[PoseNum] == 'Maju':
                #command = 'B'
                #s.sendall(command.encode())
                sock.send('B\n')
            elif Kelas[PoseNum] == 'Stop':
                #command = 'C'
                #s.sendall(command.encode())
                sock.send('C\n')
            elif Kelas[PoseNum] == 'Mundur':
                #command = 'D'
                #s.sendall(command.encode())
                sock.send('D\n')
            elif Kelas[PoseNum] == 'Kanan':
                #command = 'E'
                #s.sendall(command.encode())
                sock.send('E\n')
        
            
    #  cv2.imwrite("c:\\temp\\res\\"+GetFileName()+'.jpg', img)
      TimeNow = time.time()
      cv2.imshow('Deteksi Pose', img)
      
      if cv2.waitKey(5) & 0xFF == 27:
        break
    #end while 
    print("Exit")
    sock.close()
    cap.release()
    cv2.destroyAllWindows()
    
def LoadModel(sf):
  ModelCNN=load_model(sf)   
  return ModelCNN         

def DeteksiHandCNN(sKelas,NoKamera,sf,isize=(128,128)):
    ModelCNN = LoadModel(sf)
    hands = mp_hands.Hands( model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5) 
    # For webcam input:
    cap = cv2.VideoCapture(NoKamera)
    TimeStart = time.time() 
    
    while cap.isOpened():
      success, image = cap.read()
    
      if not success:
        print("Ignoring empty camera frame.")
        continue
    
      image = cv2.resize(image, imsize)
      lData= ExtrakLandmark(hands,image)
      
      lFeature  =EkstrakFitur(lData)
      imd = lFeature[6]
      
      TimeNow = time.time() 
      
      if TimeNow-TimeStart>1/FrameRate:
          
          if len(imd)>0:
              X = [] 
              for img in imd: 
                  img=cv2.resize(img,isize)
                  img= np.asarray(img)/255
                  img=img.astype('float32')
                  X.append(img)
              X=np.array(X)
              X=X.astype('float32')
              hs=ModelCNN.predict(X)
              LKlasifikasi=[];
              LKelasCitra =[];
              n = X.shape[0]
              for i in range(n):
                  v=hs[i,:]
                  if v.max()>0.5:
                      idx = np.max(np.where( v == v.max()))
                      LKelasCitra.append(sKelas[idx])
                  else:
                      idx=-1
                      LKelasCitra.append("-")
                  #------akhir if
                  LKlasifikasi.append(idx);
              #----akhir for
              LKlasifikasi = np.array(LKlasifikasi)
          TimeStart =TimeNow
      
      cv2.imshow('MediaPipe Hands', cv2.flip(lData[1], 1))
      if cv2.waitKey(5) & 0xFF == 27:
        break
    cap.release()
    cv2.destroyAllWindows()
    
def split(X,Y,pr): 
    br = X.shape[0]
    rd = np.random.rand(br)
    
    nr = rd<pr
    Xtr = X[nr,:]
    Ytr = Y[nr,:]
    nr = rd>=pr
    Xtest = X[nr,:]
    Ytest = Y[nr,:]
    return Xtr,Ytr,Xtest,Ytest

def splitimage(X,Y,pr): 
    br = X.shape[0]
    rd = np.random.rand(br)
    nr = rd<pr
    Xtr = X[nr,:,:,:]
    Ytr = Y[nr,:]
    nr = rd>=pr
    Xtest = X[nr,:,:,:]
    Ytest = Y[nr,:]
    return Xtr,Ytr,Xtest,Ytest
