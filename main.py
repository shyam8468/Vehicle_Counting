import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import cvzone

model=YOLO('yolov8s.pt')



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture('vid2.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0
cy1=424


tracker1=Tracker()
tracker2=Tracker()
tracker3=Tracker()



counter1=[]
counter2=[]
counter3=[]
offset=6
while True:    
    ret,frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue
    
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list1=[]
    motorcycle=[]
    list2=[]
    car=[]
    list3=[]
    truck=[]
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'motorcycle' in c:
            list1.append([x1,y1,x2,y2])
            motorcycle.append(c)
        elif 'car' in c:
            list2.append([x1,y1,x2,y2])
            car.append(c)
        elif 'truck' in c:
            list3.append([x1,y1,x2,y2])
            truck.append(c)
            
    bbox1_idx=tracker1.update(list1)
    bbox12_idx=tracker2.update(list2)
    bbox123_idx=tracker3.update(list3)

    for bbox1 in bbox1_idx:
        for i in motorcycle:
            x3,y3,x4,y4,id1=bbox1
            cxm=int(x3+x4)//2
            cym=int(y3+y4)//2
            
            cv2.circle(frame,(cxm,cym),4,(0,255,0),-1)
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),1)
            cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
            if counter1.count(id1)==0:
                counter1.append(id1)

    ##############################Car######################
    for bbox2 in bbox12_idx:
        for h in car:
            x5,y5,x6,y6,id2 = bbox2
            cxc=int(x5+x6)//2
            cyc=int(y5+y6)//2
            
            cv2.circle(frame,(cxc,cyc),4,(0,255,0),-1)
            cv2.rectangle(frame,(x5,y5),(x6,y6),(0,0,255),1)
            cvzone.putTextRect(frame,f'{id2}',(x5,y5),1,1)
            if counter2.count(id2)==0:
                counter2.append(id2)
   
    ############################Truck####################################

    for bbox3 in bbox123_idx:
        for t in truck:
            x7,y7,x8,y8,id3 = bbox3
            cxt=int(x7+x8)//2
            cyt=int(y7+y8)//2
            
            cv2.circle(frame,(cxt,cyt),4,(0,255,0),-1)
            cv2.rectangle(frame,(x7,y7),(x8,y8),(0,0,255),1)
            cvzone.putTextRect(frame,f'{id3}',(x7,y7),1,1)
            if counter3.count(id3)==0:
                counter3.append(id3)

    cv2.line(frame,(2,cy1),(794,cy1),(0,0,255),2)

  
    motorcyclec=(len(counter1))
    carc = len(counter2)
    truckc = len(counter3)
    # cvzone.putTextRect(frame,f'motorcyclec:-{motorcyclec}',(19,30),2,1)
    # cvzone.putTextRect(frame,f'Car:-{carc}',(18,71),2,1)
    # cvzone.putTextRect(frame,f'Car:-{truckc}',(17,100),2,1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(0)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()




