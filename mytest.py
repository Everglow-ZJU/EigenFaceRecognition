import cv2 as cv
import numpy as np
import os
import sys
def main(argv,filename):
    img_size=(90,90)
    model_path='./'+argv[2]+'/'
    mean,model_matrix,matrix_prime,Class=inputModel(model_path)
    #read test img
    test_img=cv.imread('./'+sys.argv[1]+'/'+filename,cv.IMREAD_GRAYSCALE)
    test_img=cv.resize(test_img,img_size,0,0,interpolation = cv.INTER_CUBIC)
    test_vector=getImgVector(test_img)
    #Projecting the test image into the PCA subspace
    test_projected=PCA_project(mean,model_matrix,test_vector)
    #project training img
    train_projectedM=np.dot(model_matrix.T,matrix_prime)
    
    #findNearest
    minClass,bestImgV=findNearest(test_projected,Class,train_projectedM)
    #reconstruct the best image
    best_img_reconstruct=PCA_reconstruct(mean,model_matrix,bestImgV)
    best_img=(best_img_reconstruct.reshape(img_size))
    best_img=cv.normalize(best_img,None,0,255,cv.NORM_MINMAX,cv.CV_8UC1)
    #show the best img
    cv.namedWindow('Best match',cv.WINDOW_AUTOSIZE)
    cv.imshow('Best match',best_img)
    #show the result
    result_img=test_img
    font = cv.FONT_HERSHEY_PLAIN
    #put the recognition result on the result_img
    cv.putText(result_img,str(int(minClass)),(50,70), font, 2,(0,0,255),2,cv.LINE_AA)
    cv.namedWindow('recognition result',cv.WINDOW_AUTOSIZE)
    if not os.path.exists("testResult"):
        os.makedirs("testResult")
    cv.imwrite('./testResult'+'/'+filename,result_img)
    cv.imshow('recognition result',result_img)
    cv.waitKey(0)
    #return the flag to show if the result is correct
    if int(filename[0:2])==int(minClass):
        return 1
    else:
        return 0
def inputModel(model_path):
    #input mean.csv
    try:
        with open(model_path+'mean.csv', "rb") as f:
            mean= np.loadtxt(f, delimiter = ",", skiprows = 0) 
    except IOError:
        return None
    #input PCA.csv
    try:
        with open(model_path+'PCA.csv', "rb") as f:
            model_matrix= np.loadtxt(f, delimiter = ",", skiprows = 0) 
    except IOError:
        return None
    #input matrix_prime.csv
    try:
        with open(model_path+'matrix_prime.csv', "rb") as f:
            matrix_prime= np.loadtxt(f, delimiter = ",", skiprows = 0)
    except IOError:
        return None
    #input Class.csv
    try:
        with open(model_path+'Class.csv', "rb") as f:
            Class= np.loadtxt(f, delimiter = ",", skiprows = 0)
    except IOError:
        return None
    #if all read,return them
    return mean,model_matrix,matrix_prime,Class
    
def PCA_project(mean,model_matrix,x):
    mean=mean.reshape(-1,1)
    y=np.dot(model_matrix.T,x-mean)
    return y
def PCA_reconstruct(mean,model_matrix,y):
    mean=mean.reshape(-1,1)
    x=np.dot(model_matrix,y)+mean
    return x
def Euclid_Distance(v1,v2): #compute the Euclid distance between 2 vectors
    v1=v1.reshape(-1,1)
    v2=v2.reshape(-1,1)
    Dist=np.sqrt(np.sum((v1-v2)**2))
    return Dist
def getImgVector(img):
    equ_img = cv.equalizeHist(img)
    final_img = cv.normalize(equ_img,None,0,255,cv.NORM_MINMAX,cv.CV_8UC1)
    vector_x=final_img.reshape((-1,1))
    return vector_x
def findNearest(test_projected,Class,train_projectedM):
    Class=Class.reshape((-1,1))
    minDist=float("inf")
    #get the best match image and its class
    for i in range(0,train_projectedM.shape[1]):
        #compute the distance between the test vector and the training vector
        Dist=Euclid_Distance(test_projected,train_projectedM[:,i:i+1])
        if Dist < minDist:
            minDist=Dist
            minClass=Class[i,0]
            bestImgV= train_projectedM[:,i:i+1]
    return minClass,bestImgV
if __name__ == "__main__":
    testpath='./'+sys.argv[1]+'/'
    filelist=os.listdir(testpath)
    length=len(filelist)
    lst=[]
    for filename in filelist:
        lst.append(main(sys.argv,filename))
    #print the accuracy
    print(sum(lst)/length)
    #Command line input:
    #argv[0] for the file name, e.g. mytest.py
    #argv[1] for the test folder
    #argv[2] for the model file folder
    # python mytest.py testData modelData
