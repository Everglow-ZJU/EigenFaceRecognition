import cv2 as cv
import numpy as np
import os
import sys
#main function
def main(argv):
    if not os.path.exists(argv[2]):
        os.makedirs(argv[2])
    path='./'+argv[3]+'/'#Get the path to the folder where the photos are stored
    energyRatio=float(argv[1])
    #filelist=os.listdir(path)#Get a list of photo file names
    img_size=(90,90)
    #read images
    ImageMatrix,count,Class=readImage(path,img_size)
    np.savetxt('./'+argv[2]+'/Class.csv', Class, delimiter = ',')
    #get meanFace
    mean_vector=meanFace(ImageMatrix,count)
    np.savetxt('./'+argv[2]+'/mean.csv', mean_vector, delimiter = ',')
    #compute the  fake_covarianceMatrix
    matrix_prime,S=covarianceMatrix(ImageMatrix,mean_vector,count)
    np.savetxt('./'+argv[2]+'/matrix_prime.csv', matrix_prime, delimiter = ',')
    #compute the eigenvalues and eigenvectors
    eigenvalues,eigenvectors = np.linalg.eig(S)
    eigenvalues=eigenvalues.reshape(1,-1)
    eigenvectors=np.dot(matrix_prime,eigenvectors)
    #get transformation matrix
    W=getPrincipal(eigenvectors,eigenvalues,energyRatio)
    np.savetxt('./'+argv[2]+'/PCA.csv', W, delimiter = ',')
    #show and write the mean image of the first 10 eigen faces
    EigenFaces=W[:,0:10].sum(1)/10
    EigenFaces=(EigenFaces.reshape(img_size))
    EigenFaces=cv.normalize(EigenFaces,None,0,255,cv.NORM_MINMAX,cv.CV_8UC1)
    cv.imwrite('EigenFaces.png',EigenFaces)
    cv.imshow('first 10 eigen faces',EigenFaces)
    cv.waitKey(0)
    return 0
def readImage(path,size):
    width=size[0]
    height=size[1]
    Matrix_X=np.empty((height*width,400),np.uint8)
    Class=np.empty((1,400),np.uint8)
    #print(Matrix_X[:,0:1].shape)
    count=0
    for i in range(1,41,1):
        cur_path=path+str(i)+'/'
        filelist=os.listdir(cur_path)
        for item in filelist: 
            if item.endswith('.png'):
                if item[0:2]=='10':
                #use the last img as test
                    if i<10:
                        string='0'+str(i)
                    else:
                        string=str(i)
                    if not os.path.exists("testData"):
                        os.makedirs("testData")
                    img=cv.imread(cur_path+'10.png',cv.IMREAD_GRAYSCALE)
                    cv.imwrite('./testData/'+string+'_'+'10.png',img)
                else:
                    #read a image
                    img=cv.imread(cur_path+item,cv.IMREAD_GRAYSCALE)
                    re_img=cv.resize(img,size,0,0,interpolation = cv.INTER_CUBIC)
                    #histogram equalization
                    equ_img = cv.equalizeHist(re_img)
                    #cv.imshow('1',equ_img)
                    final_img = cv.normalize(equ_img,None,0,255,cv.NORM_MINMAX,cv.CV_8UC1)#TODO check
                    #reshape the image to a vector
                    vector_x=final_img.reshape((-1,1))#note -1 means let python calculate
                    Matrix_X[:,count:count+1] =vector_x
                    Class[0,count]=i
                    count+=1
    return Matrix_X[:,0:count],count,Class[0,0:count]
def meanFace(matrix,count):
    mean=np.zeros((matrix.shape[0],1))
    mean=matrix.sum(1).reshape(-1,1)
    mean=mean/count
    return mean
#compute the covarianceMatrix
def covarianceMatrix(matrix,mean,count):
    matrix_prime=matrix-mean
    S=np.dot((matrix_prime.T),matrix_prime)/count
    return matrix_prime,S
def getPrincipal(eigenvectors,eigenvalues,energyRatio):
    eigenvalues=eigenvalues.reshape(1,-1)
    #sort and get index
    sorted_index=np.argsort(-eigenvalues)
    totalSum=np.sum(eigenvalues)
    thisSum=0
    k=0# number of principal components selected
    
    for i in range(0,sorted_index.shape[1]):
        index=sorted_index[0,i]
        thisSum+=eigenvalues[0,index]
        k+=1
        #find the least k
        if thisSum/totalSum >= energyRatio:
            break
    #select k components
    sorted_index=sorted_index[0,0:k].reshape(1,-1)
    W=np.empty((eigenvectors.shape[0],k))
    j=0
    for i in range(0,sorted_index.shape[1]):
        index=sorted_index[0,i]
        #normalize the eigenvectors
        W[:,j:j+1]=eigenvectors[:,index:index+1]/np.sqrt(np.sum(eigenvectors[:,index:index+1]**2))
        j+=1
    #return the transformation matrix
    return W

if __name__ == "__main__":
    main(sys.argv) 
    #Command line input:
    #argv[0] for the file name, e.g. mytrain.py
    #argv[1] for the energyRatio
    #argv[2] for the model folder's name
    #argv[3] for the name of the data folder
    # python mytrain.py 0.95 modelData data