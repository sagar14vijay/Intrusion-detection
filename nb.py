import numpy as np
import pandas as pd
import math
import random

def naivebayes(extractedData,TestData):

    def expo1(xi,j):
        y=float(math.pow((xi-means1[j]),2))*(-1)
        h=2*var1[j]
        if(h==0):
            if(xi==means1[j]):
                return 1
            else:
                return 0
        t=float(y/(1.0*h))
        inter=math.exp(t)
        denom=math.sqrt(2*math.pi)*math.sqrt(var1[j])
        ans=float(inter/(1.0*denom))
        return ans
        
    def expo2(xi,j):
        y=float(math.pow((xi-means2[j]),2))*(-1)
        h=2*var2[j]
        if(h==0):
            if(xi==means2[j]):
                return 1
            else:
                return 0
        t=float(y/(1.0*h))
        inter=math.exp(t)
        denom=math.sqrt(2*math.pi)*math.sqrt(var2[j])
        ans=float(inter/(1.0*denom))
        return ans
        
    def expo3(xi,j):
        y=float(math.pow((xi-means3[j]),2))*(-1)
        h=2*var3[j]
        if(h==0):
            if(xi==means3[j]):
                return 1
            else:
                return 0
        t=float(y/(1.0*h))
        inter=math.exp(t)
        denom=math.sqrt(2*math.pi)*math.sqrt(var3[j])
        ans=float(inter/(1.0*denom))
        return ans
        
    def expo4(xi,j):
        y=float(math.pow((xi-means4[j]),2))*(-1)
        h=2*var4[j]
        if(h==0):
            if(xi==means4[j]):
                return 1
            else:
                return 0
        t=float(y/(1.0*h))
        inter=math.exp(t)
        denom=math.sqrt(2*math.pi)*math.sqrt(var4[j])
        ans=float(inter/(1.0*denom))
        return ans
        
    def expo5(xi,j):
        y=float(math.pow((xi-means5[j]),2))*(-1)
        h=2*var5[j]
        if(h==0):
            if(xi==means5[j]):
                return 1
            else:
                return 0
        t=float(y/(1.0*h))
        inter=math.exp(t)
        denom=math.sqrt(2*math.pi)*math.sqrt(var5[j])
        ans=float(inter/(1.0*denom))
        return ans
    
    

    r,c=extractedData.shape
    
    count1=0
    count2=0
    count3=0
    count4=0
    count5=0
    for i in range(r):
        if(extractedData[i,c-1]==1):
            count1+=1
        if(extractedData[i,c-1]==2):
            count2+=1
        if(extractedData[i,c-1]==3):
            count3+=1
        if(extractedData[i,c-1]==4):
            count4+=1
        if(extractedData[i,c-1]==5):
            count5+=1
            
            
    means1=[0.0]*(c-1)
    means2=[0.0]*(c-1)
    means3=[0.0]*(c-1)
    means4=[0.0]*(c-1)
    means5=[0.0]*(c-1)
    
    
    for i in range(c-1):
       for j in range(r):
           if(extractedData[j,c-1]==1):
               means1[i]+=extractedData[j,i]    
           if(extractedData[j,c-1]==2):
               means2[i]+=extractedData[j,i]
           if(extractedData[j,c-1]==3):
               means3[i]+=extractedData[j,i]
           if(extractedData[j,c-1]==4):
               means4[i]+=extractedData[j,i]
           if(extractedData[j,c-1]==5):
               means5[i]+=extractedData[j,i]
           
    for i in range(c-1):
        means1[i]=float(means1[i]/(count1*1.0))
        means2[i]=float(means2[i]/(count2*1.0))
        means3[i]=float(means3[i]/(count3*1.0))
        means4[i]=float(means4[i]/(count4*1.0))
        means5[i]=float(means5[i]/(count5*1.0))
        
    var1=[0.0]*(c-1)
    var2=[0.0]*(c-1)
    var3=[0.0]*(c-1)
    var4=[0.0]*(c-1)
    var5=[0.0]*(c-1)
    
    for i in range(c-1):
        for j in range(r):
            if(extractedData[j,c-1]==1):
                var1[i]+=math.pow((extractedData[j,i]-means1[i]),2)
                
            if(extractedData[j,c-1]==2):
                var2[i]+=math.pow((extractedData[j,i]-means2[i]),2)
                
            if(extractedData[j,c-1]==3):
                var3[i]+=math.pow((extractedData[j,i]-means3[i]),2)
                
            if(extractedData[j,c-1]==4):
                var4[i]+=math.pow((extractedData[j,i]-means4[i]),2)
                
            if(extractedData[j,c-1]==5):
                var5[i]+=math.pow((extractedData[j,i]-means5[i]),2)
            
    for i in range(c-1):
        var1[i]=float(var1[i]/((count1-1)*1.0))
        var2[i]=float(var2[i]/((count2-1)*1.0))
        var3[i]=float(var3[i]/((count3-1)*1.0))
        var4[i]=float(var4[i]/((count4-1)*1.0))
        var5[i]=float(var5[i]/((count5-1)*1.0))
        r1=r
        
        
        
        #start table testing
    r,c=TestData.shape
    acc=0
    for i in range(r):
        p1=float(count1/(r1*1.0))
        p2=float(count2/(r1*1.0))
        p3=float(count3/(r1*1.0))
        p4=float(count4/(r1*1.0))
        p5=float(count5/(r1*1.0))
        for j in range(c-1):
            p1*=expo1(TestData[i,j],j)  
            p2*=expo2(TestData[i,j],j)
            p3*=expo3(TestData[i,j],j)
            p4*=expo4(TestData[i,j],j)
            p5*=expo5(TestData[i,j],j)   
        pred = max(p1,p2,p3,p4,p5)

        if(pred==p1):
            pred=1
            if(TestData[i,c-1]==pred):
                acc+=1 
                
        elif(pred==p2):
            pred=2
            if(TestData[i,c-1]==pred):
                acc+=1 
                
        elif(pred==p3):
            pred=3
            if(TestData[i,c-1]==pred):
                acc+=1 
                
        elif(pred==p4):
            pred=4
            if(TestData[i,c-1]==pred):
                acc+=1 
                
        elif(pred==p5):
            pred=5
            if(TestData[i,c-1]==pred):
                acc+=1 
        else:
            print p1,p2,p3,p4,p5
            print TestData[i,c-1],pred

    accuracy=float(acc/(r*1.0))
    #print accuracy
    return accuracy


#def naiveAccu():
#    ans=naivebayes(data)
#    return ans


#print "Naive bais accuracy"
#print naiveAccu()
