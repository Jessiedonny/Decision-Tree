
import csv
import sys
import os
from genericpath import exists
from operator import indexOf
import random
##################################
#Load training_data into 2 lists- training_instances and attributes
##################################
#take command line argument
try:
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
except IndexError:
    raise SystemExit(f"Usage: {sys.argv[0]} <argument missing - check readme.txt>")
file_dir=os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
data_dir=file_dir+'/data/'

outputlines=[]

training_instances=[]
training_attributes=[]  
class_label=[] 
#read training data
with open(data_dir+str(arg1),newline='')  as data_csv:
    csvreader=csv.reader(data_csv,delimiter=' ')
    training_attributes=next(csvreader)
    #print(header)
    for row in csvreader:
        training_instances.append(row)

#read test data
test_insts=[]
test_attributes=[]
with open(data_dir+str(arg2),newline='')  as data_csv:
    csvreader=csv.reader(data_csv,delimiter=' ')
    test_attributes=next(csvreader)
    #print(header)
    for row in csvreader:
        test_insts.append(row)

#get class label
classlabel=[]
for i in range(len(training_instances)):
    class_label.append(training_instances[i][0])    

label=set(class_label)
for x in label:
    classlabel.append(x)

A=classlabel[0]
B=classlabel[1]



def prob(instances):
    m=0
    n=0
    if range(len(instances)>0):
        for i in range(len(instances)):
            #print(instances[i][0])
            if instances[i][0]==A:
                m=m+1
            elif instances[i][0]==B:
                n=n+1  
        if m>n:
            classlabel=A
            return classlabel+', prob is '+str(m/(m+n)*100)+'%'
        if m<n:
            classlabel=B
            return classlabel+', prob is '+str(n/(m+n)*100)+'%'
        if m==n:
            classlabel=random.choice([A,B])
            return classlabel+', prob is '+str(m/(m+n)*100)+'%'



##function to calculate impurity of a instance set
def impurity(instances):
    m=0 #count the live lable
    n=0 #count the die lable
    if range(len(instances)>0):
        for i in range(len(instances)):
            #print(instances[i][0])
            if instances[i][0]==A:
                m=m+1
            elif instances[i][0]==B:
                n=n+1    
        imp=m*n/((m+n)**2)
    else:
        imp=0
    return imp
    


def choosebestatt(attributes,instances):
    bestinststrue=[]
    bestinstsfalse=[]
    bestatt=''
    min_wimp=1000
    for i in range(len(attributes)-1):
    #for i in range(1):   
        inststrue=[]
        instsfalse=[]
        attindexoriginal=training_attributes.index(attributes[i+1])
        for j in range(len(instances)):     
            if instances[j][attindexoriginal]=='true':   
                inststrue.append(instances[j])
            elif instances[j][attindexoriginal]=='false':
                instsfalse.append(instances[j])   
        #return(inststrue)
        imp_inststrue=impurity(inststrue)
        imp_instsfalse=impurity(instsfalse)
        weightedimp=float(len(inststrue))/float(len(instances))*imp_inststrue+float(len(instsfalse))/float(len((instances)))*imp_instsfalse
        #print(attributes[i+1]+str(weightedimp))
        if weightedimp<min_wimp:
            min_wimp=weightedimp
            bestatt=attributes[i+1]
            bestinststrue=inststrue
            bestinstsfalse=instsfalse
        #print(min_wimp)
        #print(bestatt)
    return bestatt, bestinststrue, bestinstsfalse
        

##function to build tree
def buildtree(instances, attributes):  
    lefttree=''
    righttree=''
    ds=[]
    if instances==False:
        return str(prob(training_instances))+'   #baseline probability'
    elif impurity(instances)==0:
        if(len(instances)>0):
            return str(instances[0][0])+',prob is 1'+ '     #'+str(len(instances))+' instances'
        else:
            return str(instances)+',prob is 1'+ '     #'+str(len(instances))+' instances' 
        #return bestatt+',prob is 1'
    elif len(attributes)==1:   ##which is the first column - Class
        return prob(instances)
    else:
        #level 0
        ds=choosebestatt(attributes,instances)
        #print(ds)
        if ds!=False:
            bestatt=ds[0]
            besttrueins=ds[1]
            bestfalseins=ds[2]
            #print(besttrueins)
            #attributes.remove(bestatt)
            index=attributes.index(bestatt)
            sublist1=attributes[:index]
            sublist2=attributes[index+1:]
            subatt=sublist1+sublist2
            #print(subatt)
            #print(bestatt+' removed')
            #print(attributes)

            
            if(len(attributes)>0):
                if(besttrueins!=False):
                    lefttree=buildtree(besttrueins,subatt)
                if(bestfalseins!=False):
                    righttree=buildtree(bestfalseins,subatt)
        return bestatt,lefttree,righttree 
    


def readtree(learnedtree,s): 
    s=s+'|--'
    if not learnedtree[0]:
        print('')
        outputlines.append('')
    if type(learnedtree)==str:
        print(s+str(learnedtree))   
        outputlines.append(s+str(learnedtree))         
    else:
        print(s+str(learnedtree[0])+'=true')
        outputlines.append(s+str(learnedtree[0])+'=true')
        readtree(learnedtree[1],s)        
        print(s+str(learnedtree[0])+'=false')
        outputlines.append(s+str(learnedtree[0])+'=false')
        readtree(learnedtree[2],s)
   
learnedtree=buildtree(training_instances,training_attributes)  
print("The learned tree based on the training data is as below:")
outputlines.append("The learned tree based on the training data is as below:")
readtree(learnedtree,'')   



#predict test-data

#print(len(test_insts))



#predict test-data
def readtree(tree,input_insts):
    predictedclass=()
    if type(tree)==str:
        return tree
    else:
        #for i in range(len(input_insts)):
        att=tree[0]
        trueset=tree[1]
        falseset=tree[2]
        attindex=test_attributes.index(att)

        classlabel=input_insts[attindex]
        if classlabel=='true':
            predictedclass=readtree(trueset,input_insts)
        else:
            predictedclass=readtree(falseset,input_insts)
        return predictedclass


print('The predicted result is as below:')
outputlines.append(('The predicted result is as below:'))
prediction=[]
for i in range(len(test_insts)):
    result=readtree(learnedtree,test_insts[i])
    prediction.append(result)
    print('Test instance'+str(i)+' labeled Class:'+str(test_insts[i][0])+' predicted class:'+ str(result.rpartition(',')[0]))
    outputlines.append('Test instance'+str(i)+' labeled Class:'+str(test_insts[i][0])+' predicted class:'+ str(result.rpartition(',')[0]))
#caculating accuracy
#accuracy=0
m=0
for i in range(len(test_insts)):
    if test_insts[i][0]==str(prediction[i]).rpartition(',')[0]:
        m=m+1
accuracy=round(m/float(len(test_insts))*100,2)
print('Prediction accuracy is: '+str(accuracy)+'%')
outputlines.append('Prediction accuracy is: '+str(accuracy)+'%')


#calculating accuracy with baseline prediction
print('Baseline predictor is: '+ str(prob(training_instances)))
outputlines.append('Baseline predictor is: '+ str(prob(training_instances)))
baselinepred=str(prob(training_instances).rpartition(',')[0])
n=0
for i in range(len(test_insts)):
    if test_insts[i][0]==baselinepred:
        n=n+1

baselineacc=round(n/float(len(test_insts))*100)
print('Baseline predictor accuracy  is: '+str(baselineacc)+'%')
outputlines.append('Baseline predictor accuracy  is: '+str(baselineacc)+'%')

with open(file_dir+'/sampleoutput.txt','w')  as f:
    f.write('\n'.join(outputlines))







