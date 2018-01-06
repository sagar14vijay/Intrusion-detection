import csv
import random
import math
import pandas
import numpy as np
from time import time
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from nb import *

col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

num_features = ["duration","protocol_type","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files",
    "is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"]


#traing data preperation

training = pandas.read_csv("training", header=None, names = col_names)

labels = training['protocol_type']
labels[labels=='icmp'] = 1
labels[labels=='tcp'] = 2
labels[labels=='udp'] = 3

labels = training['flag']
labels[labels=='SF'] = 1
labels[labels=='REJ'] = 2
labels[labels=='S0'] = 3
labels[labels=='RSTO'] = 4
labels[labels=='RSTR'] = 5
labels[labels=='S3'] = 6
labels[labels=='SH'] = 7
labels[labels=='S1'] = 8
labels[labels=='S2'] = 9
labels[labels=='OTH'] = 10
labels[labels=='RSTOS0'] = 11

labels = training['label']
labels[labels=='back.'] = 'dos'
labels[labels=='buffer_overflow.'] = 'u2r'
labels[labels=='ftp_write.'] = 'r2l'
labels[labels=='guess_passwd.'] = 'r2l'
labels[labels=='imap.'] = 'r2l'
labels[labels=='ipsweep.'] = 'probe'
labels[labels=='land.'] = 'dos'
labels[labels=='loadmodule.'] = 'u2r'
labels[labels=='multihop.'] = 'r2l'
labels[labels=='neptune.'] = 'dos'
labels[labels=='nmap.'] = 'probe'
labels[labels=='perl.'] = 'u2r'
labels[labels=='phf.'] = 'r2l'
labels[labels=='pod.'] = 'dos'
labels[labels=='portsweep.'] = 'probe'
labels[labels=='rootkit.'] = 'u2r'
labels[labels=='satan.'] = 'probe'
labels[labels=='smurf.'] = 'dos'
labels[labels=='spy.'] = 'r2l'
labels[labels=='teardrop.'] = 'dos'
labels[labels=='warezclient.'] = 'r2l'
labels[labels=='warezmaster.'] = 'r2l'

labels = training['label']
labels[labels=='dos'] = '1'
labels[labels=='normal.'] = '2'
labels[labels=='probe'] = '3'
labels[labels=='r2l'] = '4'
labels[labels=='u2r'] = '5'

training_features = training[num_features]
training_label = training['label']


testing= pandas.read_csv("testing", header=None, names = col_names)

labels = testing['protocol_type']
labels[labels=='icmp'] = 1
labels[labels=='tcp'] = 2
labels[labels=='udp'] = 3

labels = testing['flag']
labels[labels=='SF'] = 1
labels[labels=='REJ'] = 2
labels[labels=='S0'] = 3
labels[labels=='RSTO'] = 4
labels[labels=='RSTR'] = 5
labels[labels=='S3'] = 6
labels[labels=='SH'] = 7
labels[labels=='S1'] = 8
labels[labels=='S2'] = 9
labels[labels=='OTH'] = 10
labels[labels=='RSTOS0'] = 11

labels = testing['label']
labels[labels=='back.'] = 'dos'
labels[labels=='buffer_overflow.'] = 'u2r'
labels[labels=='ftp_write.'] = 'r2l'
labels[labels=='guess_passwd.'] = 'r2l'
labels[labels=='imap.'] = 'r2l'
labels[labels=='ipsweep.'] = 'probe'
labels[labels=='land.'] = 'dos'
labels[labels=='loadmodule.'] = 'u2r'
labels[labels=='multihop.'] = 'r2l'
labels[labels=='neptune.'] = 'dos'
labels[labels=='nmap.'] = 'probe'
labels[labels=='perl.'] = 'u2r'
labels[labels=='phf.'] = 'r2l'
labels[labels=='pod.'] = 'dos'
labels[labels=='portsweep.'] = 'probe'
labels[labels=='rootkit.'] = 'u2r'
labels[labels=='satan.'] = 'probe'
labels[labels=='smurf.'] = 'dos'
labels[labels=='spy.'] = 'r2l'
labels[labels=='teardrop.'] = 'dos'
labels[labels=='warezclient.'] = 'r2l'
labels[labels=='warezmaster.'] = 'r2l'
labels[labels=='saint.'] = 'probe'
labels[labels=='mscan.'] = 'probe'
labels[labels=='apache2.'] = 'dos'
labels[labels=='udpstorm.'] = 'dos'
labels[labels=='processtable.'] = 'dos'
labels[labels=='mailbomb.'] = 'dos'
labels[labels=='xterm.'] = 'u2r'
labels[labels=='ps.'] = 'u2r'
labels[labels=='sqlattack.'] = 'u2r'
labels[labels=='snmpgetattack.'] = 'r2l'
labels[labels=='named.'] = 'r2l'
labels[labels=='xlock.'] = 'r2l'
labels[labels=='xsnoop.'] = 'r2l'
labels[labels=='sendmail.'] = 'r2l'
labels[labels=='httptunnel.'] = 'r2l'
labels[labels=='worm.'] = 'r2l'
labels[labels=='snmpguess.'] = 'r2l'

labels = testing['label']
labels[labels=='dos'] = '1'
labels[labels=='normal.'] = '2'
labels[labels=='probe'] = '3'
labels[labels=='r2l'] = '4'
labels[labels=='u2r'] = '5'

testing_features = testing[num_features]
testing_label = testing['label']



for i in range(50):
	print ""


print "***************Dataset Details***********"
print "\nFeatures are\n",num_features

print"\n************Training Dataset Details*********"
print"\n\Number of Training examples are 494021"
print"\nTypes of attacks and their percentages Dos(1),Normal(2),Probe(3),R2L(4),U2R(5) respectively are\n",training_label.value_counts()/494021*100

print"\n\n\n\n"

print"************Testing Dataset Details*********"
print"\nNumber of Testing examples are 311029"
print"\nTypes of attacks and their percentages Dos(1),Normal(2),Probe(3),R2L(4),U2R(5) respectively are\n",testing_label.value_counts()/311029*100

#making 10 subsubsets of training data

print"\n****Subsets Formation*******\n"
unit = 494021/10

subset1_features = training_features[0:unit]
subset1_label = training_label[0:unit]

subset2_features = training_features[unit:2*unit]
subset2_label = training_label[unit:2*unit]

subset3_features = training_features[2*unit:3*unit]
subset3_label = training_label[2*unit:3*unit]

subset4_features = training_features[3*unit:4*unit]
subset4_label = training_label[3*unit:4*unit]

subset5_features = training_features[4*unit:5*unit]
subset5_label = training_label[4*unit:5*unit]

subset6_features = training_features[5*unit:6*unit]
subset6_label = training_label[5*unit:6*unit]

subset7_features = training_features[6*unit:7*unit]
subset7_label = training_label[6*unit:7*unit]

subset8_features = training_features[7*unit:8*unit]
subset8_label = training_label[7*unit:8*unit]

subset9_features = training_features[8*unit:9*unit]
subset9_label = training_label[8*unit:9*unit]

subset10_features = training_features[9*unit:10*unit]
subset10_label = training_label[9*unit:10*unit]


#training 10 tree classiefiers

print"\n****Tree Classifiers Training*******"

clf1 = tree.DecisionTreeClassifier()
clf1.fit(subset1_features,subset1_label)


clf2 = tree.DecisionTreeClassifier()
clf2.fit(subset2_features,subset2_label)

clf3 = tree.DecisionTreeClassifier()
clf3.fit(subset3_features,subset3_label)

clf4 = tree.DecisionTreeClassifier()
clf4.fit(subset4_features,subset4_label)

clf5 = tree.DecisionTreeClassifier()
clf5.fit(subset5_features,subset5_label)

clf6 = tree.DecisionTreeClassifier()
clf6.fit(subset6_features,subset6_label)

clf7 = tree.DecisionTreeClassifier()
clf7.fit(subset7_features,subset7_label)

clf8 = tree.DecisionTreeClassifier()
clf8.fit(subset8_features,subset8_label)

clf9 = tree.DecisionTreeClassifier()
clf9.fit(subset9_features,subset9_label)

clf10 = tree.DecisionTreeClassifier()
clf10.fit(subset10_features,subset10_label)

#predicting on training datasubset
print"\n****Predictions on Training data by Tree Classifier*******"
pred1 = clf1.predict(training_features)
pred2 = clf2.predict(training_features)
pred3 = clf3.predict(training_features)
pred4 = clf4.predict(training_features)
pred5 = clf5.predict(training_features)
pred6 = clf6.predict(training_features)
pred7 = clf7.predict(training_features)
pred8 = clf8.predict(training_features)
pred9 = clf9.predict(training_features)
pred10 = clf10.predict(training_features)

#making a table of all training predictions
table = np.empty((494021,10),dtype='float')
table[0:: ,0] = pred1
table[0:: ,1] = pred2
table[0:: ,2] = pred3
table[0:: ,3] = pred4
table[0:: ,4] = pred5
table[0:: ,5] = pred6
table[0:: ,6] = pred7
table[0:: ,7] = pred8
table[0:: ,8] = pred9
table[0:: ,9] = pred10

table2 = np.empty((494021,11),dtype='float')
table2[0:: ,0] = pred1
table2[0:: ,1] = pred2
table2[0:: ,2] = pred3
table2[0:: ,3] = pred4
table2[0:: ,4] = pred5
table2[0:: ,5] = pred6
table2[0:: ,6] = pred7
table2[0:: ,7] = pred8
table2[0:: ,8] = pred9
table2[0:: ,9] = pred10
table2[0:: ,10] = training_label

#Training naive bias classifier
print"\n****Training Naive Bias Classifier*******\n"
gnb = GaussianNB()
gnb.fit(table,training_label)
print"\n****Training Naive Bias Classifier ended*******\n"

#predictions for testing data using tree classifier
print"\n****Predictions on Testing data by Tree Classifier*******\n"
pred1 = clf1.predict(testing_features)
pred2 = clf2.predict(testing_features)
pred3 = clf3.predict(testing_features)
pred4 = clf4.predict(testing_features)
pred5 = clf5.predict(testing_features)
pred6 = clf6.predict(testing_features)
pred7 = clf7.predict(testing_features)
pred8 = clf8.predict(testing_features)
pred9 = clf9.predict(testing_features)
pred10 = clf10.predict(testing_features)

#making a table of all training predictions
table = np.empty((311029,10),dtype='float')
table[0:: ,0] = pred1
table[0:: ,1] = pred2
table[0:: ,2] = pred3
table[0:: ,3] = pred4
table[0:: ,4] = pred5
table[0:: ,5] = pred6
table[0:: ,6] = pred7
table[0:: ,7] = pred8
table[0:: ,8] = pred9
table[0:: ,9] = pred10

table3 = np.empty((311029,11),dtype='float')
table3[0:: ,0] = pred1
table3[0:: ,1] = pred2
table3[0:: ,2] = pred3
table3[0:: ,3] = pred4
table3[0:: ,4] = pred5
table3[0:: ,5] = pred6
table3[0:: ,6] = pred7
table3[0:: ,7] = pred8
table3[0:: ,8] = pred9
table3[0:: ,9] = pred10
table3[0:: ,10] = testing_label


print"\n****Predictions on Testing data by Naive Bias*******\n"
predictions = gnb.predict(table)
print "final predictions on testing data using NB"
print(predictions)

#finding accuracy
print "\nAccuracy"
print(100*naivebayes(table2,table3))

acc = accuracy_score(testing_label,predictions)
print(acc*100)

