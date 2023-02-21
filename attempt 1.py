import pandas as pd
import numpy as np
from astropy.stats import median_absolute_deviation

fft_size = 300
overlap = 0.5
hop = int((fft_size) * (1-overlap))
train = 'train'
len_train_data = 619780

def feat_extract(sensor_name, fft_size=fft_size, hop=hop, train = train):
	train_feat = []
	c = 5999 - fft_size

	sensor_data = pd.read_csv(train+'_'+sensor_name+'.csv', header =None)

	train_data = sensor_data.loc[:, 0:fft_size-1]
	for j in range(hop, c, hop):
		train_data = np.r_[train_data, sensor_data.loc[:, j:j+fft_size-1]]
	print(train_data.shape)
	print(sensor_name+ ' read done')
	'''
	train_feat.append(np.mean(train_data, axis=1))
	print(sensor_name+ ' mean done')
	train_feat.append(np.std(train_data, axis=1))
	print(sensor_name+ ' std done')
	train_feat.append(np.var(train_data, axis = 1)) 
	print(sensor_name+ ' var done')
	train_feat.append(np.max(train_data, axis = 1))
	print(sensor_name+ ' max done')
	train_feat.append(np.min(train_data, axis = 1))
	print(sensor_name+ ' min done')
	train_feat.append(median_absolute_deviation(train_data, axis =1))
	print(sensor_name+ ' mad done')
	train_feat = np.array(train_feat)
	return train_feat.transpose()
	'''
	return train_data
'''
jerk = np.array(['','_jerk'])
sensor = np.array(['Acc','Gyr', 'Gra', 'LAcc', 'Mag'])
axis = np.array(['_x', '_y', '_z', '_mag'])
ori_axis = np.array(['_x', '_y', '_z', '_w'])

train_features = np.zeros(len_train_data)

for m in range(len(jerk)):
	for n in range(len(sensor)):
		for i in range(len(axis)):
			train_feat = feat_extract(sensor[n]+jerk[m]+axis[i])
			train_features = np.c_[train_features, train_feat]

for m in range(len(jerk)):
	for i in range(len(axis)):
		train_feat = feat_extract('Ori'+jerk[m]+ori_axis[i])
		train_features = np.c_[train_features, train_feat]

for m in range(len(jerk)):
	train_feat = feat_extract('Pressure'+jerk[m])
	train_features = np.c_[train_features, train_feat]

print(train_features.shape)
np.savetxt('features_'+train+'.csv', train_features[:, 1:], delimiter=",", fmt ='%.3f')
print('features_'+train+' done')
'''

'''

train_y = pd.read_csv(train+'_labels.csv', header =None)
train_label = train_y
for i in range ((6000/hop)-3):
	train_label = np.r_[train_label,train_y]
train_label = train_label[:,0]
#train_label = train_label.values
print(train_label.shape)
#np.savetxt('labels_'+train+'.csv', np.array(train_label), delimiter=",", fmt ='%.1f')
'''

jerk = np.array(['','_jerk'])
sensor = np.array(['Acc','Gyr', 'Gra', 'LAcc', 'Mag'])
axis = np.array(['_x', '_y', '_z', '_mag'])
ori_axis = np.array(['_x', '_y', '_z', '_w'])

#train_features = np.zeros(len_train_data)

for m in range(len(jerk)):
	for n in range(len(sensor)):
		for i in range(len(axis)):
			train_feat = feat_extract(sensor[n]+jerk[m]+axis[i])

			#train_features = np.c_[train_features, train_feat]

for m in range(len(jerk)):
	for i in range(len(axis)):
		train_feat = feat_extract('Ori'+jerk[m]+ori_axis[i])
		#train_features = np.c_[train_features, train_feat]

for m in range(len(jerk)):
	train_feat = feat_extract('Pressure'+jerk[m])
	#train_features = np.c_[train_features, train_feat]

#print(train_features.shape)
#np.savetxt('features_'+train+'.csv', train_features[:, 1:], delimiter=",", fmt ='%.3f')
#print('features_'+train+' done')