data_X = []
data_y = []
base=''
for i in tqdm(subdata[['record','before RR', 'after RR', 'annotation']].values):
    record_ = i[0]
    before_ = i[1]
    after_ = i[2]
    ann_ = i[3]
    
    if base!=record_:
        record1 = wfdb.rdrecord('data/before/physionet.org/files/mitdb/1.0.0/{}'.format(record_))
        base=record_
        plt.show()
        
    if 'MLII' in record1.sig_name:
        if record1.sig_name[0]=='MLII':
            p_signal1 = record1.p_signal[:,0]
        else:
            p_signal1 = record1.p_signal[:,1]
            
            
        if ann_ in custom_ants:
            data_X.append(minmax_scale(signal.resample(p_signal1[before_:after_], 200)))
            data_y.append(ann_)
    # #plt.plot(p_signal1[before_:after_])
    # if ann_ =='N':
    #     plt.plot(minmax_scale(signal.resample(p_signal1[before_:after_], 200)))
X_data = np.array(data_X)
y_data = np.array(data_y)
y_data = pd.get_dummies(y_data, dummy_na = True)[pd.get_dummies(y_data, dummy_na = True).columns[:-1]]

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=721)



mezoo_val = []

for i in tqdm(mezoo__[['before RR', 'after RR']].values):
    before_ = int(i[0])
    after_ = int(i[1])
    mezoo_.ECG_waveform.values[before_:after_]
    
    mezoo_val.append(minmax_scale(signal.resample(mezoo_.ECG_waveform.values[before_:after_], 200)))

X_mezoo_val = np.array(mezoo_val)
X_mezoo_val = X_mezoo_val.reshape(X_mezoo_val.shape + (1,1,))
