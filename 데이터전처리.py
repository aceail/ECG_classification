def MEZOO_RR_interval(data, file_name, before, after):
    data_total = []
    p_signal = data.ECG_waveform.values
    rr = data[data.R_peak==1].index.values[:-1]
    rr_interval_ = data[data.R_peak==1].index.values[1:] - data[data.R_peak==1].index.values[:-1]
    
    for NUM in range(len(rr)-1):
        ann = []
        RR_ = []
        before_rr = rr[NUM] - int(rr_interval_[NUM]*(before))
        after_rr = rr[NUM] + int(rr_interval_[NUM+1]*(after))


        # RR범위 출력
        for i in [ann__ for ann__ in range(before_rr,after_rr)]:

            # RR 범위 RR 확인
            if len(np.where(rr == i)[0])>=1:
                RR_.append(len(np.where(rr == i)))

        data_dict = {
                     'before RR': f'{before_rr}', 
                     'RR': f'{rr[NUM]}',
                     'after RR': f'{after_rr}',
                     'diff': f'{after_rr-before_rr}',
                     'len_RR':f'{len(RR_)}'
                    }
        data_total.append(data_dict)

    return pd.DataFrame.from_dict(data_total)
 

def MEZOO_wfdb_RR_interval(data, file_name, before, after):
    data_total = []
    p_signal = data.ECG_waveform.values
    
    xqrs = processing.XQRS(sig=p_signal, fs=250)
    xqrs.detect()
    rr =  xqrs.qrs_inds[:-1]
    rr_interval_ = xqrs.qrs_inds[1:] - xqrs.qrs_inds[:-1]
    for NUM in range(len(rr)-1):
        ann = []
        RR_ = []
        before_rr = rr[NUM] - int(rr_interval_[NUM]*(before))
        after_rr = rr[NUM] + int(rr_interval_[NUM+1]*(after))

        # RR범위 출력
        for i in [ann__ for ann__ in range(before_rr,after_rr)]:

            # RR 범위 RR 확인
            if len(np.where(rr == i)[0])>=1:
                RR_.append(len(np.where(rr == i)))

        data_dict = {
                     'before RR': f'{before_rr}', 
                     'RR': f'{rr[NUM]}',
                     'after RR': f'{after_rr}',
                     'diff': f'{after_rr-before_rr}',
                     'len_RR':f'{len(RR_)}'
                    }
        data_total.append(data_dict)

    return pd.DataFrame.from_dict(data_total)
  
  
  
  
def MITDB_RR_interval(data_path, file_name, before, after):
    data_total = []
    for record in label_MLII:
        # 데이터 불러오기 MLII만
        sig, fields = wfdb.rdsamp('{}{}'.format(data_path, record))
        if 'MLII' in fields['sig_name']:
            if fields['sig_name'][0]=='MLII':
                p_signal = sig[:,0]
            else:
                p_signal = sig[:,1]

        # 데이터 Annotation   
        annotation = wfdb.rdann('{}{}'.format(data_path, record), 'atr')
        atr_sym1 = annotation.symbol
        atr_sample1 = annotation.sample

        # 데이터 PQRST and RR-inverval search
        xqrs = processing.XQRS(sig=p_signal, fs=fields['fs'])
        xqrs.detect()
        rr_interval_ = xqrs.qrs_inds[1:] - xqrs.qrs_inds[:-1]
        rr =  xqrs.qrs_inds[:-1]

        
        for NUM in range(len(rr)-1):
            ann = []
            RR_ = []
            before_rr = rr[NUM] - int(rr_interval_[NUM]*(before))
            after_rr = rr[NUM] + int(rr_interval_[NUM+1]*(after))


            # RR범위 출력
            for i in [ann__ for ann__ in range(before_rr,after_rr)]:

                # RR 범위 symbol 확인
                if len(np.where(annotation.sample == i)[0])>=1:
                    ann.append(annotation.symbol[np.where(annotation.sample == i)[0][0]])
                
                # RR 범위 RR 확인
                if len(np.where(rr == i)[0])>=1:
                    RR_.append(len(np.where(rr == i)))

            data_dict = {'record' : f'{record}', 
                         'before RR': f'{before_rr}', 
                         'RR': f'{rr[NUM]}',
                         'after RR': f'{after_rr}',
                         'diff': f'{after_rr-before_rr}',
                         'annotation': f'{ann}',
                         'len(annotation)': f'{len(ann)}',
                         'len_RR':f'{len(RR_)}'
                        }
            data_total.append(data_dict)

            #print('record : {} \t before RR: {} \t RR: {} \t after RR: {} \t diff: {} \t annotation: {}'.format(record, before_rr, rr[NUM] ,after_rr, after_rr-before_rr,ann))
    pd.DataFrame.from_dict(data_total).to_excel('{}.xlsx'.format(file_name),index=False)
    
    
    
  def MITDB_RR_interval_annotation_plot(record):
    record1 = wfdb.rdrecord('data/before/physionet.org/files/mitdb/1.0.0/{}'.format(record))
    annotation = wfdb.rdann('data/before/physionet.org/files/mitdb/1.0.0/{}'.format(record), 'atr', shift_samps=True)
    atr_sym1 = annotation.symbol
    atr_sample1 = annotation.sample
    if 'MLII' in record1.sig_name:
        if record1.sig_name[0]=='MLII':
            p_signal1 = record1.p_signal[:,0]
        else:
            p_signal1 = record1.p_signal[:,1]

        xqrs = processing.XQRS(sig=p_signal1, fs=record1.fs)
        xqrs.detect()

        before_rr = xqrs.qrs_inds[:-1]
        rr = xqrs.qrs_inds
        after_rr = xqrs.qrs_inds[1:]

        cc = after_rr - before_rr
        data = pd.DataFrame(p_signal1, columns=['signal'])

        data.loc[data.index.isin(atr_sample1), 'ann'] = atr_sym1
        data.loc[data.index.isin(xqrs.qrs_inds), 'rr'] = 1
        data.loc[~data.index.isin(xqrs.qrs_inds), 'rr'] = 0
        data.loc[data.index.isin(data[data.rr==1].index[1:]), 'rr_interval'] = data[data.rr==1].index[1:] - data[data.rr==1].index[:-1]   

        plt.figure(figsize=(20,5))
        plt.title("{} xqrs".format(i))
        ann_N = []
        ann_no = []
        for xq in xqrs.qrs_inds:
            for i in [ann__ for ann__ in range(xq-100,xq+100)]:
                if len(np.where(atr_sample1==i)[0])>=1:
                    if atr_sym1[np.where(atr_sample1==i)[0][0]] in ['N', 'L', 'R', 'B']:
                        if (xq-100> 0) & (xq+100<650000):
                            ann_N.append(atr_sym1[np.where(atr_sample1==i)[0][0]])
                            plt.subplot(3,3,1)
                            plt.title('normal class \n Non-Ectopic')
                            plt.plot(p_signal1[xq-100:xq+100])
                    elif atr_sym1[np.where(atr_sample1==i)[0][0]] in ['A', 'a', 'J', 'S', 'e','j','n']:
                        if (xq-100> 0) & (xq+100<650000):
                            ann_no.append(atr_sym1[np.where(atr_sample1==i)[0][0]])
                            plt.subplot(2,3,2)
                            plt.title('SupraVentricular Ectopic Beats (SVEBs)')
                            plt.plot(p_signal1[xq-100:xq+100])
                    elif atr_sym1[np.where(atr_sample1==i)[0][0]] in ['V', 'r']:
                        if (xq-100> 0) & (xq+100<650000):
                            ann_no.append(atr_sym1[np.where(atr_sample1==i)[0][0]])
                            plt.subplot(2,3,3)
                            plt.title('Ventricular Ectopic Beats (VEBs)')
                            plt.plot(p_signal1[xq-100:xq+100])
                    elif atr_sym1[np.where(atr_sample1==i)[0][0]] in ['F']:
                        if (xq-100> 0) & (xq+100<650000):
                            ann_no.append(atr_sym1[np.where(atr_sample1==i)[0][0]])
                            plt.subplot(2,3,4)
                            plt.title('Fusion Beats')
                            plt.plot(p_signal1[xq-100:xq+100])                            
                    elif atr_sym1[np.where(atr_sample1==i)[0][0]] in ['Q','?']:
                        if (xq-100> 0) & (xq+100<650000):
                            ann_no.append(atr_sym1[np.where(atr_sample1==i)[0][0]])
                            plt.subplot(2,3,5)
                            plt.title('Unrecognized')
                            plt.plot(p_signal1[xq-100:xq+100]) 
                    elif atr_sym1[np.where(atr_sample1==i)[0][0]] in ['f','/']:
                        if (xq-100> 0) & (xq+100<650000):
                            ann_no.append(atr_sym1[np.where(atr_sample1==i)[0][0]])
                            plt.subplot(2,3,6)
                            plt.title("Unmapped")
                            plt.plot(p_signal1[xq-100:xq+100]) 
        unique, counts= np.unique(atr_sym1, return_counts = True) 
        uniq_cnt_dict = dict(zip(unique, counts))
            
        unique1, counts1= np.unique(ann_N, return_counts = True) 
        uniq_cnt_dict1 = dict(zip(unique1, counts1))
        
        unique2, counts2= np.unique(ann_no, return_counts = True) 
        uniq_cnt_dict2 = dict(zip(unique2, counts2))
        print('ANN : {} \t N : {} \t NO : {} \t'.format(uniq_cnt_dict, uniq_cnt_dict1, uniq_cnt_dict2))
        # for xq in xqrs.qrs_inds:
        #     for num, samp in enumerate(atr_sample1):
        #         if samp in [ann__ for ann__ in range(xq-100,xq+100)]:
        #             if atr_sym1[num] == 'N':
        #                 if (xq-100> 0) & (xq+100<650000):
        #                     plt.plot(p_signal1[xq-100:xq+100])
        plt.tight_layout(h_pad=5, w_pad=8)
        #plt.savefig("classification_output_plot/ann6_{}.png".format(record_num), dpi=300)
        plt.show()
