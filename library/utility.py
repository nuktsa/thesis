import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from random import randrange
from scipy import interpolate
from tqdm import tqdm

class temporal:
    def __init__(self, n = 100, ts = 1, progress = False):#, self.ts = 1):
        self.n = n
        self.ts = ts
        self.progress = progress
        # self.ts = self.ts

    def extract(self, data):
        p = int(len(data)/self.n)
        data = data[:p * self.n]
        # return self._all_temporal_extract(data, self.n)#, self.ts)
        extracted_all = pd.DataFrame()
        if data.ndim != 1 :
            idx = data.columns.values
            for i in range(data.shape[1]):
                df = data.iloc[:,i]
                extracted = self._temporal_extract(df, self.n, idx[i])#, self.ts)# self.ts,
                extracted_all = pd.concat([extracted_all, extracted], axis=1)
        else :
            idx = data.columns.values[0]
            extracted = self._temporal_extract(data, self.n, idx)#, self.ts) #, self.ts
            extracted_all = extracted#pd.concat([extracted_all, extracted], axis=1, ignore_index=True)
        return extracted_all

    def _temporal_extract(self, data, n, idx):#, self.ts) :#, self.ts
        extracted = pd.DataFrame()
        if self.progress :
            print(f'\n{idx} Timeseries Feature Extraction')
            for i in tqdm(range(0,len(data), n)) :
                df = data.loc[i:i+n-1]
                extract_feat = pd.DataFrame(self._feat(df)).transpose()#, self.ts
                extracted = pd.concat([extracted, extract_feat], ignore_index=True)
        else :
            for i in range(0,len(data), n) :
                df = data.loc[i:i+n-1]
                extract_feat = pd.DataFrame(self._feat(df)).transpose()#, self.ts
                extracted = pd.concat([extracted, extract_feat], ignore_index=True)
        if self.ts :
            extracted.columns = [f'{idx} Auto Correlation', f'{idx} Mean Absolute Differences', f'{idx} Mean Differences', 
                                f'{idx} Median Absolute Differences', f'{idx} Median Differences', f'{idx} Signal Distance', 
                                f'{idx} Sum of Absolute Differences', f'{idx} Entropy', 
                                f'{idx} Peak to Peak Distance', f'{idx} Absolute Energy',
                                f'{idx} Neighbourhood Peaks', f'{idx} Negative Turning Point', f'{idx} Positive Turning Point', 
                                f'{idx} Zero Crossing Rate', f'{idx} Slope', f'{idx} Centroid',  f'{idx} Total Energy', f'{idx} Area Under Curve']
        else :
            
            extracted.columns = [f'{idx} Auto Correlation', f'{idx} Mean Absolute Differences', f'{idx} Mean Differences', 
                                f'{idx} Median Absolute Differences', f'{idx} Median Differences', f'{idx} Signal Distance', 
                                f'{idx} Sum of Absolute Differences', f'{idx} Entropy', 
                                f'{idx} Peak to Peak Distance', f'{idx} Absolute Energy',
                                f'{idx} Neighbourhood Peaks', f'{idx} Negative Turning Point', f'{idx} Positive Turning Point', 
                                f'{idx} Zero Crossing Rate', f'{idx} Slope']
        extracted = extracted.sort_index(axis=1)
        return extracted

    def _feat(self, df): #, self.ts
        if self.ts :
            features = [self._autocorr(df), self._mean_abs_diff(df), self._mean_diff(df), self._median_abs_diff(df), self._median_diff(df), 
                        self._distance(df), self._sum_abs_diff(df), self._entropy(df), self._pk_pk_distance(df), self._abs_energy(df), 
                        self._neighbourhood_peaks(df), self._negative_turning(df), self._positive_turning(df), self._zero_cross(df), self._slope(df), 
                        self._calc_centroid(df, self.ts), self._total_energy(df, self.ts), self._auc(df, self.ts)]
        else :
            features = [self._autocorr(df), self._mean_abs_diff(df), self._mean_diff(df), self._median_abs_diff(df), self._median_diff(df), 
                        self._distance(df), self._sum_abs_diff(df), self._entropy(df), self._pk_pk_distance(df), self._abs_energy(df), 
                        self._neighbourhood_peaks(df), self._negative_turning(df), self._positive_turning(df), self._zero_cross(df), self._slope(df)]            
        return features
    def _compute_time(self, signal, ts):
        return np.arange(0, len(signal))*ts

    def _autocorr(self, signal):
        signal = np.array(signal)
        return float(np.correlate(signal, signal))

    def _calc_centroid(self, signal, ts):
        time = self._compute_time(signal, ts)

        energy = np.array(signal) ** 2

        t_energy = np.dot(np.array(time), np.array(energy))
        energy_sum = np.sum(energy)

        if energy_sum == 0 or t_energy == 0:
            centroid = 0
        else:
            centroid = t_energy / energy_sum

        return centroid

    def _mean_abs_diff(self, signal): #mean absolute diff
        return np.mean(np.abs(np.diff(signal)))

    def _mean_diff(self, signal): #mean diff
        return np.mean(np.diff(signal))

    def _median_abs_diff(self, signal):#Median absolute difference
        return np.median(np.abs(np.diff(signal)))

    def _median_diff(self, signal):#Median difference
        return np.median(np.diff(signal))

    def _distance(self, signal):#Signal _distance
        diff_sig = np.diff(signal).astype(float)
        return np.sum([np.sqrt(1 + diff_sig ** 2)])

    def _sum_abs_diff(self, signal):#Sum absolute difference
        return np.sum(np.abs(np.diff(signal)))
        
    def _total_energy(self, signal, ts):#total energy
        time = self._compute_time(signal, ts)

        return np.sum(np.array(signal) ** 2) / (time[-1] - time[0])

    def _entropy(self, signal): #_entropy , prob='standard'

        value, counts = np.unique(signal, return_counts=True)
        p = counts / counts.sum()

        if np.sum(p) == 0:
            return 0.0

        # Handling zero probability values
        p = p[np.where(p != 0)]

        # If probability all in one value, there is no _entropy
        if np.log2(len(signal)) == 1:
            return 0.0

        elif np.sum(p * np.log2(p)) / np.log2(len(signal)) == 0:
            return 0.0

        else:
            return - np.sum(p * np.log2(p)) / np.log2(len(signal))

    def _pk_pk_distance(self, signal):#peak to peak _distance
        return np.abs(np.max(signal) - np.min(signal))

    def _auc(self, signal, ts):#The area under the curve value
        t = self._compute_time(signal, ts)

        return np.sum(0.5 * np.diff(t) * np.abs(np.array(signal[:-1]) + np.array(signal[1:])))

    def _abs_energy(self, signal):#Absolute energy
        return np.sum(np.abs(signal) ** 2)

    def _neighbourhood_peaks(self, signal, n=10):
        signal = np.array(signal)
        subsequence = signal[n:-n]
        # initial iteration
        peaks = ((subsequence > np.roll(signal, 1)[n:-n]) & (subsequence > np.roll(signal, -1)[n:-n]))
        for i in range(2, n + 1):
            peaks &= (subsequence > np.roll(signal, i)[n:-n])
            peaks &= (subsequence > np.roll(signal, -i)[n:-n])
        return np.sum(peaks)

    def _negative_turning(self, signal):
        diff_sig = np.diff(signal)
        array_signal = np.arange(len(diff_sig[:-1]))
        _negative_turning_pts = np.where((diff_sig[array_signal] < 0) & (diff_sig[array_signal + 1] > 0))[0]

        return len(_negative_turning_pts)

    def _positive_turning(self, signal):
        diff_sig = np.diff(signal)

        array_signal = np.arange(len(diff_sig[:-1]))

        _positive_turning_pts = np.where((diff_sig[array_signal + 1] < 0) & (diff_sig[array_signal] > 0))[0]

        return len(_positive_turning_pts)

    def _zero_cross(self, signal):
        return len(np.where(np.diff(np.sign(signal)))[0])

    def _slope(self, signal):
        t = np.linspace(0, len(signal) - 1, len(signal))

        return np.polyfit(t, signal, 1)[0]

class confussion:
    def __init__(self):
        pass

    def matrix(self, y_real, y_pred, lab_pos = 1, lab_neg = 0, name = 'confussion matrix', metric = False, plot = True, save=False):
        y_real = np.asarray(y_real)
        y_pred = np.asarray(y_pred)

        true = y_pred[np.where(y_real == y_pred)]
        false = y_pred[np.where(y_real != y_pred)]

        TP = np.where(true == lab_pos)[0].size #true positive
        TN = np.where(true == lab_neg)[0].size #true negative
        FP = np.where(false == lab_pos)[0].size #false positive
        FN = np.where(false == lab_neg)[0].size #false negative

        TA = np.where(y_real == lab_pos)[0].size #true actual
        FA = np.where(y_real == lab_neg)[0].size #false actual
        n = TA + FA #total sample

        TPF = TP/TA #sensitivity/true positive fraction
        TNF = TN/FA #specificitytrue negative fraction
        FPF = 1 - TNF #false positive fraction
        FNF = 1 - TPF #false negative fraction

        if (TP + FP) == 0 or (TP + FN) == 0 :
            precision = -1
            f1 = -1
        else :
            precision = TP/(TP + FP) * 100
        
        accuracy = (TP + TN)/n * 100
        recall = TP/(TP + FN) * 100

        if (precision + recall) == 0 or precision == -1:
            f1 = -1
        
        else :
            f1 = 2 * precision * recall/(precision + recall)

        matrix = pd.DataFrame([[TN, FP], [FN, TP]], columns = [f'{lab_neg}', f'{lab_pos}'], index = [f'{lab_neg}', f'{lab_pos}'])
        #evaluation metric
        eval = pd.DataFrame([['True Positive', '   False Positive', '   True Negative', '   False Negative'], [TP, FP, TN, FN], 
                            ['', '', '', ''], ['TP Fraction', 'FP Fraction', 'TN Fraction', 'FN Fraction'], [TPF, FPF, TNF, FNF], 
                            ['', '', '', ''], ['Accuracy', 'Precision', 'Recall', 'f1 score'], [accuracy, precision, recall, f1]])
        eval.style.set_properties(**{'text-align': 'center'})
        eval1 = eval.to_string(index=False, header=False)

        if plot :
            #plot confussion matrix
            plt.figure(dpi=100, figsize=(5,5))
            ax= sns.heatmap(matrix,  cbar=False, cmap="BuGn", annot=True, fmt="d")
            plt.setp(ax.get_xticklabels(), rotation=0)
            plt.ylabel('Label Sebenarnya', fontweight='bold', fontsize = 10)
            plt.xlabel('Label Prediksi', fontweight='bold', fontsize = 10)
            plt.title('Confussion Matrix', fontweight='bold', fontsize = 15)
            plt.tight_layout()
            if save :
                plt.savefig(f'{name}.jpg')
                eval.to_csv(f'{name}.csv', index = False)

            print('\n', eval1, '\n')
            plt.show()            

        if metric :
            return FPF, TPF, accuracy, precision, recall, f1
    
        return FPF, TPF

class roc_auc:
    def __init__(self):
        pass

    def score(self, y_real, y_prob, name = 'roc auc', plot = True, shade = True, auc_score = True, save = False, return_value = False):
        thr = np.linspace(0, 1, 1000)

        points = []
        for i in thr :
            y_thr = []
            for j in y_prob[:,1] :
                if j > i :
                    y_thr.append(1)
                else :
                    y_thr.append(0)
            fpf, tpf = confussion().matrix(y_real = y_real, y_pred = y_thr, plot = False)
            point = np.asarray([fpf, tpf])
            points.append(point)
        points = np.asarray(points)

        roc_point = []
        for k in np.unique(points[:,0]):
            l = np.where(points[:, 0] == k)[0]
            fpf2 = np.mean(points[:, 0][l])
            tpf2 = np.mean(points[:, 1][l])#points[:,1][l[-1]]
            roc_point.append(np.asarray([fpf2, tpf2]))
        roc_point = np.asarray(roc_point)

        roc_point = roc_point[roc_point[:, 0].argsort()]
        x = np.append(np.insert(roc_point[:, 0], 0, 0), 1)
        y = np.append(np.insert(roc_point[:, 1], 0, 0), 1)

        auc = np.trapz(y, x)

        if plot :
            plt.figure(dpi=100, figsize=(5,5))
            plt.plot(x, y)
            plt.ylabel('True Positive Fraction', fontweight='bold', fontsize = 10)
            plt.xlabel('False Positive Fraction', fontweight='bold', fontsize = 10)
            plt.title('ROC Curve', fontweight='bold', fontsize = 15)

            if shade :
                plt.fill_between(x, y, color = 'orange')

            if auc_score :
                plt.text(0.37, 0.5, f'AUC : {round(auc, 3)}', fontweight='bold', fontsize = 13)
            
            plt.tight_layout()

            if save :
                plt.savefig(f'{name}.jpg')
                rep = pd.DataFrame({'FPF' : x, 'TPF' : y})
                rep.to_csv(f'{name}.csv', index = False)
        
            plt.show()

        if return_value :
            return [x, y, auc]

        return auc

class kFold:
    def __init__(self):
        pass

    def _cv_split(self, data, folds):
            data_split = []
            fold_size = int(data.shape[0] / folds)
            
            # for loop to save each fold
            for i in range(folds):
                fold = []
                # while loop to add elements to the folds
                while len(fold) < fold_size:
                    # select a random element
                    r = randrange(data.shape[0])
                    # determine the index of this element 
                    index = data.index[r]
                    # save the randomly selected line 
                    fold.append(data.loc[index].values.tolist())
                    # delete the randomly selected line from
                    # dataframe not to select again
                    data = data.drop(index)
                # save the fold     
                data_split.append(np.asarray(fold))
                
            return data_split

    def cv(self, df, model, k, lab_pos = 1, lab_neg = 0) :
        df = self._cv_split(df, k)
        accuracy_tr = np.asarray([])
        accuracy_tt = np.asarray([])

        for i in range(k):
            data = df.copy()
            test = data.pop(i)
            train = np.asarray(data)
            train = train.reshape(train.shape[0] * train.shape[1], train.shape[2])
            X_tr = train[:, :-1]
            y_tr = train[:, -1].astype(int)
            X_tt = test[:, :-1]
            y_tt = test[:, -1].astype(int)

            model.fit(X_tr, y_tr)

            y_trp = model.predict(X_tr)
            y_ttp = model.predict(X_tt)

            acc_tr = confussion().matrix(y_tr, y_trp, lab_pos, lab_neg, plot=False, metric = True)[2]
            acc_tt = confussion().matrix(y_tt, y_ttp, lab_pos, lab_neg, plot=False, metric = True)[2]

            accuracy_tr = np.append(accuracy_tr, acc_tr)
            accuracy_tt = np.append(accuracy_tt, acc_tt)

        avg_acc_tr = np.average(accuracy_tr)
        avg_acc_tt = np.average(accuracy_tt)

        return accuracy_tr, avg_acc_tr, accuracy_tt, avg_acc_tt