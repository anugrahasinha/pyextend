import numpy as np
from sklearn.metrics import auc,roc_curve,confusion_matrix


class PyMetricsExtend(object):
    def __init__(self):
        pass
    
    def cm2aucBinomial(self,cm):
        '''
        Description : 
        In sklearn, there is no way to generate AUC value if you just have an old confusion matrix data
        This function is for binomial classification result (confusion matrix) which can be used to 
        generate AUC value. This is a lame function, if you have an old confusion matrix data and you 
        would like to generate the AUC value from the confusion matrix.
        
        Args :
        cm = sklearn.metrics.confusion_matrix output      
        '''
        try:
            if type(cm) == np.ndarray and cm.shape[0] == 2 and cm.shape[1] == 2:
                actual = np.hstack((np.repeat(0,cm[0][0] + cm[0][1]),np.repeat(1,cm[1][0] + cm[1][1])))
                pred = np.hstack((np.repeat(0,cm[0][0]),np.repeat(1,cm[0][1]),np.repeat(0,cm[1][0]),np.repeat(1,cm[1][1])))
                print("Actual confusion matrix:\n%s\n\nRegenerated Confusion Matrix:\n%s" %(str(cm),str(confusion_matrix(actual,pred))))
                fpr,tpr,threshold = roc_curve(actual,pred)
                auc_cal = auc(fpr,tpr)
                print("AUC = %f" %(auc_cal))
                return auc_cal
            else:
                raise(Exception(str("PyMetricExtend Exception : Invalid input provided for cm2aucBinomial")))
        except Exception as e:
            raise(e)