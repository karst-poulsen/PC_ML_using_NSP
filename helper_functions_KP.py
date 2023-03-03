import re

import matplotlib.pyplot as plt

# from interpro_scraping import interpro_scraping_pandas
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import numpy as np
import scipy.stats
from datetime import datetime
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def feat_elim_rand(df,labels,out_name,feats,estimator, step):


    estimator = RandomForestRegressor(n_estimators=estimator)
    selector = RFE(estimator, n_features_to_select=feats, step=step)
    selector = selector.fit(df, labels)
    selector.support_
    feat_list = selector.get_feature_names_out()
    a = pd.DataFrame(list(zip(df.columns, selector.ranking_)), columns=['feature', 'selector rankings'])
    a.to_excel('Output_data/FeatSelection'+str(out_name)+'.xlsx')
    return df[feat_list]

def rand_forest_reg_fit(df,labels,out_name,test_size,estimator):
    x_train, x_test, y_train, y_test = train_test_split(df, labels,
                                                        test_size=test_size,
                                                        random_state=42)
    rfg = RandomForestRegressor(n_estimators=estimator)
    rfg.fit(x_train, y_train)
    feat_importances = rfg.feature_importances_
    b=list(zip(df.columns, feat_importances * 100))
    score=rfg.score(x_test,y_test)
    # a = pd.DataFrame(b, columns=['feature', 'feat_importance'])
    # a.to_excel("Output_data/Featimportances"+str(out_name)+".xlsx")
    return b, score


if __name__ == "__main__":
    print(type(ProteinAnalysis))
    print('found biopython')