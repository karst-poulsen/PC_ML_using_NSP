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
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import d2_tweedie_score


def feat_elim_rand(df, labels, out_name, feats, estimator, step):
    estimator = RandomForestRegressor(n_estimators=estimator)
    selector = RFE(estimator, n_features_to_select=feats, step=step)
    selector = selector.fit(df, labels)
    selector.support_
    feat_list = selector.get_feature_names_out()
    a = pd.DataFrame(list(zip(df.columns, selector.ranking_)), columns=['feature', 'selector rankings'])
    a.to_excel('Output_data/FeatSelection' + str(out_name) + '.xlsx')
    return df[feat_list]


def rand_forest_reg_fit(df, labels, out_name, test_size, estimator):
    x_train, x_test, y_train, y_test = train_test_split(df, labels,
                                                        test_size=test_size,
                                                        random_state=42)
    rfg = RandomForestRegressor(n_estimators=estimator)
    rfg.fit(x_train, y_train)
    feat_importances = rfg.feature_importances_
    b = list(zip(df.columns, feat_importances * 100))
    score = rfg.score(x_test, y_test)
    # a = pd.DataFrame(b, columns=['feature', 'feat_importance'])
    # a.to_excel("Output_data/Featimportances"+str(out_name)+".xlsx")
    return b, score


def scram_score(df, label, model, identifier, test_percent):
    id = identifier
    feats = []
    r2s = []
    pearson = []
    mse = []
    accuracy = []
    x_train, x_test, y_train, y_test = train_test_split(df, label, test_size=test_percent, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    r2 = r2_score(y_test, predictions)
    corr, _ = pearsonr(y_test, predictions)
    mse_score = mean_squared_error(y_test, predictions)
    acc_score = accuracy_score(y_test, predictions.round())
    r2s.append(r2)
    pearson.append(corr)
    mse.append(mse_score)
    accuracy.append(acc_score)
    feat_import = model.feature_importances_
    feats.append('allfeats')
    feat_import = np.insert(feat_import, [0], 0)
    for j in x_test.columns:
        tmp = x_test.copy()
        # print(x_test)
        np.random.shuffle(tmp[j].values)
        scram_score = model.score(tmp, y_test)
        predictions = model.predict(tmp)
        r2 = r2_score(y_test, predictions)
        corr, _ = pearsonr(y_test, predictions)
        mse_score = mean_squared_error(y_test, predictions)
        acc_score = accuracy_score(y_test, predictions.round())
        r2s.append(r2)
        pearson.append(corr)
        feats.append(j)
        mse.append(mse_score)
        accuracy.append(acc_score)
    a = pd.DataFrame(list(zip(feats, pearson, r2s, feat_import)), columns=['feat', 'pearson', 'R2', 'importances'])
    a.to_excel("Output_data/scram_loss_feats" + id + ".xlsx")

    fig, ax = plt.subplots()
    plt.xticks(rotation=90)
    ax.plot(feats, pearson, color='tab:blue', marker='o', label='pearson')
    ax.plot(feats, r2s, color='tab:red', marker='s', label='R2')
    ax.plot(feats, mse, color='tab:green', marker='^', label='neg mse')
    ax.plot(feats, accuracy, color='tab:orange', marker='*', label='accuracy')
    ax.plot(feats, feat_import, color='tab:purple', marker='x', label='feature importance')
    ax.legend()

    plt.title('Score as a function of feature scrambling\n'+id)
    plt.savefig('Output_data/FeatScramLoss' + id + '.png', bbox_inches='tight')
    plt.close('all')
    print('Scramble Scoring ran successfully')

def feat_drop(df, label, model, identifier, test_percent):
    id = identifier
    feats = []
    r2s = []
    pearson = []
    mse = []
    accuracy = []
    x_train, x_test, y_train, y_test = train_test_split(df, label, test_size=test_percent, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    r2 = r2_score(y_test, predictions)
    corr, _ = pearsonr(y_test, predictions)
    mse_score = mean_squared_error(y_test, predictions)
    acc_score = accuracy_score(y_test, predictions.round())
    a = list(zip(model.feature_importances_, model.feature_names_in_))
    a.sort(reverse=True)
    feat_import = model.feature_importances_
    col_import = pd.DataFrame(a, columns=['importances', 'names'])
    sorted_cols = col_import['names']
    feat_import = np.insert(feat_import, [0], 0)
    feat_import = np.delete(feat_import, [-1])
    feats.append('All Feats')
    r2s.append(r2)
    pearson.append(corr)
    mse.append(mse_score)
    accuracy.append(acc_score)

    df_3 = df.copy()
    for i in sorted_cols:
        if i == sorted_cols.iloc[-1]:
            break
        # df_3=df_2.copy() #remove if you only want to drop each feature instead of dropping one feature at a time
        df_3.drop(columns=[i], inplace=True)
        x_train, x_test, y_train, y_test = train_test_split(df_3, label, test_size=test_percent)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        r2 = r2_score(y_test, predictions)
        corr, _ = pearsonr(y_test, predictions)
        mse_score = mean_squared_error(y_test, predictions)
        acc_score = accuracy_score(y_test, predictions.round())
        r2s.append(r2)
        pearson.append(corr)
        mse.append(mse_score)
        accuracy.append(acc_score)
        feats.append(i)

    # df_out = pd.DataFrame(list(zip(feats, pearson, r2s, mse, accuracy)), columns=['dropped feat', 'Pearson', 'r2', 'neg_mse', 'accuracy'])
    # df_out.to_excel("Output_data/feat_drop_cumulative" + id + ".xlsx")
    plt.close('all')
    fig, ax = plt.subplots()
    plt.xticks(rotation=90)
    ax.plot(feats, pearson, color='tab:blue', marker='o',label='pearson')
    ax.plot(feats, r2s, color='tab:red', marker='s',label='R2')
    ax.plot(feats, mse, color='tab:green', marker='^',label='neg mse')
    ax.plot(feats, accuracy, color='tab:orange', marker='*',label='accuracy')
    ax.plot(feats, feat_import, color='tab:purple', marker='x',label='feature importance')
    ax.legend()

    plt.title('Score as a function of Feat Drop\n'+id)
    plt.savefig('Output_data/feat_drop_cumulative' + id + '.png', bbox_inches='tight')
    plt.close('all')
    print('feat drop ran successfully')



def scorer(df, label, model, identifier, folds):
    from sklearn.model_selection import StratifiedKFold
    id=identifier
    y = label
    X = df

    # Initialize lists to store Pearson and R2 scores for each fold
    pearson_scores = []
    r2_scores = []
    mse = []
    # Split your data into 10 folds using KFold
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Create a 2x1 subplot figure for the scores and the average/standard deviation
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    # Loop through each fold and train/test a linear regression model
    for fold_idx, (train_index, test_index) in enumerate(kfold.split(X, y)):
        # Split the data into training and test sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit a linear regression model on the training set
        model.fit(X_train, y_train)

        # Predict on the test set and evaluate the model
        y_pred = model.predict(X_test)
        pearson, _ = pearsonr(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Append the scores for this fold to the lists
        pearson_scores.append(pearson)
        r2_scores.append(r2)
        mse.append(mse_score)

        # Plot the Pearson and R2 scores for this fold on a separate subplot
        axs[0].plot(fold_idx, pearson, 'bo',label='Pearson')
        axs[0].plot(fold_idx, r2, 'go',label='R2')
        axs[0].plot(fold_idx, mse_score, 'x',label='MSE')

        axs[0].legend()

    # Set the title and labels for the subplot
    axs[0].set_title('Scores\n'+id)
    axs[0].set_xlabel('Fold index')
    axs[0].set_ylabel('Score')

    # Plot the average and standard deviation of the scores on a separate subplot
    axs[1].bar(['Pearson', 'R2','MSE'], [np.mean(pearson_scores), np.mean(r2_scores),np.mean(mse)],
               yerr=[np.std(pearson_scores), np.std(r2_scores), np.std(mse)])
    axs[1].set_title('Average and standard deviation of scores')
    axs[1].set_ylabel('Score')
    # Show the plots
    plt.tight_layout()
    plt.savefig('Output_data/foldscores' + id + '.png', bbox_inches='tight')
    plt.close('all')
    print('Scorer ran successfully')

if __name__ == "__main__":
    print(type(ProteinAnalysis))
    print('found biopython')
