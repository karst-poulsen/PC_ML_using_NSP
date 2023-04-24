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
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import d2_tweedie_score
from sklearn.model_selection import KFold


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
    x_train, x_test, y_train, y_test = train_test_split(df, label, test_size=test_percent, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    r2 = r2_score(y_test, predictions)
    corr, _ = pearsonr(y_test, predictions)
    mse_score = mean_squared_error(y_test, predictions)
    r2s.append(r2)
    pearson.append(corr)
    mse.append(mse_score)
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
        r2s.append(r2)
        pearson.append(corr)
        feats.append(j)
        mse.append(mse_score)
    # a = pd.DataFrame(list(zip(feats, pearson, r2s, feat_import)), columns=['feat', 'pearson', 'R2', 'importances'])
    # a.to_excel("Output_data/scram_loss_feats" + id + ".xlsx")
    plt.rcParams['figure.dpi'] = 300
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(feats, pearson, color='tab:blue', marker='o', label='Pearson')
    ax.plot(feats, r2s, color='tab:red', marker='s', label='R2')
    ax.plot(feats, mse, color='tab:green', marker='^', label='MSE')
    ax.plot(feats, feat_import, color='tab:purple', marker='x', label='Feature Importance')

    ax.set_xticklabels(feats, rotation=90)
    ax.legend()

    ax.set_title('Score as a Function of Scrambled Feature\n{}'.format(id))
    ax.set_xlabel('Feature')
    ax.set_ylabel('Score')
    plt.savefig('Output_data/FeatScramLoss' + id + '.png', bbox_inches='tight')
    plt.close('all')
    print('Scramble Scoring ran successfully')


def feat_drop(df, label, model, identifier, test_percent):
    id = identifier
    feats = []
    r2s = []
    pearson = []
    mse = []
    x_train, x_test, y_train, y_test = train_test_split(df, label, test_size=test_percent, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    r2 = r2_score(y_test, predictions)
    corr, _ = pearsonr(y_test, predictions)
    mse_score = mean_squared_error(y_test, predictions)
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
        r2s.append(r2)
        pearson.append(corr)
        mse.append(mse_score)
        feats.append(i)

    # df_out = pd.DataFrame(list(zip(feats, pearson, r2s, mse, accuracy)), columns=['dropped feat', 'Pearson', 'r2', 'neg_mse', 'accuracy'])
    # df_out.to_excel("Output_data/feat_drop_cumulative" + id + ".xlsx")
    plt.rcParams['figure.dpi'] = 300
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(feats, pearson, color='tab:blue', marker='o', label='Pearson')
    ax.plot(feats, r2s, color='tab:red', marker='s', label='R2')
    ax.plot(feats, mse, color='tab:green', marker='^', label='MSE')
    ax.plot(feats, feat_import, color='tab:purple', marker='x', label='Feature Importance')

    ax.set_xticklabels(feats, rotation=90)
    ax.legend()

    ax.set_title('Score as a Function of Feature Drop\n{}'.format(id))
    ax.set_xlabel('Features Dropped')
    ax.set_ylabel('Score')

    plt.savefig('Output_data/feat_drop_cumulative_{}.png'.format(id), bbox_inches='tight')
    plt.close(fig)
    print('Feat drop ran successfully')


def feat_drop_multifold(df, label, model, identifier, test_percent, folds):
    id = identifier
    feats = []
    r2s = []
    pearson = []
    mse = []
    feat_importances = []

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    x_train, x_test, y_train, y_test = train_test_split(df, label, test_size=test_percent, random_state=42)
    model.fit(x_train, y_train)
    feat_import = model.feature_importances_

    a = list(zip(feat_import, df.columns))
    a.sort(reverse=True)
    col_import = pd.DataFrame(a, columns=['importances', 'names'])
    sorted_cols = col_import['names']
    feats.append('All Feats')
    feats.extend(sorted_cols.tolist())
    feats = feats[:-1]

    for train_index, test_index in kf.split(df):
        x_train, x_test = df.iloc[train_index], df.iloc[test_index]
        y_train, y_test = label[train_index], label[test_index]

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        r2 = r2_score(y_test, predictions)
        corr, _ = pearsonr(y_test, predictions)
        mse_score = mean_squared_error(y_test, predictions)
        feat_import = model.feature_importances_

        a = list(zip(feat_import, df.columns))
        a.sort(reverse=True)

        feat_import = np.insert(feat_import, [0], 0)
        feat_import = np.delete(feat_import, [-1])

        r2s.append(r2)
        pearson.append(corr)
        mse.append(mse_score)
        feat_importances.append(feat_import)

        df_3 = df.copy()
        for i in sorted_cols:
            if i == sorted_cols.iloc[-1]:
                break
            df_3.drop(columns=[i], inplace=True)
            x_train, x_test = df_3.iloc[train_index], df_3.iloc[test_index]
            y_train, y_test = label[train_index], label[test_index]

            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            r2 = r2_score(y_test, predictions)
            corr, _ = pearsonr(y_test, predictions)
            mse_score = mean_squared_error(y_test, predictions)
            r2s.append(r2)
            pearson.append(corr)
            mse.append(mse_score)

    feats = np.array(feats)
    r2s = np.array(r2s).reshape(folds, -1)
    pearson = np.array(pearson).reshape(folds, -1)
    mse = np.array(mse).reshape(folds, -1)
    feat_importances = np.array(feat_importances).reshape(folds, -1)

    import matplotlib.pyplot as plt

    mean_r2s = np.mean(r2s, axis=0)
    std_r2s = np.std(r2s, axis=0)
    mean_pearson = np.mean(pearson, axis=0)
    std_pearson = np.std(pearson, axis=0)
    mean_mse = np.mean(mse, axis=0)
    std_mse = np.std(mse, axis=0)
    mean_feat_importances = np.mean(feat_importances, axis=0)
    std_feat_importances = np.std(feat_importances, axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot R2 scores
    ax.plot(feats, mean_r2s, color='tab:red', label='R2')
    ax.fill_between(feats, mean_r2s - std_r2s, mean_r2s + std_r2s, alpha=0.2, color='tab:red')

    # Plot Pearson correlation coefficients
    ax.plot(feats, mean_pearson, color='tab:blue', label='Pearson')
    ax.fill_between(feats, mean_pearson - std_pearson, mean_pearson + std_pearson, alpha=0.2, color='tab:blue')

    # Plot mean squared errors
    ax.plot(feats, mean_mse, color='tab:green', label='MSE')
    ax.fill_between(feats, mean_mse - std_mse, mean_mse + std_mse, alpha=0.2, color='tab:green')

    # Plot feature importances
    ax.plot(feats, mean_feat_importances, color='tab:purple', label='Feature Importance')
    ax.fill_between(feats, mean_feat_importances - std_feat_importances, mean_feat_importances + std_feat_importances,
                    alpha=0.2, color='tab:purple')

    ax.set_xticklabels(feats, rotation=90)
    ax.legend()

    ax.set_title('Score as a Function of Feature Drop\n{}'.format(id))
    ax.set_xlabel('Features Dropped')
    ax.set_ylabel('Score')

    plt.savefig('Output_data/feat_drop_multifold_{}.png'.format(id), bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print('Feat drop multifold ran successfully')


def scorer(df, label, model, identifier, folds):
    id = identifier
    y = label
    X = df

    # Initialize lists to store Pearson and R2 scores for each fold
    pearson_scores = []
    r2_scores = []
    mse = []
    # Split your data into 10 folds using KFold
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)

    # Create a 2x1 subplot figure for the scores and the average/standard deviation

    # Loop through each fold and train/test a linear regression model
    for train_index, test_index in kfold.split(X, y):
        # Split the data into training and test sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit a linear regression model on the training set
        model.fit(X_train, y_train)

        # Predict on the test set and evaluate the model
        y_pred = model.predict(X_test)
        pearson, _ = pearsonr(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse_score = mean_squared_error(y_test, y_pred)

        # Append the scores for this fold to the lists
        pearson_scores.append(pearson)
        r2_scores.append(r2)
        mse.append(mse_score)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # plot individual and aggregate scores for each scoring method
    fold_idx = range(len(pearson_scores))
    axs[0].plot(fold_idx, pearson_scores, 'bo', label='Pearson')
    axs[0].plot(fold_idx, r2_scores, 'go', label='R2')
    axs[0].plot(fold_idx, mse, 'kx', label='MSE')
    axs[0].legend()
    axs[0].set_title('Individual Scores\n{}'.format(id))
    axs[0].set_xlabel('Fold Index')
    axs[0].set_ylabel('Score')

    pearson_mean = np.mean(pearson_scores)
    R2_mean =np.mean(r2_scores)
    MSE_mean = np.mean(mse)
    pearson_std = np.std(pearson_scores)
    R2_std = np.std(r2_scores)
    MSE_std = np.std(mse)
    feat_import = model.feature_importances_
    # plot the average and standard deviation of the scores on a separate subplot
    axs[1].bar(['Pearson', 'R2', 'MSE'], [pearson_mean, R2_mean, MSE_mean],
               yerr=[pearson_std, R2_std, MSE_std])
    axs[1].set_title('Average and Standard Deviation of Scores')
    axs[1].set_ylabel('Score')

    # adjust spacing between subplots and save the figure
    fig.subplots_adjust(wspace=0.3)
    fig.set_dpi(300)
    plt.tight_layout()
    plt.savefig('Output_data/scores_{}.png'.format(id), bbox_inches='tight')
    plt.close(fig)
    data=[[pearson_mean,pearson_std,R2_mean,R2_std,MSE_mean,MSE_std,df.shape[1],id,feat_import,df.columns.tolist()]]
    scores=pd.DataFrame(data,columns=['pearson_mean','pearson_std','R2_mean','R2_std','MSE_mean','MSE_std','Number of Features','ID','Feature Importances','Features'])
    print('Scorer ran successfully')
    return scores


def PCA_plot(df, label, identifier):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    id = identifier
    scaler = StandardScaler()
    X_std = scaler.fit_transform(df)
    pca = PCA(n_components=5)
    x_pca = pca.fit_transform(X_std)
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=label, cmap='viridis')
    plt.colorbar()
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('Output_data/PCA' + id + '.png')
    plt.close('all')
    print('PCA Ran successfully')


def RFECV_plot(df, label, model, identifier, folds, step, scoring='neg_mean_squared_error'):
    from sklearn.feature_selection import RFECV
    id = identifier
    min_feats = 8
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    estimator = model
    selector = RFECV(estimator=estimator, cv=cv, scoring=scoring, min_features_to_select=min_feats,
                     step=step)
    selector = selector.fit(df, label)
    selector.support_
    feat_list2 = selector.get_feature_names_out()
    selected_features = df.columns[selector.support_]
    df = df[feat_list2]
    # df.to_excel("Input_data/Save_files/df_RFECV"+id+id2+".xlsx")
    # rfecv_df=pd.DataFrame(selector.cv_results_)
    # rfecv_df.to_excel("Output_data/RFECV_results"+id+id2+".xlsx")
    # label_abund_df.to_excel("Input_data/Save_files/label_abund_all.xlsx")
    n_scores = len(selector.cv_results_["mean_test_score"])
    fig, ax = plt.subplots(figsize=(8, 6))

    x = range(1, n_scores + 1)
    y = selector.cv_results_["mean_test_score"]
    err = selector.cv_results_["std_test_score"]

    ax.plot(x, y, 'k-', label=scoring)
    ax.fill_between(x, y - err, y + err, alpha=0.2, label='Standard Deviation')
    ax.legend()
    ax.set_xlabel('Number of Features Selected')
    ax.set_ylabel(scoring)
    ax.set_title('Recursive Feature Elimination with Correlated Features\n{}'.format(id))

    fig.set_dpi(300)
    plt.tight_layout()
    plt.savefig('Output_data/RFECV_{}.png'.format(id), bbox_inches='tight')
    plt.close(fig)

    print('Recursive Feature Elimination with Correlated Features ran successfully')
    return df


def lasso_feature_selection(df, label, identifier):
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    X = df
    y = label

    # Scale the input features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform Lasso regression
    lasso = Lasso(alpha=0.1, random_state=42)
    pipe = Pipeline([('scaler', scaler), ('lasso', lasso)])
    pipe.fit(X_scaled, y)

    # Get feature importances
    importances = np.abs(pipe.named_steps['lasso'].coef_)
    feature_names = X.columns

    # Sort features by importance
    sorted_idx = importances.argsort()[::-1]
    importances = importances[sorted_idx]
    feature_names = feature_names[sorted_idx]
    id = identifier
    # Create plot of feature importances
    plt.figure()
    plt.title("Feature importances using Lasso Regression for {}".format(id))
    plt.bar(range(X.shape[1]), importances)
    plt.xticks(range(X.shape[1]), feature_names, rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.savefig('Output_data/LassoReg_{}.png'.format(id), bbox_inches='tight')
    plt.close('all')

    # Reduce dataset to high-importance features
    X_reduced = X.iloc[:, sorted_idx[:10]]

    return X_reduced


if __name__ == "__main__":
    print(type(ProteinAnalysis))
    print('found biopython')
