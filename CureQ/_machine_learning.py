import pandas as pd
import joblib
import os
import re
import fnmatch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_sample_weight
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from matplotlib.colors import LinearSegmentedColormap

from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from xgboost import XGBClassifier


def percentify(num, decimals=2):
    return f"{num * 100:.{decimals}f}%"

def ML_data_prep(folder):
    """
    This function prepares the data to be used for ML

    Parameters
    ----------
    folder : str
        The path file to the folder where the electrode_features file is stored.

    Returns
    -------
    Electrode_features : dataframe
       This dataframe contains the preprocessed data that models can use to make predictions.

    Notes
    -----
    
    """
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames,"*.csv"):
            matches.append(os.path.join(root,filename))

    for match in matches:
        if match.endswith("Electrode_Features.csv"):
            Electrode_features = pd.read_csv(match)
            Electrode_features['Spikes'] = Electrode_features['Spikes'].fillna(0)
            Electrode_features['Bursts'] = Electrode_features['Bursts'].fillna(0)
            Electrode_features = Electrode_features.fillna(-666)
            Electrode_features = Electrode_features.drop(columns = ["Active_electrodes", "Electrode"])

    return(Electrode_features)

def ML_predict(data, given_model, well, version):
    """
    Makes singular predictions based on the data the user is investigating. This works for one well.

    Parameters
    ----------
    data : str
        Name of the electrode held in the hdf5 file.
    given_model : str
        This shows what the user wants to use to make the prediction, this could be, Random Forest, XGBoost or Support Vector Machine.
    well : int
        This number shows which well the user wants to make a prediction of
    version : str
        This variable tells the program if the user wants to use the base 'MEAlytics version' of the Machine Learning model or the 'User version'.

    Returns
    -------
    text : str
        The text that tells the percentage of the chance the prediction matches certain classifications.

    Notes
    -----
    """
    if version == "MEAlytics version":
        if given_model == "Random Forest":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'RF.pkl')
        elif given_model == "Support Vector Machine":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'SVM.pkl')
            scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'svm_scaler.pkl')
            scaler = joblib.load(scaler_path)
        elif given_model == "XGBoost":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'XGB.pkl')
    elif version == "User version":
        if given_model == "Random Forest":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'RF_user.pkl')
        elif given_model == "Support Vector Machine":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'SVM_user.pkl')
            scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'svm_scaler_user.pkl')
            scaler = joblib.load(scaler_path)
        elif given_model == "XGBoost":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'XGB_user.pkl')

    model = joblib.load(model_path)
    welldata = data[data["Well"].isin([well])]
    if welldata.empty or sum(welldata["Spikes"] == 0):
        return f"Well {well} has no spikes and thus no classification."
    SCA_preds = []
    control_preds = []
    welldata = welldata.drop(columns = "Well")
    for i in range(len(welldata)):
        X = welldata.iloc[[i]]
        if given_model != "Random Forest" and scaler is not None:
            X = scaler.transform(X)
        probs = model.predict_proba(X)[0]
        control_preds.append(probs[0])
        SCA_preds.append(probs[1])

    control_preds = sum(control_preds) / len(control_preds)
    SCA_preds = sum(SCA_preds) / len(SCA_preds)
    if SCA_preds > control_preds:
        text = f"Well {well} has a {percentify(SCA_preds)} chance to be SCA."
    elif SCA_preds < control_preds:
        text = f"Well {well} has a {percentify(SCA_preds)} chance to be control."
    elif SCA_preds == control_preds:
        text = "This model cannot differentiate which classification this is."
    return(text)

def Feature_importance(given_model, data, version):
    """
    Shows the importance of all features used in the predictions of the model.

    Parameters
    ----------
    data : str
        Name of the electrode held in the hdf5 file.
    given_model : str
        This shows what the user wants to use to make the prediction, this could be, Random Forest, XGBoost or Support Vector Machine.
    version : str
        This variable tells the program if the user wants to use the base 'MEAlytics version' of the Machine Learning model or the 'User version'.

    Returns
    -------
    fig : plt
        A plot showcasing the importances of all features, ranked from top to bottom.

    Notes
    -----
    """ 
    if version == "MEAlytics version":
        if given_model == "Random Forest":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'RF.pkl')
        elif given_model == "Support Vector Machine":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'SVM.pkl')
            scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'svm_scaler.pkl')
            scaler = joblib.load(scaler_path)
        elif given_model == "XGBoost":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'XGB.pkl')
    elif version == "User version":
        if given_model == "Random Forest":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'RF_user.pkl')
        elif given_model == "Support Vector Machine":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'SVM_user.pkl')
            scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'svm_scaler_user.pkl')
            scaler = joblib.load(scaler_path)
        elif given_model == "XGBoost":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'XGB_user.pkl')

    model = joblib.load(model_path)

    if given_model != "Support Vector Machine":
        data = data.drop(columns = "Well")
        importance = model.feature_importances_
        importance_df = pd.DataFrame(importance,index =list(data.columns.values),columns=['importance']) 
        importance_df.sort_values(inplace=True,by='importance',ascending=False)

        # Plotting the feature importances
        fig, ax = plt.subplots(figsize=(10, 6))  
        ax.barh(importance_df.index, importance_df['importance'], color='skyblue')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title(f'Feature Importance for {given_model}')
        fig.tight_layout()

        return fig  
    else:
        X_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'models', 'X_train.csv'), sep=',')
        y_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'models', 'y_train.csv'), sep=',')
        y_train = y_train.values.ravel()
        X_train_scaled = scaler.transform(X_train)  

        result = permutation_importance(model, X_train_scaled, y_train, n_repeats=10, random_state=42, scoring='f1_macro')

        importance_df = pd.DataFrame(result.importances_mean, index=X_train.columns, columns=["importance"])
        importance_df.sort_values(by="importance", ascending=False, inplace=True)

        fig, ax = plt.subplots(figsize=(10, 6))  
        ax.barh(importance_df.index, importance_df['importance'], color='skyblue')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title(f'Feature Importance for {given_model}')
        fig.tight_layout()

        return fig
    
def ML_predict_full(data, given_model, wells, rows, cols, version):
    """
    Makes all well predictions based on the data the user is investigating. 

    Parameters
    ----------
    data : str
        Name of the electrode held in the hdf5 file.
    given_model : str
        This shows what the user wants to use to make the prediction, this could be, Random Forest, XGBoost or Support Vector Machine.
    wells : int
        This number shows the amount of wells the file has
    rows : int
        This number shows the amount of rows the file has
    cols : int
        This number shows the amount of cols the file has
    version : str
        This variable tells the program if the user wants to use the base 'MEAlytics version' of the Machine Learning model or the 'User version'.

    Returns
    -------
    fig : plt
        A plot showing the predictions of each well.

    Notes
    -----
    """
    if version == "MEAlytics version":
        if given_model == "Random Forest":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'RF.pkl')
        elif given_model == "Support Vector Machine":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'SVM.pkl')
            scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'svm_scaler.pkl')
            scaler = joblib.load(scaler_path)
        elif given_model == "XGBoost":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'XGB.pkl')
    elif version == "User version":
        if given_model == "Random Forest":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'RF_user.pkl')
        elif given_model == "Support Vector Machine":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'SVM_user.pkl')
            scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'svm_scaler_user.pkl')
            scaler = joblib.load(scaler_path)
        elif given_model == "XGBoost":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'XGB_user.pkl')
    model = joblib.load(model_path)
    classes = []

    for i in range(1, (wells+1)):
        welldata = data[data["Well"].isin([i])]
        if welldata.empty or sum(welldata["Spikes"] == 0):
            wellclass = "Unknown (no spikes)"
        else:
            welldata = welldata.drop(columns = "Well")
            SCA_preds = []
            control_preds = []
            
            for i in range(len(welldata)):
                X = welldata.iloc[[i]]
                if given_model != "Random Forest" and scaler is not None:
                    X = scaler.transform(X)
                probs = model.predict_proba(X)[0]
                control_preds.append(probs[0])
                SCA_preds.append(probs[1])

            control_preds = sum(control_preds) / len(control_preds)
            SCA_preds = sum(SCA_preds) / len(SCA_preds)
            if SCA_preds > control_preds:
                wellclass = "SCA"
            elif SCA_preds < control_preds:
                wellclass = "Control"

        classes.append(wellclass)

    class_labels = ["Unknown (no spikes)", "SCA", "Control"]
    class_colors = {
        "Unknown (no spikes)": "gray",
        "SCA": "darkred",
        "Control": "darkgreen"
    }
    class_to_int = {label: i for i, label in enumerate(class_labels)}
    color_list = [class_colors[label] for label in class_labels]
    cmap = plt.matplotlib.colors.ListedColormap(color_list)

    grid_data = np.array([class_to_int[label] for label in classes]).reshape((rows, cols))

    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.imshow(grid_data, cmap=cmap)

    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    wellnum = 1
    for i in range(rows):
        for j in range(cols):
            ax.text(j, i, str(wellnum), ha="center", va="center", color="black", fontsize=8)
            wellnum += 1

    legend_elements = [Patch(facecolor=class_colors[label], label=label) for label in class_labels]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(f'All predictions of the {given_model} model')

    plt.tight_layout()
    return fig

def ML_performance(given_model, version):
    """
    Appends the values of the model the user wishes to investigate.

    Parameters
    ----------
    given_model : str
        This shows what the user wants to use to make the prediction, this could be, Random Forest, XGBoost or Support Vector Machine.
    version : str
        This variable tells the program if the user wants to use the base 'MEAlytics version' of the Machine Learning model or the 'User version'.

    Returns
    -------
    acc : float
        This returns the 'accuracy' value of the chosen model up to four digits.
    precision : float
        This returns the 'Precision' value of the chosen model up to four digits.
    recall : float
        This returns the 'Recall' value of the chosen model up to four digits.
    f1 : float
        This returns the 'F1-score' value of the chosen model up to four digits.
    Notes
    -----
    """
    if version == "MEAlytics version":
        if given_model == "Random Forest":
            acc = 0.7007
            f1 = 0.6884
            recall = 0.6890
            precision = 0.6878
        elif given_model == "Support Vector Machine":
            acc = 0.6667
            f1 = 0.6150
            recall = 0.6174
            precision = 0.6555
        elif given_model == "XGBoost":
            acc = 0.6941
            f1 = 0.6857
            recall = 0.6897
            precision = 0.6843
    elif version == "User version":
        metrics_filepath = os.path.join(os.path.dirname(__file__), 'models', 'User_model_results.csv')
        perf_metrics_df = pd.read_csv(metrics_filepath)
        row = perf_metrics_df[perf_metrics_df["Model"] == given_model]

        if given_model == "Random Forest":
            acc = row["Accuracy"]
            f1 = row["F1 Score"]
            recall = row["Recall"]
            precision = row["Precision"]
        elif given_model == "Support Vector Machine":
            acc = row["Accuracy"]
            f1 = row["F1 Score"]
            recall = row["Recall"]
            precision = row["Precision"]
        elif given_model == "XGBoost":
            acc = row["Accuracy"]
            f1 = row["F1 Score"]
            recall = row["Recall"]
            precision = row["Precision"]
    
    return acc, precision, recall, f1

def classify_number(num, classification_dict):
    for name, lst in classification_dict.items():
        if num in lst:
            return name
    return "Unknown"  

def add_data(folder, class_dict):
    """
    Adds the data the user uploaded to a database for future training of the model.

    Parameters
    ----------
    folder : str
        This variable is the path to the folder the user uploaded and contains the 'Feature' and 'Electrode_Feature' files.
    class_dict : dict
        Dictionary containing the classification if each well.
    
    Returns
    -------

    Notes
    -----
    """
    Electrode_path = os.path.join(os.path.dirname(__file__), 'models', 'Electrode_dataset.csv')
    Electrode_set = pd.read_csv(Electrode_path)
    Electrode_sources = Electrode_set["data_source"].unique()

    Well_path = os.path.join(os.path.dirname(__file__), 'models', 'Well_dataset.csv')
    Well_set = pd.read_csv(Well_path)
    Well_sources = Well_set["data_source"].unique()

    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames,"*.csv"):
            matches.append(os.path.join(root,filename))

    Electrode_duplicates = 0
    Well_duplicates = 0
    
    for match in matches:
        if match.endswith("Electrode_Features.csv"):
            data_source = os.path.basename(match)
            cleaned_filename = re.sub(r'_\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}', '', data_source)
            if cleaned_filename in Electrode_sources:
                Electrode_duplicates = Electrode_duplicates + 1
            else:
                Electrode_features = pd.read_csv(match)
                Electrode_features["Classification"] = Electrode_features["Well"].apply(
                    lambda x: classify_number(x, class_dict)
                )
                Electrode_features['Spikes'] = Electrode_features['Spikes'].fillna(0)
                Electrode_features['Bursts'] = Electrode_features['Bursts'].fillna(0)
                Electrode_features = Electrode_features.fillna(-666)
                Electrode_features = Electrode_features.drop(columns = ["Well", "Active_electrodes", "Electrode"])
                Electrode_features['data_source'] = cleaned_filename
                Electrode_set = pd.concat([Electrode_set, Electrode_features])

        elif match.endswith("Features.csv") and not match.endswith("Electrode_Features.csv"):
            data_source = os.path.basename(match)
            cleaned_filename = re.sub(r'_\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}', '', data_source)
            if cleaned_filename in Well_sources:
                Well_duplicates = Well_duplicates + 1
            else:
                Well_features = pd.read_csv(match)
                Well_features["Classification"] = Well_features["Well"].apply(
                    lambda x: classify_number(x, class_dict)
                )
                Well_features['Spikes'] = Well_features['Spikes'].fillna(0)
                Well_features['Bursts'] = Well_features['Bursts'].fillna(0)
                Well_features = Well_features.fillna(-666)
                Well_features = Well_features.drop(columns = ["Well", "Active_electrodes"])
                Well_features['data_source'] = cleaned_filename
                Well_set = pd.concat([Well_set, Well_features])

    Electrode_set.to_csv(Electrode_path, index=False)
    Well_set.to_csv(Well_path, index=False)

def Params_show(given_model, version):
    """
    Shows the parameters of the selected model.

    Parameters
    ----------
    given_model : str
        This shows what the user wants to use to make the prediction, this could be, Random Forest, XGBoost or Support Vector Machine.
    version : str
        This variable tells the program if the user wants to use the base 'MEAlytics version' of the Machine Learning model or the 'User version'.

    Returns
    -------
    params : dataframe
        A dataframe containing the name of the parameters and the values of the parameters.
    Notes
    -----
    """
    if version == "MEAlytics version":
        if given_model == "Random Forest":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'RF.pkl')
        elif given_model == "Support Vector Machine":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'SVM.pkl')

        elif given_model == "XGBoost":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'XGB.pkl')
    elif version == "User version":
        if given_model == "Random Forest":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'RF_user.pkl')
        elif given_model == "Support Vector Machine":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'SVM_user.pkl')
        elif given_model == "XGBoost":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'XGB_user.pkl')
    model = joblib.load(model_path)
    params = model.get_params()
    return params
    
def Train_model(given_model):
    """
    Trains a new machine learning model based on the current dataset in the 'models' folder.

    Parameters
    ----------
    given_model : str
        This shows what the user wants to use to make the prediction, this could be, Random Forest, XGBoost or Support Vector Machine.

    Returns
    -------
    fig : plt
        A plot containing a confusion matrix which describes the performance of the newly trained model.
    class_report : array
        An array containing the performance metrics 'accucary', 'recall', 'precision' and 'f1-score' for each individual class and the total performance metrics.
    
    Notes
    -----
    """    
    metrics_filepath = os.path.join(os.path.dirname(__file__), 'models', 'User_model_results.csv')
    perf_metrics_df = pd.read_csv(metrics_filepath)

    filepath = os.path.join(os.path.dirname(__file__), 'models', 'Electrode_dataset.csv')
    rawdata = pd.read_csv(filepath, sep=',') 
        
    mapping = {}
    for label in rawdata["Classification"].unique():
        if label == "control":
            continue
        parts = label.split("_")
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        number = int(parts[1])
        if 40 <= number <= 55:
            mapping[label] = "SCA1_late_aoo"
        elif number > 55:
            mapping[label] = "SCA1_early_aoo"

    rawdata["Classification"] = rawdata["Classification"].replace(mapping)
    le = LabelEncoder()

    X = rawdata.copy().drop(columns=["Classification", "data_source", "Well"])
    y = rawdata[['Classification']].copy()

    y['Classification'] = le.fit_transform(y['Classification'])

    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    colors = ["red", "green"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

    if given_model == "Random Forest":
        param_grid = {
            'rf__n_estimators': [100, 300],
            'rf__max_depth': [None, 10, 20],
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf': [1, 2, 4],
            'rf__max_features': ['sqrt', 'log2'],
            'rf__bootstrap': [True, False]
        }

        pipeline = Pipeline([
            ('rf', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ])

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='f1_macro',
            cv=skf,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_test) 
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))  
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=le.classes_, yticklabels=le.classes_, cbar=True, ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix for Random Forest")
        fig.tight_layout()

        class_report = classification_report(y_test, y_pred, digits=4)

        acc = round(accuracy_score(y_test, y_pred), 4)
        prec = round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4)
        rec = round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4)
        f1 = round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4)

        full_metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1, "Model": "Random Forest"}
        full_metrics_df = pd.DataFrame([full_metrics])

        perf_metrics_df = perf_metrics_df[perf_metrics_df["Model"] != "Random Forest"]
        perf_metrics_df = pd.concat([perf_metrics_df, full_metrics_df], ignore_index=True)

        perf_metrics_df.to_csv(metrics_filepath, index=False)

        model_path = os.path.join(os.path.dirname(__file__), 'models', 'RF_user.pkl')
        rf_model = best_model.named_steps['rf']
        joblib.dump(rf_model, model_path)


    elif given_model == "Support Vector Machine":
        param_grid = {
            'svm__C': [0.1, 1, 10],
            'svm__kernel': ['linear', 'rbf', 'poly'],
            'svm__gamma': ['scale', 'auto', 0.01, 0.001],
            'svm__degree': [2, 3, 4],
            'svm__coef0': [0, 0.1, 0.5]
        }

        pipeline = Pipeline([
            ('scaler', StandardScaler()), 
            ('svm', SVC(probability = True, random_state=42, class_weight='balanced'))                       
        ])

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            scoring='f1_macro',
            cv=skf,
            verbose=2,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))  
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=le.classes_, yticklabels=le.classes_, cbar=True, ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix for SVM")
        fig.tight_layout()

        class_report = classification_report(y_test, y_pred, digits=4)

        acc = round(accuracy_score(y_test, y_pred), 4)
        prec = round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4)
        rec = round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4)
        f1 = round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4)

        full_metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1, "Model": "SVM"}
        full_metrics_df = pd.DataFrame([full_metrics])

        perf_metrics_df = perf_metrics_df[perf_metrics_df["Model"] != "SVM"]
        perf_metrics_df = pd.concat([perf_metrics_df, full_metrics_df], ignore_index=True)

        perf_metrics_df.to_csv(metrics_filepath, index=False)

        model_path = os.path.join(os.path.dirname(__file__), 'models', 'SVM_user.pkl')
        svm_model = best_model.named_steps['svm']
        joblib.dump(svm_model, model_path)

        scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'svm_scaler_user.pkl')
        svm_scaler = best_model.named_steps['scaler']
        joblib.dump(svm_scaler, scaler_path)

    elif given_model == "XGBoost":
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

        param_grid = {
            'xgb__n_estimators': [500, 800],
            'xgb__learning_rate': [0.01, 0.05],
            'xgb__max_depth': [6, 10],
            'xgb__min_child_weight': [1, 3],
            'xgb__gamma': [0.1, 0.3],
            'xgb__subsample': [0.8, 1.0],
            'xgb__colsample_bytree': [0.8, 1.0]
        }

        xgb = XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', xgb)
        ])

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='f1_macro',  
            n_jobs=-1,              
            cv=skf,
        )

        grid_search.fit(X_train, y_train, **{'xgb__sample_weight': sample_weights})
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))  
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=le.classes_, yticklabels=le.classes_, cbar=True, ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix for XGBoost")
        fig.tight_layout()
        
        class_report = classification_report(y_test, y_pred, digits=4)

        acc = round(accuracy_score(y_test, y_pred), 4)
        prec = round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4)
        rec = round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4)
        f1 = round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4)

        full_metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1, "Model": "XGBoost"}
        full_metrics_df = pd.DataFrame([full_metrics])

        perf_metrics_df = perf_metrics_df[perf_metrics_df["Model"] != "XGBoost"]
        perf_metrics_df = pd.concat([perf_metrics_df, full_metrics_df], ignore_index=True)

        perf_metrics_df.to_csv(metrics_filepath, index=False)

        model_path = os.path.join(os.path.dirname(__file__), 'models', 'XGB_user.pkl')
        xgb_model = best_model.named_steps['xgb']
        joblib.dump(xgb_model, model_path)

    return fig, class_report
