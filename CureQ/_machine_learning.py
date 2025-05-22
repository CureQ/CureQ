import pandas as pd
import joblib
import os
import fnmatch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.inspection import permutation_importance

def percentify(num, decimals=2):
    return f"{num * 100:.{decimals}f}%"

def ML_data_prep(folder):
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

def ML_predict(data, given_model, well):
    if given_model == "Random Forest":
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'RF.pkl')
    elif given_model == "Support Vector Machine":
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'SVM.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'svm_scaler.pkl')
        scaler = joblib.load(scaler_path)
    elif given_model == "XGBoost":
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'XGB.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'xgb_scaler.pkl')
        scaler = joblib.load(scaler_path)
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

def Feature_importance(given_model, data):
    if given_model == "Random Forest":
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'RF.pkl')
    elif given_model == "Support Vector Machine":
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'SVM.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'svm_scaler.pkl')
        scaler = joblib.load(scaler_path)
    elif given_model == "XGBoost":
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'XGB.pkl')
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
    
def ML_predict_full(data, given_model, wells, rows, cols):
    if given_model == "Random Forest":
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'RF.pkl')
    elif given_model == "Support Vector Machine":
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'SVM.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'svm_scaler.pkl')
        scaler = joblib.load(scaler_path)
    elif given_model == "XGBoost":
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'XGB.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'xgb_scaler.pkl')
        scaler = joblib.load(scaler_path)
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
    ax.set_title(f'Feature Importance for {given_model}')

    plt.tight_layout()
    return fig

def ML_performance(given_model):
    if given_model == "Random Forest":
        acc = 0.6776
        f1 = 0.6701
        recall = 0.6751
        precision = 0.6693
    elif given_model == "Support Vector Machine":
        acc = 0.6513
        f1 = 0.6464
        recall = 0.6547
        precision = 0.6483
    elif given_model == "XGBoost":
        acc = 0.6798
        f1 = 0.6704
        recall = 0.6736
        precision = 0.6692
    
    return precision, recall, f1, acc