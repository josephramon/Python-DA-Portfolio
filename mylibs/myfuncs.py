import torch
import numpy as np
import matplotlib.pyplot as plt
import psutil
import shutil
import math
import datetime as dt
import pandas as pd
import sweetviz as sv
import seaborn as sns
import pytz
from IPython.display import Audio, display
from IPython.core.display import HTML

import tensorflow as tf
from tensorflow import keras

# check if Kaggle GPU is enabled or not
from tensorflow.python.client import device_lib
def is_kaggle_gpu_enabled():
    # Return whether GPU is enabled in the running Kaggle kernel
    search_string = str(device_lib.list_local_devices())
    return 'GPU' in search_string
'''
if (is_kaggle_gpu_enabled()) == False:
    return 'hist'
else:
    return 'gpu_hist'
'''   


# clear GPU cache
def clear_gpu(tree_method='gpu_hist'):
    #if (is_kaggle_gpu_enabled()) == True:
    # I use this if, instead of above if, so it won't generate tensorflow messages
    # this means above cell must be run at least once before this cell is run
    if tree_method == 'gpu_hist':
        # using torch
        torch.cuda.empty_cache() 
  
  
# copy from corochann (Kaggle Grandmaster) notebook 
# https://www.kaggle.com/code/corochann/ashrae-feather-format-for-fast-loading/notebook

from pandas.api.types import is_datetime64_any_dtype as is_datetime
def reduce_mem_usage(df, print_info = True, use_float16=False):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    if print_info == True:
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        if is_datetime(df[col]):
            # skip datetime type
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min >= np.iinfo(np.uint8).min and c_max <= np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and \
                            c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    if print_info == True:
        print()
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
    
    
def runtime(rt1,rt2):
    tdiff=rt2 - rt1
    # get seconds and convert to h:m:s
    print(f'Runtime : {str(dt.timedelta(seconds=tdiff.total_seconds()))}')


def create_download_link(title = "Download ", filename = "data.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title + filename,filename=filename)
    return HTML(html)
    
    
def GetRam():
    # Getting available (2nd field)
    #print(round(psutil.virtual_memory()[3]/(1024.0 ** 3),2))
    # Getting % usage of virtual_memory ( 3rd field)
    #print('RAM memory % used:', psutil.virtual_memory()[2])
    return psutil.virtual_memory()[2]
    
    
def convertFloatToDecimal(f=0.0, precision=2):
    '''
    Convert a float to string of decimal.
    precision: by default 2.
    If no arg provided, return "0.00".
    '''
    return ("%." + str(precision) + "f") % f

def formatFileSize(size, sizeIn, sizeOut, precision=0):
    '''
    Convert file size to a string representing its value in B, KB, MB and GB.
    The convention is based on sizeIn as original unit and sizeOut
    as final unit. 
    '''
    assert sizeIn.upper() in {"B", "KB", "MB", "GB"}, "sizeIn type error"
    assert sizeOut.upper() in {"B", "KB", "MB", "GB"}, "sizeOut type error"
    if sizeIn == "B":
        if sizeOut == "KB":
            return convertFloatToDecimal((size/1024.0), precision)
        elif sizeOut == "MB":
            return convertFloatToDecimal((size/1024.0**2), precision)
        elif sizeOut == "GB":
            return convertFloatToDecimal((size/1024.0**3), precision)
    elif sizeIn == "KB":
        if sizeOut == "B":
            return convertFloatToDecimal((size*1024.0), precision)
        elif sizeOut == "MB":
            return convertFloatToDecimal((size/1024.0), precision)
        elif sizeOut == "GB":
            return convertFloatToDecimal((size/1024.0**2), precision)
    elif sizeIn == "MB":
        if sizeOut == "B":
            return convertFloatToDecimal((size*1024.0**2), precision)
        elif sizeOut == "KB":
            return convertFloatToDecimal((size*1024.0), precision)
        elif sizeOut == "GB":
            return convertFloatToDecimal((size/1024.0), precision)
    elif sizeIn == "GB":
        if sizeOut == "B":
            return convertFloatToDecimal((size*1024.0**3), precision)
        elif sizeOut == "KB":
            return convertFloatToDecimal((size*1024.0**2), precision)
        elif sizeOut == "MB":
            return convertFloatToDecimal((size*1024.0), precision)

# Usage Example : formatFileSize(46658,'KB','MB',2)   -> will result in 45.56


def check_cols_with_nulls(df):
    cols_with_missing = [col for col in df.columns if df[col].isnull().any()]
    if len(cols_with_missing) == 0:
        print("No Missing Values")
    else:
        print(cols_with_missing)
    
    sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
    
    
def check_infinity_nan(df,dfname):
    print("checking for infinity")
  
    #ds = sba.isin([np.inf, -np.inf])
    #print(ds)
  
    # printing the count of infinity values
    print()
    print("printing the count of infinity values")
  
    count = np.isinf(df).values.sum()
    print(f"{dfname} contains {str(count)} infinite values")
    print()
    
    has_nan = df.isnull().values.any()
    print(f"Does {dfname} have Nan or Null values ?  {has_nan}")
    
    
# used as a converter when loading csv
def fixvals(val):
    retval = val.replace('$','').replace(',','')
    return retval
    

class color:
    purple = '\033[95m'
    cyan = '\033[96m'
    darkcyan = '\033[36m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    bold = '\033[1m'
    underline = '\033[4m'
    end = '\033[0m'
    bdunl = '%s%s' % (bold, underline)
    bdblue = '%s%s' % (bold, blue)
    bdgreen = '%s%s' % (bold, green)
    bdred = '%s%s' % (bold, red)


# METRICS Function
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
def model_eval(y_valid,predictions, cmDisplay=False):
    print('MAE:', metrics.mean_absolute_error(y_valid, predictions))
    #print('MSE:', metrics.mean_squared_error(y_valid, predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, predictions)))
    print()
    
    ClassificationReport = classification_report(y_valid,predictions.round(),output_dict=True)

    print(f'{color.bold}Classification Report:{color.end}')
    print(classification_report(y_valid,predictions.round()))
    
    print()
    print(f"{color.bold}Confusion Matrix:{color.end}")

    if cmDisplay == True:
        cm = confusion_matrix(y_valid, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(dpi=100,figsize=(5,5))
        disp.plot(ax=ax,colorbar=False,values_format='d')
    
    cmv = confusion_matrix(y_valid, predictions)
    
    TrueNeg = cmv[0][0]
    FalsePos = cmv[0][1]
    FalseNeg = cmv[1][0]
    TruePos = cmv[1][1]

    TotalNeg = TrueNeg + FalseNeg
    TotalPos = TruePos + FalsePos
    
    print()
    print('True Negative  : CHGOFF (0) was predicted {} times correctly ({} %)'.format(
        TrueNeg, round((TrueNeg/TotalNeg)*100,2)
        ))
    print('False Negative : CHGOFF (0) was predicted {} times incorrectly ({} %)'.format(
        FalseNeg, round((FalseNeg/TotalNeg)*100,2)
        ))
    print('True Positive  : P I F (1) was predicted {} times correctly ({} %)'.format(
        TruePos, round((TruePos/TotalPos)*100,2)
        ))
    print('False Positive : P I F (1) was predicted {} times incorrectly ({} %)'.format(
        FalsePos, round((FalsePos/TotalPos)*100,2)
    ))
    
    print()
    asm = (accuracy_score(y_valid, predictions.round()) * 100)
    print(f'{color.bdgreen}Accuracy for model: %.2f{color.end}' % asm)
    print(f'{color.bdblue}f1-score: {color.end}')
    print(f"   CHGOFF (0) : {round(ClassificationReport['0']['f1-score']*100,2)}")
    print(f"   P I F (1)  : {round(ClassificationReport['1']['f1-score']*100,2)}")
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, predictions)))
    
    return {'cmv':cmv, 'ClassificationReport':ClassificationReport, 'AccuracyScore':asm}

def model_eval2(model, X_train, y_train, X_testdata, y_testdata,
                cmDisplay = False, prtstr = 'y_valid', multi_label = False,
                prt_acc = False):
                
    p_train = model.predict(X_train)
    p_testdata = model.predict(X_testdata)
    
    rmse_testdata = np.sqrt(metrics.mean_squared_error(y_testdata, p_testdata))
    
    if multi_label == True:
        f1_train = f1_score(y_train, p_train, average='micro') * 100
        f1_testdata = f1_score(y_testdata, p_testdata, average='micro') * 100
    else:
        f1_train = f1_score(y_train, p_train) * 100
        f1_testdata = f1_score(y_testdata, p_testdata) * 100
        
    print() 
    print('{}{} RMSE    : {}{}{}{}'.format(
            color.bold,
            prtstr,
            color.bdgreen,
            color.underline,
            rmse_testdata,
            color.end)
         )
    print()
    '''
    #Uncomment if you want to show this
    print("{}y_train f1 score: {}{}{}".format(
           color.bold, color.bdblue,
           f1_train,
           color.end)
         )
    '''
    print("{}{} f1 score: {}{}{} %{}".format(
           color.bold, 
           prtstr,
           color.bdgreen,
           color.underline,
           f1_testdata,
           color.end)
         )

    if prt_acc == True:
        print()
        asm = (accuracy_score(y_testdata, p_testdata.round()) * 100)
        print(f'{color.bdgreen}Accuracy for model: %.2f{color.end}' % asm)
                 
    print()
    print('-'*50)   

    clf_report = classification_report(y_testdata, p_testdata.round(),output_dict=True)
    df_cr = pd.DataFrame(clf_report)
    df_cr = df_cr.drop(['accuracy'], axis=1)
    df_cr = df_cr.transpose()
    print(f'\n{color.bold}Classification Report{color.end}')
    display(df_cr)
                                                 
    #f1_score_0 = clf_report['0']['f1-score'] * 100
    #f1_score_1 = clf_report['1']['f1-score'] * 100
    accuracy   = clf_report['accuracy'] * 100    
        
    cmv = confusion_matrix(y_testdata, p_testdata)
    
    TrueNeg = cmv[0][0]
    FalsePos = cmv[0][1]
    FalseNeg = cmv[1][0]
    TruePos = cmv[1][1]

    TotalNeg = TrueNeg + FalseNeg
    TotalPos = TruePos + FalsePos
    
    print()
    if prtstr == 'y_valid':
        print(f'{color.bold}Confusion Matrix using Validation Data (y_valid){color.end}')
    else:
        print(f'{color.bold}Confusion Matrix using Unseen Test Data (y_test){color.end}')
    
    print()
    print('True Negative  : CHGOFF (0) was predicted {} times correctly ({} %)'.format(
        TrueNeg, round((TrueNeg/TotalNeg)*100,2)
        ))
    print('False Negative : CHGOFF (0) was predicted {} times incorrectly ({} %)'.format(
        FalseNeg, round((FalseNeg/TotalNeg)*100,2)
        ))
    print('True Positive  : P I F (1) was predicted {} times correctly ({} %)'.format(
        TruePos, round((TruePos/TotalPos)*100,2)
        ))
    print('False Positive : P I F (1) was predicted {} times incorrectly ({} %)'.format(
        FalsePos, round((FalsePos/TotalPos)*100,2)
    ))
    
    if multi_label == False:
        precision = metrics.precision_score(y_testdata, p_testdata) * 100
        recall_sensitivity = metrics.recall_score(y_testdata, p_testdata, pos_label=1) * 100
        recall_specificity = metrics.recall_score(y_testdata, p_testdata, pos_label=0) * 100
    else:
        precision = metrics.precision_score(y_testdata, p_testdata, average='micro') * 100
        recall_sensitivity = metrics.recall_score(y_testdata, p_testdata, pos_label=1) * 100
        recall_specificity = metrics.recall_score(y_testdata, p_testdata, pos_label=0) * 100
        
    g_mean = math.sqrt(recall_sensitivity * recall_specificity)

    data = {f'Metrics Summary: {prtstr} in %':[
                               rmse_testdata,
                               f1_testdata,
                               #accuracy,
                               precision,
                               recall_sensitivity,
                               recall_specificity,
                               g_mean
                              ]
           }
 
    # Creates pandas DataFrame.
    df = pd.DataFrame(data,
                      index =['rmse',
                              'f1_score',
                              #'Accuracy',
                              'Precision',
                              'Recall Sensitivity',
                              'Recall Specificity',
                              'G-Mean'
                             ]
                      )

    display(df) 
     
    if cmDisplay == True:
        plt.rcParams["axes.grid"] = False
        plt.rcParams.update({'font.size': 4})
        cm = confusion_matrix(y_testdata, p_testdata)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(dpi=300,figsize=(1.5,1.5))
        
        disp.plot(ax=ax,colorbar=False, values_format='d')
        disp.ax_.set_xlabel('Predicted', fontsize=4)
        disp.ax_.set_ylabel('True', fontsize=4)
        # Set tick font size
        for label in (disp.ax_.get_xticklabels() + disp.ax_.get_yticklabels()):
            label.set_fontsize(4)
        disp.ax_.set_title(f"Confusion Matrix ({prtstr})", fontsize = 4)

    return {'df':df,
            'cmv':cmv, 
            'rmse_testdata':rmse_testdata,
            'f1_score_train':f1_train,
            'f1_score_testdata':f1_testdata}

from xgboost import plot_importance
# Plot xgboost feature importance
def plot_features(booster, figsize):  
    plt.rcParams.update({'font.size': 8})
    plt.style.use('Solarize_Light2')
    fig, ax = plt.subplots(1,1,figsize=figsize,dpi=300)
    # Set tick font size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(8)
    plot_importance(booster=booster, ax=ax)
    

# MUTUAL INFORMATION
from sklearn.feature_selection import mutual_info_regression
def make_mi_scores(X, y):
    print()
    print("Please wait, Mutual Information gathering can take time ...")
    #X = X.copy()
    #for colname in X.select_dtypes(["object", "category"]):
    #    X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    #discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    #mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = mutual_info_regression(X, y, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    print("Mutual Information gathering done ...")
    return mi_scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    
from IPython.display import clear_output   # to be able to use clear_output(wait=True)   
def GetSweetVizReport(df, htmlpath, kaggle_flag):
    print(f'{color.bold}Please wait, preparing SweetViz report{color.end}')
    try:
        my_report = sv.analyze(df)
    
        my_report.show_html(filepath=f'{htmlpath}', 
                open_browser=True, 
                layout='vertical', 
                scale=None)
        clear_output(wait=True)
        if kaggle_flag == 0:
            print(f'SweetViz Report has been downloaded to kaggle working directory {htmlpath}')
        else:
            print(f'SweetViz Report has been downloaded to {htmlpath}')
    except Exception as e:
        print(f'Error: {e}')
        

import pyttsx3        
''' 
Set up voice object.  Used in different areas of notebook to indicate completion of long processes.
'''
def SetVoice(kaggle_flag):
    if kaggle_flag == 0:   # not Kaggle
        engine = pyttsx3.init()  # object creation

        """ RATE"""
        #rate = engine.getProperty('rate')   # getting details of current speaking rate
        #print (rate)                        #printing current voice rate
        engine.setProperty('rate', 175)     # setting up new voice rate

        """VOLUME"""
        #volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
        #print (volume)                         #printing current volume level
        engine.setProperty('volume',0.7)        # setting up volume level  between 0 and 1

        """VOICE"""
        voices = engine.getProperty('voices')       #getting details of current voice
        #engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
        engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for female
        
        return engine
        
        
def InitTPUStrategy():
    print("Tensorflow version " + tf.__version__)

    # Detect and init the TPU
    try:
        # detect and init the TPU
        # TPUs are network-connected accelerators and you must first locate them on the network. 
        # This is what TPUClusterResolver.connect() does.  No parameters necessary if TPU_NAME
        # environment variable is set: this is always the case on Kaggle.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        print('Running on TPU ', tpu.master())
        # instantiate a distribution strategy
        # This object contains the necessary distributed training code that will work on TPUs 
        # with their 8 compute cores 
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        tpu_strategy = tf.distribute.TPUStrategy(tpu)
        tf.config.optimizer.set_jit(True)
        #tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

    except ValueError: # detect GPUs
        # default strategy that works on CPU and single GPU
        tpu_strategy = tf.distribute.get_strategy() 

    print("Number of accelerators: ", tpu_strategy.num_replicas_in_sync)

    '''With a TPUStrategy running on a single TPU v3-8, the core count is 8. This is the hardware 
    available on Kaggle. It could be more on larger configurations called TPU pods available on 
    Google Cloud.'''

    return tpu_strategy
    
    
def ZipDir(zippath):
    # Zip model directory.  Only doing this to be able to easily download from Kaggle 
    # working dir.
    #Give name of your final zipped file. .zip extension will be added automatically.
    output_file = zippath

    # Give the name of directory which you want to zip.
    # If you are in same location as the directory you can simply give it's name or
    # else give full path of directory.
    # Check your current directory using this command
    #print(os.getcwd())

    #full path of directory to be zipped
    zip_dir = zippath

    #Create a zip archive
    shutil.make_archive(output_file,'zip',zip_dir)
    
    '''
    # To unzip
    #zipped file full path
    zipped_file = f'{workdir}ks_bc_model.zip'

    #full path of directory to where the zipped files will be extracted
    extracted_shutil_dir = f'{workdir}ks_bc_model_newdir'

    #extract the files
    shutil.unpack_archive(zipped_file,extracted_shutil_dir,'zip')
    '''
    
def GetTimeZone():
    tz_flag = None
    tzlist = list(pytz.all_timezones)
    tzlist = list(map(lambda x: x, tzlist))
    df = pd.DataFrame(tzlist, columns = ['timezone'])
    print('Input your city or a city with same timezone as yours')
    print('   e.g. Manila has same timezone as Singapore\n')
    for i in range(5):
        city = input('City (Enter without keying in anything will display all): ')
    
        if len(city) > 0:
            if '+' in city:
                index = city.find('+')
                city = f'{city[:index]}\{city[index:]}'
            #endif

            filter=df['timezone'].str.contains(city, case=False)
            if len(df[filter]) > 0:
                display(df[filter])
                print()
                rownum =input\
                    ('Row Number (Enter without keying in anything will ask for city again): ')
                if len(rownum) > 0:
                    tz_flag = df.iloc[int(rownum),0]
                    print()
                    print(f'{color.bold}Your choice: {color.bdblue}{tz_flag}{color.end}')

                    # show local time to check that your tz_flag is correct
                    IST = pytz.timezone(tz_flag)  
                    local_time = dt.datetime.now(IST)  # get correct time as per local time zone
                    print(f'\n{color.bold}Your Current Local Time : {color.bdgreen}{local_time.strftime("%I:%M:%S %p")}{color.end}')
                    break
                #endif
            else:
                print(f'\nAttempt {i+1}/{len(range(5))}: Not Found, try again')
            #endif
        else:
            display(tzlist)
            break
        #endif
    #endfor
    
    return tz_flag
    
def GetRatio(num1, num2):
    # get ratio between two numbers

    # get greatest common denominator
    div = math.gcd(num1, num2)
    return f"{int(num1/div)}:{int(num2/div)}"
