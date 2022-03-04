# //////////////////////////////////////////////////////////////////////////////////

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import micro

# //////////////////////////////////////////////////////////////////////////////////

'''Installer le narrative ici'''

# //////////////////////////////////////////////////////////////////////////////////
    
def acclValues(dictfilenames,lenCalib):

####################################################################################
#                                                                                  #
#  Reading, editing and plotting the values yielded by all three accelerometers.   #
#  Parameters                                                                      #
#  ------------------------------------------------------------------------------  #
#  dictfilenames : dict ------------------- Qualifying the CSV files to be read.   #
#  lenCalib : int ------------ The synchronised start index for the actual data.   #
#  Returns                                                                         #
#  ------------------------------------------------------------------------------  #
#  true : DataFrame ----------------- The amended, isolated acceleration values.   #
#                                                                                  #
####################################################################################
    
    print("\n                                   \033[1m1. 📡 It alls starts with the accelerometers. 📡\033[0m")
    print("------------------------------------------------------------------------------------------------------------------\n")
    
    allthree = {}
    for filename, quality in dictfilenames.items():
        print("\033[1mConsidering the "+quality+" signal.\033[0m") # ---------------
        print("Reading and recomputing the accelerometers' measured values.") # ----
        print("Retrieving CSV file...") # -----------------------------------------
        acci = micro.readAcclValues(filename)
        print("Computing compensated accelerations...") # --------------------------
        trac = pd.DataFrame(np.array(micro.acceleration(acci,lenCalib))[lenCalib:])
        trac.columns = ['AccX','AccY','AccZ']
        
        # remplacer par fonction de GridSearch si besoin
        print("Straightening things out...") # -------------------------------------
        trueAccelX = trac['AccX'] - micro.kalmanFilter(trac, 'AccX', 0.25)
        trueAccelY = trac['AccY'] - micro.kalmanFilter(trac, 'AccY', 0.25)
        trueAccelZ = trac['AccZ'] - micro.kalmanFilter(trac, 'AccZ', 0.25)
        acci = acci.drop(range(lenCalib), axis=0)
        acci = acci.reset_index(drop = True)
        true = pd.DataFrame({'AccX':trueAccelX, 'AccY':trueAccelY, 'AccZ':trueAccelZ, 'Time':acci['Time']})
        
        print("Showing results.") # ------------------------------------------------
        accs = plt.figure(figsize=(16,4))
        ax0 = accs.add_subplot(131)
        plt.plot(trac['AccX'],color='papayawhip')
        plt.plot(true['AccX'],color='sandybrown')
        plt.plot(acci['AccX'],color='floralwhite')
        ax0.set_title('Successive renderings of x-axis acceleration ('+quality+').')
        ax1 = accs.add_subplot(132)
        plt.plot(trac['AccY'],color='lightcoral')
        plt.plot(true['AccY'],color='firebrick')
        plt.plot(acci['AccY'],color='mistyrose')
        ax1.set_title('Successive renderings of y-axis acceleration ('+quality+').')
        ax2 = accs.add_subplot(133)
        plt.plot(trac['AccZ'],color='darkkhaki')
        plt.plot(true['AccZ'],color='olive')
        plt.plot(acci['AccZ'],color='beige')
        ax2.set_title('Successive renderings of z-axis acceleration ('+quality+').')
        plt.tight_layout()
        plt.show()
        print("Done with the "+quality+".\n") # ------------------------------------
        true['Time'] -= true['Time'][0]
        allthree[quality] = true
    # Renaming for better handling. ------------------------------------------------
    allthree['left wrist'].columns=['LacX','LacY','LacZ','Time']
    allthree['right wrist'].columns=['RacX','RacY','RacZ','Time']
    allthree['waist'].columns=['MacX','MacY','MacZ','Time']
    print("                                                         \033[1mDone.\033[0m")
    print("------------------------------------------------------------------------------------------------------------------\n")
    return list(allthree.values())
    
# //////////////////////////////////////////////////////////////////////////////////

'''commentaires texte'''

# //////////////////////////////////////////////////////////////////////////////////

def knctValues(filename):

####################################################################################
#                                                                                  #
#  Reading and cleaning the values provided by the Kinect sensor.                  #
#  Parameters                                                                      #
#  ------------------------------------------------------------------------------  #
#  filename : string ---------------------------------- The CSV file being read.   #
#  Returns                                                                         #
#  ------------------------------------------------------------------------------  #
#  knct : DataFrame ----------------- The amended, isolated acceleration values.   #  
#  start : int --------------- The amount of rows subtracted from the DataFrame.   #                                                                           #                                                                         #
#                                                                                  #
####################################################################################

    print("\n                               \033[1m2. 📡 We then turn to the Kinect RGB-depth sensor. 📡\033[0m")
    print("------------------------------------------------------------------------------------------------------------------\n")
    
    # Skipping non-numerical data. -------------------------------------------------
    init = pd.read_csv(""+filename,skiprows=[0,1,2,3,4])
    init = init[init['Kinect FrameNumber'].apply(lambda x: x.isnumeric())]
    # Pauses are now invisible. ----------------------------------------------------
    print("Separating the walks...") # ---------------------------------------------
    # Getting the start and stop indexes for the six "Funambule" series. -----------
    reft = micro.consecutiveSeries(init['Kinect FrameNumber'].astype(int))
    # Cleaning the DataFrame by skipping any untimed (or badly timed) rows. --------
    clen, zerolist, start = micro.cleanupRef(init,reft,'AnimationTime')
    print("Restoring the timeflow...") # -------------------------------------------
    clen = micro.timeRestore(zerolist,clen)
    print("Saving up the data...") # -----------------------------------------------
    right, left, waist = micro.kinectWristData(clen)
    # Arranging it in a single DataFrame. ------------------------------------------
    knct = pd.concat([right, left, waist], axis=1, sort=False)
    knct['Time'] = clen['RealTime']
    knct.loc[0,'Time'] = 0.0
    print("Showing results.") # ----------------------------------------------------
    accs = plt.figure(figsize=(16,4))
    ax0 = accs.add_subplot(131)
    plt.plot(knct['LwZ'][1:]-1180,color='olive')
    plt.plot(knct['LwY'][1:],color='firebrick')
    plt.plot(knct['LwX'][1:],color='sandybrown')
    ax0.set_title('Kinect-based x, y, and z-axis acceleration (left wrist).')
    ax1 = accs.add_subplot(132)
    plt.plot(knct['RwZ'][1:]-1180,color='olive')
    plt.plot(knct['RwY'][1:],color='firebrick')
    plt.plot(knct['RwX'][1:],color='sandybrown')
    ax1.set_title('Kinect-based x, y, and z-axis acceleration (right wrist).')
    ax2 = accs.add_subplot(133)
    plt.plot(knct['MwZ'][1:]-1180,color='olive')
    plt.plot(knct['MwY'][1:],color='firebrick')
    plt.plot(knct['MwX'][1:],color='sandybrown')
    ax2.set_title('Kinect-based x, y, and z-axis acceleration (waist).')
    plt.tight_layout()
    plt.show()
    print("                                                         \033[1mDone.\033[0m")
    print("------------------------------------------------------------------------------------------------------------------\n")
    return knct, start   

# //////////////////////////////////////////////////////////////////////////////////

'''commentaires texte'''

# //////////////////////////////////////////////////////////////////////////////////
    
def alignSensors(accl,knct,start):
    print("\n                                      \033[1m3. 🕒 An endeavour to align the sensors. 🕒\033[0m")
    print("------------------------------------------------------------------------------------------------------------------\n")
    print("Preparing interpolation...") # ------------------------------------------  
    finl = micro.shareTime(knct,accl,start)
    print("Computing the shift...") # ----------------------------------------------
    shift = micro.correlData(finl)
    print("Improving precision...") # ----------------------------------------------
    prec = micro.integrateData(knct,accl,shift)
    print("                                                        \033[1mDone.\033[0m")
    print("------------------------------------------------------------------------------------------------------------------\n")
    return prec
    
# //////////////////////////////////////////////////////////////////////////////////
    
def computeFourier(df):    
    
####################################################################################
#                                                                                  #
#  Computing and displaying the Fourier Transforms for all the time series.        #
#  Parameters                                                                      #
#  ------------------------------------------------------------------------------  #
#  df : pd.DataFrame ----------------------------------- The DataFrame envelope.   #
#  Example                                                                         #
#  ------------------------------------------------------------------------------  #
#  >>>                                                                             #
#                                                                                  #
####################################################################################
    
    diff = diffT(df['Time'], 128)
    
    # Initialisation. --------------------------------------------------------------
    windowAccX = micro.window(df['AccX'], 128, 1, True)
    windowAccY = micro.window(df['AccY'], 128, 1, True)
    windowAccZ = micro.window(df['AccZ'], 128, 1, True)
    windowGyrX = micro.window(df['GyrX'], 128, 1, True)
    windowGyrY = micro.window(df['GyrY'], 128, 1, True)
    windowGyrZ = micro.window(df['GyrZ'], 128, 1, True)
    
    # Initialisation. --------------------------------------------------------------
    TwindowAccX = micro.transform(windowAccX, diff)
    TwindowAccY = micro.transform(windowAccY, diff) 
    TwindowAccZ = micro.transform(windowAccZ, diff) 
    TwindowGyrX = micro.transform(windowGyrX, diff)    
    TwindowGyrY = micro.transform(windowGyrY, diff)    
    TwindowGyrZ = micro.transform(windowGyrZ, diff) 
    
    # Initialisation. --------------------------------------------------------------
    micro.affiFourier(TwindowAccX, diff, 'AccX')
    micro.affiFourier(TwindowAccY, diff, 'AccY')
    micro.affiFourier(TwindowAccZ, diff, 'AccZ')
    micro.affiFourier(TwindowGyrX, diff, 'GyrX')
    micro.affiFourier(TwindowGyrY, diff, 'GyrY')
    micro.affiFourier(TwindowGyrZ, diff, 'GyrZ')	 
    
    
def normAccel(df):
    df['kfX'] = micro.kalmanFilter(df,'AccX', -0.5,0.5, 0.01)
    df['kfY'] = micro.kalmanFilter(df,'AccY', -0.5,0.5, 0.01)
    df['kfZ'] = micro.kalmanFilter(df,'AccZ', -0.5,0.5, 0.01)
    df['normAccel'] = (df['kfX']**2 + df['kfY']**2 + df['kfZ']**2).apply(math.sqrt)
    return df
    
    
def computeSpectre(df, exo, type, recouvr=64, ):
	
####################################################################################
#                                                                                  #
#  Computing and displaying the Fourier Transforms for all the time series.        #
#  Parameters                                                                      #
#  ------------------------------------------------------------------------------  #
#  data : pd.DataFrame --------------------------------- The DataFrame envelope.   #
#  etc. etc.
#                                                                                  #
####################################################################################
    
    kf = pd.DataFrame({'Time' : df['Time'], 'AccX':kalmanFilter(df, 'AccX',0.01), 'AccY':kalmanFilter(df, 'AccY',0.01), 'AccZ':kalmanFilter(df, 'AccZ', 0.01), 'GyrX':kalmanFilter(df, 'GyrX', 0.1), 'GyrY':kalmanFilter(df, 'GyrY',0.1), 'GyrZ':kalmanFilter(df, 'GyrZ',0.1)}) (df)
    diff = diffT(kf['Time'], 128)
    
    # Initialisation. --------------------------------------------------------------
    windowAccX = micro.window(kf['AccX'], 128, recouvr, True)
    windowAccY = micro.window(kf['AccY'], 128, recouvr, True)
    windowAccZ = micro.window(kf['AccZ'], 128, recouvr, True)
    windowGyrX = micro.window(kf['GyrX'], 128, recouvr, True)
    windowGyrY = micro.window(kf['GyrY'], 128, recouvr, True)
    windowGyrZ = micro.window(kf['GyrZ'], 128, recouvr, True)
    windowAccN = micro.window(df['normAccel'], 128, recouvr, True)
    
    # Initialisation. --------------------------------------------------------------
    TwindowAccX = micro.transform(windowAccX, diff)
    TwindowAccY = micro.transform(windowAccY, diff)
    TwindowAccZ = micro.transform(windowAccZ, diff)
    TwindowGyrX = micro.transform(windowGyrX, diff)
    TwindowGyrY = micro.transform(windowGyrY, diff)
    TwindowGyrZ = micro.transform(windowGyrZ, diff)
    TwindowAccN = micro.transform(windowAccN, diff)
    
    # Initialisation. --------------------------------------------------------------
    micro.affiSpectre(TwindowAccX,diff,'AccX')
    micro.affiSpectre(TwindowAccY,diff,'AccY')
    micro.affiSpectre(TwindowAccZ,diff,'AccZ')
    micro.affiSpectre(TwindowGyrX,diff,'GyrX')
    micro.affiSpectre(TwindowGyrY,diff,'GyrY')
    micro.affiSpectre(TwindowGyrZ,diff,'GyrZ')
    micro.affiSpectre(TwindowAccN,diff,'AccN')
    
# //////////////////////////////////////////////////////////////////////////////////
