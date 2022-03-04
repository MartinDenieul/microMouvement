
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy.fft as fft
import matplotlib.animation as animation
from IPython.display import HTML
from scipy import integrate
import numpy as np
import os
import quat
import vector
import rotmat

def matrix_rotation(X, Y, Z, W):
    matrix = []
    
    xx      = X * X;
    xy      = X * Y;
    xz      = X * Z;
    xw      = X * W;

    yy      = Y * Y;
    yz      = Y * Z;
    yw      = Y * W;

    zz      = Z * Z;
    zw      = Z * W;
    matrix = [[1 - 2 * ( yy + zz ), 2 * ( xy - zw ), 2 * ( xz + yw )],
              [2 * ( xy + zw ), 1 - 2 * ( xx + zz ), 2 * ( yz - xw )],
              [2 * ( xz - yw ), 2 * ( yz + xw ), 1 - 2 * ( xx + yy )]]
    
    
    return matrix

def trueAccel(df):
    #Convertion des accelerations en m.s-2
    ConvertAccelX=[]
    ConvertAccelY=[]
    ConvertAccelZ=[]
    for i in range(0,len(df)):
        ConvertAccelX.append(df['Accel X'][i]*9.81)
        ConvertAccelY.append(df['Accel Y'][i]*9.81)
        ConvertAccelZ.append(df['Accel Z'][i]*9.81)
    
    
    
    #Calcul des accelerations absolues en fonction de la matrice de rotation a chaque instant
    TrueAccelX=[]
    TrueAccelY=[]
    TrueAccelZ=[]
    
    #Definition matrice de rotation
    rotamatrix =[]
    for i in range(0, len(df)):
        rotamatrix.append(matrix_rotation(df['Quat X'][i], df['Quat Y'][i], df['Quat Z'][i], df['Quat W'][i]))
        
    for i in range(0,len(df)):
        TrueAccelX.append(ConvertAccelX[i]*rotamatrix[i][0][0] + ConvertAccelY[i]*rotamatrix[i][0][1] + ConvertAccelZ[i]*rotamatrix[i][0][2])
        TrueAccelY.append(ConvertAccelX[i]*rotamatrix[i][1][0] + ConvertAccelY[i]*rotamatrix[i][1][1] + ConvertAccelZ[i]*rotamatrix[i][1][2])
        TrueAccelZ.append(ConvertAccelX[i]*rotamatrix[i][2][0] + ConvertAccelY[i]*rotamatrix[i][2][1] + ConvertAccelZ[i]*rotamatrix[i][2][2])

    #Creation de la df translation
    dfTrueAccel = pd.DataFrame({'time':df['Time'], 'trueAccelX':TrueAccelX, 'trueAccelY':TrueAccelY, 'trueAccelZ':TrueAccelZ})
    return dfTrueAccel


def translation(df):
    x = df['time']
    y = df['trueAccelX']
    y_int = integrate.cumtrapz(y,x,initial = 0)
    y_intint = integrate.cumtrapz(y_int, x, initial = 0)
    
    translation = pd.DataFrame(columns = ['time', 'translationX', 'translationY', 'translationZ'])

    y2 = df['trueAccelY']
    y_int2 = integrate.cumtrapz(y2,x,initial = 0)
    y_intint2 = integrate.cumtrapz(y_int2, x, initial = 0)
    y3 = df['trueAccelZ']
    y_int3 = integrate.cumtrapz(y3, x, initial = 0)
    y_intint3 = integrate.cumtrapz(y_int3, x, initial = 0)

    translation['time'] = df['time']
    translation['translationX'] = y_intint
    translation['translationY'] = y_intint2
    translation['translationZ'] = y_intint3
    
    plt.subplot(221)
    plt.plot(translation['time'], translation['translationX'])
    plt.subplot(222)
    plt.plot(translation['time'], translation['translationY'])
    plt.subplot(223)
    plt.plot(translation['time'], translation['translationZ'])
    plt.subplot(224)
    plt.plot(translation['time'], translation['translationX'])
    plt.plot(translation['time'], translation['translationY'])
    plt.plot(translation['time'], translation['translationZ'])
    
    
    return translation


def calculAngle(df):
    t = df['Time']
    gyroX = df['Gyro X']
    gyroY = df['Gyro Y']
    gyroZ = df['Gyro Z']
    angleX = integrate.cumtrapz(gyroX,t,initial = 0)
    angleY = integrate.cumtrapz(gyroY, t, initial = 0)
    angleZ = integrate.cumtrapz(gyroZ, t, initial = 0)
    angle = pd.DataFrame(columns = ['time', 'angleX', 'angleY', 'angleZ'])
    angle['time'] = df['Time']
    angle['angleX'] = angleX
    angle['angleY'] = angleY
    angle['angleZ'] = angleZ
    
    plt.subplot(221)
    plt.plot(angle['time'], angle['angleX'])
    plt.title('Angle X')
    plt.subplot(222)
    plt.plot(angle['time'], angle['angleY'])
    plt.title('Angle Y')
    plt.subplot(223)
    plt.plot(angle['time'], angle['angleZ'])
    plt.title('Angle Z')
    plt.subplot(224)
    plt.plot(angle['time'], angle['angleX'])
    plt.plot(angle['time'], angle['angleY'])
    plt.plot(angle['time'], angle['angleZ'])
    plt.title('3 angles')
    
    return angle


def quaternion(df):
    plt.subplot(221)
    plt.plot(df['Time'], df['Quat X'])
    plt.title('Quat X')
    plt.subplot(222)
    plt.plot(df['Time'], df['Quat Y'])
    plt.title('Quat Y')
    plt.subplot(223)
    plt.plot(df['Time'], df['Quat Z'])
    plt.title('Quat Z', loc="left")
    plt.subplot(224)
    plt.plot(df['Time'], df['Quat W'])
    plt.title('Quat W', loc='right')
    
    
def window(a, w = 128, o = 64, copy = False):
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view
    

def diffT(df, w):
    Tmin = df[0]
    Tmax = df[w]
    diffT = df[w] - df[0]
    return diffT


def transform(window, diffT):
    #Calcul FFT
    Fwindow = 1/(diffT) * fft.fft(window, axis=1) 
    return Fwindow


def affichage(Fwindow, diffT, titre):
    N = len(Fwindow[0])
    fe = N/(diffT)
    f = np.arange(-fe/2.0, +fe/2.0, fe/N)
    FwindowShifted = fft.fftshift(Fwindow)
    plt.figure()
    plt.contourf(np.log(Fwindow.T), 128, cmap='jet')
    plt.title(titre)
    plt.figure()
    plt.plot(f, np.log(np.mean(np.absolute(FwindowShifted.T), axis=1)))
    plt.title(titre)
    
    
def fourierTransform(df):
    diff = diffT(df['Time'], 128)
    
    #AccelX
    windowAccelX = window(df['Accel X'], 128, 1, True)
    TwindowAccelX = transform(windowAccelX, diff)
    
    #AccelY
    windowAccelY = window(df['Accel Y'], 128, 1, True)
    TwindowAccelY = transform(windowAccelY, diff)
    
    #AccelZ
    windowAccelZ = window(df['Accel Z'], 128, 1, True)
    TwindowAccelZ = transform(windowAccelZ, diff)
    
    #GyroX
    windowGyroX = window(df['Gyro X'], 128, 1, True)
    TwindowGyroX = transform(windowGyroX, diff)
    
    #GyroY
    windowGyroY = window(df['Gyro Y'], 128, 1, True)
    TwindowGyroY = transform(windowGyroY, diff)
    
    #GyroZ
    windowGyroZ = window(df['Gyro Z'], 128, 1, True)
    TwindowGyroZ = transform(windowGyroZ, diff)
    
    
    
    affichage(TwindowAccelX, diff, 'Accel X')
    affichage(TwindowAccelY, diff, 'Accel Y')
    affichage(TwindowAccelZ, diff, 'Accel Z')
    affichage(TwindowGyroX, diff, 'Gyro X')
    affichage(TwindowGyroY, diff, 'Gyro Y')
    affichage(TwindowGyroZ, diff, 'Gyro Z')
    
    
def kinectHandData(df):
        
    rightHandDf = df['Right hand position'].str[1:-1].str.split(',', expand=True)
    rightHandDf.columns = ['translation X', 'translation Y', 'translation Z']
    leftHandDf = df['Left hand position'].str[1:-1].str.split(',', expand=True)
    leftHandDf.columns = ['translation X', 'translation Y', 'translation Z']
    rightHandDf.index = df['AnimationTime']
    leftHandDf.index = df['AnimationTime']
        
    return [rightHandDf, leftHandDf]


def midData(df, debut, fin):
    midX=[]
    midY = []
    midZ = []
    time =[]
    for i in df.index:
        if float(i) > debut and float(i)<fin:
            midX.append(df['translation X'][i])
            midY.append(df['translation Y'][i])
            midZ.append(df['translation Z'][i])
            time.append(i)
            
    return midX, midY, midZ, time


def trajectoireMain(leftX, leftY, rightX, rightY):
    trueLeftX=[]
    trueLeftY=[]
    
    for i in range(len(leftX)):
        trueLeftX.append(float(leftX[i]) - float(leftX[0]))
        trueLeftY.append(float(leftY[i]) - float(leftY[0]))
        
    trueRightX=[]
    trueRightY=[]
    
    for i in range(len(rightX)):
        
        trueRightX.append(float(rightX[i]) - float(leftX[0]))
        trueRightY.append(float(rightY[i]) - float(leftY[0]))
        
    plt.plot(trueLeftX, trueLeftY)
    plt.plot(trueRightX, trueRightY)
    plt.xlim(-10,20)
    plt.ylim(-10,20)
	
	
def kalmanFilter(df, data, ylim, ylim2, str):

    plt.rcParams['figure.figsize'] = (10, 8)

    # intial parameters
    n_iter = len(df[data])
    sz = (n_iter,) # size of array
    x = -0.02 # truth value 
    z = df[data] # observations (normal about x, sigma=0.1)

    Q = 1e-5 # process variance

    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor

    R = str**2 # estimate of measurement variance, change to see effect

    # intial guesses
    xhat[0] = 0.0
    P[0] = 1.0

    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q

        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]

    plt.figure()
    plt.plot(z,'k+',label=data +' capteur')
    plt.plot(xhat,'b-',label='post Kalman ' + data)
    #plt.axhline(x,color='g',label='truth value')
    plt.legend()
    plt.ylim(ylim,ylim2)
    plt.title('Kalman filter ' + data, fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel(data)

    #plt.figure()
    #valid_iter = range(1,n_iter) # Pminus not valid at step 0
    #plt.plot(valid_iter,Pminus[valid_iter],label='a priori error estimate')
    #plt.title('Estimated $\it{\mathbf{a \ priori}}$ error vs. iteration step', fontweight='bold')
    #plt.xlabel('Iteration')
    #plt.ylabel('$(Voltage)^2$')
    #plt.setp(plt.gca(),'ylim',[0,.01])
    #plt.show()
    
    return xhat
	
def enregistrerFT(df, dossier, nom):
    
    diff = diffT(df['Time'], 128)
    
    #AccelX
    windowAccelX = window(df['Accel X'], 128, 1, True)
    TwindowAccelX = transform(windowAccelX, diff)
    
    #AccelY
    windowAccelY = window(df['Accel Y'], 128, 1, True)
    TwindowAccelY = transform(windowAccelY, diff)
    
    #AccelZ
    windowAccelZ = window(df['Accel Z'], 128, 1, True)
    TwindowAccelZ = transform(windowAccelZ, diff)
    
    #GyroX
    windowGyroX = window(df['Gyro X'], 128, 1, True)
    TwindowGyroX = transform(windowGyroX, diff)
    
    #GyroY
    windowGyroY = window(df['Gyro Y'], 128, 1, True)
    TwindowGyroY = transform(windowGyroY, diff)
    
    #GyroZ
    windowGyroZ = window(df['Gyro Z'], 128, 1, True)
    TwindowGyroZ = transform(windowGyroZ, diff)
	
	#Creation du plot
    N = len(TwindowAccelX[0])
    fe = N/(diffT(df['Time'],128))
    f = np.arange(-fe/2.0, +fe/2.0, fe/N)
    filename='C:/Users/Martin/Desktop/liste_reference_AccelX.csv'
    df = pd.read_csv(filename, header=None)
    TwindowAccelXShifted = fft.fftshift(TwindowAccelX)
    plt.figure()
    plt.contourf(np.log(TwindowAccelXShifted.T), 128, cmap='jet')
    plt.colorbar()
    plt.savefig(dossier + '/' +  'contourf_AccelX' + nom + '.png')
    plt.figure()
    plt.plot(f, np.log(np.mean(np.absolute(TwindowAccelXShifted.T), axis=1)))
    plt.legend('Freq Accel X')
    plt.plot(f, df)
    plt.legend('Accel X reference')
    plt.legend()
    plt.title('Accel X')
    plt.savefig(dossier + '/' + 'AccelX' + nom +'.png')
	
    TwindowAccelYShifted = fft.fftshift(TwindowAccelY)
    plt.figure()
    plt.contourf(np.log(TwindowAccelYShifted.T), 128, cmap='jet')
    plt.colorbar()
    plt.savefig(dossier + '/' + 'contourf_AccelY' + nom + '.png')
    plt.figure()
    plt.plot(f, np.log(np.mean(np.absolute(TwindowAccelYShifted.T), axis=1)))
    plt.title('AccelY')
    plt.savefig(dossier + '/' + 'AccelY' + nom + '.png')
	
    TwindowAccelZShifted = fft.fftshift(TwindowAccelZ)
    plt.figure()
    plt.contourf( np.log(TwindowAccelZShifted.T), 128, cmap='jet')
    plt.colorbar()
    plt.savefig(dossier + '/' +  'contourf_AccelZ' + nom + '.png')
    plt.figure()
    plt.plot(f, np.log(np.mean(np.absolute(TwindowAccelZShifted.T), axis=1)))
    plt.title('Accel Z')
    plt.savefig(dossier + '/' + 'AccelZ' + nom +'.png')
	
	
    TwindowGyroXShifted = fft.fftshift(TwindowGyroX)
    plt.figure()
    plt.contourf(np.log(TwindowGyroXShifted.T), 128, cmap='jet')
    plt.colorbar()
    plt.savefig(dossier + '/' +  'contourf_GyroX' + nom + '.png')
    plt.figure()
    plt.plot(f, np.log(np.mean(np.absolute(TwindowGyroXShifted.T), axis=1)))
    plt.title('Gyro X')
    plt.savefig(dossier + '/'  + 'GyroX' + '.png')
    
    TwindowGyroYShifted = fft.fftshift(TwindowGyroY)
    plt.figure()
    plt.contourf(np.log(TwindowGyroYShifted.T), 128, cmap='jet')
    plt.colorbar()
    plt.savefig(dossier + '/' +  'contourf_GyroY' + '.png')
    plt.figure()
    plt.plot(f, np.log(np.mean(np.absolute(TwindowGyroYShifted.T), axis=1)))
    plt.title('Gyro Y')
    plt.savefig(dossier + '/' + 'GyroY' + nom + '.png')
    
    TwindowGyroZShifted = fft.fftshift(TwindowGyroZ)
    plt.figure()
    plt.contourf(np.log(TwindowGyroZShifted.T), 128, cmap='jet')
    plt.colorbar()
    plt.savefig(dossier + '/' +  'contourf_GyroZ' + nom + '.png')
    plt.figure()
    plt.plot(f, np.log(np.mean(np.absolute(TwindowGyroZShifted.T), axis=1)))
    plt.title('Gyro Z')
    plt.savefig(dossier + '/' + 'GyroZ' + nom + '.png')
	
def enregistrerTrajectoire(df, dossier):
	right, left = kinectHandData(df)
	
	leftX=[]
	leftY = []
	leftZ = []
	time =[]
	
	for i in left.index:
		leftX.append(left['translation X'][i])
		leftY.append(left['translation Y'][i])
		leftZ.append(left['translation Z'][i])
		time.append(i)
	
	rightX=[]
	rightY = []
	rightZ = []
	time =[]
	
	for i in right.index:
		rightX.append(right['translation X'][i])
		rightY.append(right['translation Y'][i])
		rightZ.append(right['translation Z'][i])
		time.append(i)
	
	
	trueLeftX=[]
	trueLeftY=[]
    
	for i in range(len(leftX)):
		trueLeftX.append(float(leftX[i]) - float(leftX[0]))
		trueLeftY.append(float(leftY[i]) - float(leftY[0]))
        
	trueRightX=[]
	trueRightY=[]
    
	for i in range(len(rightX)):
        
		trueRightX.append(float(rightX[i]) - float(leftX[0]))
		trueRightY.append(float(rightY[i]) - float(leftY[0]))
    
	plt.figure()
	plt.plot(trueLeftX, trueLeftY)
	plt.plot(trueRightX, trueRightY)
	plt.xlim(-10,20)
	plt.ylim(-10,20)
	plt.savefig(dossier + "/" +  "trajectoireMain.png")
	
def kalmanFilterDF(dfCapteur):
	kfdf = pd.DataFrame({'Time' : dfCapteur['Time'], 'Accel X':kalmanFilter(dfCapteur, 'Accel X', -0.5,0.5,0.01), 'Accel Y':kalmanFilter(dfCapteur, 'Accel Y', -0.5,0.5,0.01), 'Accel Z':kalmanFilter(dfCapteur, 'Accel Z', -0.5,0.5,0.01), 'Gyro X':kalmanFilter(dfCapteur, 'Gyro X', -10,10,0.1), 'Gyro Y':kalmanFilter(dfCapteur, 'Gyro Y', -10,10,0.1), 'Gyro Z':kalmanFilter(dfCapteur, 'Gyro Z', -10,10,0.1)}) 
	return kfdf
	
def synthese(dfCapteur, dfKinect, dossier):
	os.mkdir(dossier)
	os.mkdir(dossier + '/funambule')
	os.mkdir(dossier + '/marche')
	os.mkdir(dossier + '/funambule/main_droite')
	os.mkdir(dossier + '/funambule/main_gauche')
	os.mkdir(dossier + '/funambule/ceinture')
	os.mkdir(dossier + '/marche/main_droite')
	os.mkdir(dossier + '/marche/main_gauche')
	os.mkdir(dossier + '/marche/ceinture')
	enregistrerTrajectoire(dfKinect, dossier + '/funambule' )
	enregistrerFT(dfCapteur, dossier + '/funambule/main_droite', 'main_droite')
	dfCapteur.to_csv(dossier + '/funambule/main_droite/postKalmanData.csv')
	angle = calculAngle(dfCapteur)
	plt.figure()
	plt.plot(angle['time'], angle['angleX'])
	plt.plot(angle['time'], angle['angleY'])
	plt.plot(angle['time'], angle['angleZ'])
	plt.legend(['Angle X', 'Angle Y', 'Angle Z'])
	plt.savefig(dossier + '/funambule/main_droite/' + 'angle.png')
	
def allData(dffunambuledroite, dffunambulegauche, dffunambuleceinture, dfmarchedroite, dfmarchegauche, dfmarcheceinture, dffunambulekinect, dossier):

	#Creation des sous dossier
	os.mkdir(dossier)
	os.mkdir(dossier + '/funambule')
	os.mkdir(dossier + '/marche')
	os.mkdir(dossier + '/funambule/main_droite')
	os.mkdir(dossier + '/funambule/main_gauche')
	os.mkdir(dossier + '/funambule/ceinture')
	os.mkdir(dossier + '/marche/pied_droit')
	os.mkdir(dossier + '/marche/pied_gauche')
	os.mkdir(dossier + '/marche/ceinture')
	
	#Filtrage de kalman de toutes les df
	kffunambuledroite = kalmanFilterDF(dffunambuledroite)
	kffunambulegauche = kalmanFilterDF(dffunambulegauche)
	kffunambuleceinture = kalmanFilterDF(dffunambuleceinture)
	kfmarchedroite = kalmanFilterDF(dfmarchedroite)
	kfmarchegauche = kalmanFilterDF(dfmarchegauche)
	kfmarcheceinture = kalmanFilterDF(dfmarcheceinture)
	
	#Data funambule main droite > kffunambuledroite
	enregistrerFT(kffunambuledroite, dossier + '/funambule/main_droite', 'main_droite')
	kffunambuledroite.to_csv(dossier + '/funambule/main_droite/postKalmanData.csv')
	angle = calculAngle(kffunambuledroite)
	plt.figure()
	plt.plot(angle['time'], angle['angleX'])
	plt.plot(angle['time'], angle['angleY'])
	plt.plot(angle['time'], angle['angleZ'])
	plt.legend(['Angle X', 'Angle Y', 'Angle Z'])
	plt.savefig(dossier + '/funambule/main_droite/' + 'angles.png')
	
	#Data funambule main gauche > kffunambulegauche
	enregistrerFT(kffunambulegauche, dossier + '/funambule/main_gauche', 'main_gauche')
	kffunambulegauche.to_csv(dossier + '/funambule/main_gauche/postKalmanData.csv')
	angle = calculAngle(kffunambulegauche)
	plt.figure()
	plt.plot(angle['time'], angle['angleX'])
	plt.plot(angle['time'], angle['angleY'])
	plt.plot(angle['time'], angle['angleZ'])
	plt.legend(['Angle X', 'Angle Y', 'Angle Z'])
	plt.savefig(dossier + '/funambule/main_gauche/' + 'angles.png')
	
	#Data funambule ceinture > kffunambuleceinture
	enregistrerFT(kffunambuleceinture, dossier + '/funambule/ceinture', 'ceinture')
	kffunambuleceinture.to_csv(dossier + '/funambule/ceinture/postKalmanData.csv')
	angle = calculAngle(kffunambuleceinture)
	plt.figure()
	plt.plot(angle['time'], angle['angleX'])
	plt.plot(angle['time'], angle['angleY'])
	plt.plot(angle['time'], angle['angleZ'])
	plt.legend(['Angle X', 'Angle Y', 'Angle Z'])
	plt.savefig(dossier + '/funambule/ceinture/' + 'angles.png')
	
	#Data marche pied droit > kfmarchedroite
	enregistrerFT(kfmarchedroite, dossier + '/marche/pied_droit', 'pied_droit')
	kfmarchedroite.to_csv(dossier + '/marche/pied_droit/postKalmanData.csv')
	angle = calculAngle(kfmarchedroite)
	plt.figure()
	plt.plot(angle['time'], angle['angleX'])
	plt.plot(angle['time'], angle['angleY'])
	plt.plot(angle['time'], angle['angleZ'])
	plt.legend(['Angle X', 'Angle Y', 'Angle Z'])
	plt.savefig(dossier + '/marche/pied_droit/' + 'angles.png')
	
	#Data marche pied gauche > kfmarchegauche
	enregistrerFT(kfmarchegauche, dossier + '/marche/pied_gauche', 'pied_gauche')
	kfmarchegauche.to_csv(dossier + '/marche/pied_gauche/postKalmanData.csv')
	angle = calculAngle(kfmarchegauche)
	plt.figure()
	plt.plot(angle['time'], angle['angleX'])
	plt.plot(angle['time'], angle['angleY'])
	plt.plot(angle['time'], angle['angleZ'])
	plt.legend(['Angle X', 'Angle Y', 'Angle Z'])
	plt.savefig(dossier + '/marche/pied_gauche/' + 'angles.png')
	
	#Data marche ceinture > kfmarcheceinture
	enregistrerFT(kfmarcheceinture, dossier + '/marche/ceinture', 'ceinture')
	kfmarcheceinture.to_csv(dossier + '/marche/ceinture/postKalmanData.csv')
	angle = calculAngle(kfmarcheceinture)
	plt.figure()
	plt.plot(angle['time'], angle['angleX'])
	plt.plot(angle['time'], angle['angleY'])
	plt.plot(angle['time'], angle['angleZ'])
	plt.legend(['Angle X', 'Angle Y', 'Angle Z'])
	plt.savefig(dossier + '/marche/ceinture/' + 'angles.png')
	
	enregistrerTrajectoire(dffunambulekinect, dossier + '/funambule' )
	
def spectrogramme(df, exo, type, recouvr=64, ):
	kf = kalmanFilterDF(df)
	diff = diffT(kf['Time'], 128)
    
	#AccelX
	windowAccelX = window(kf['Accel X'], 128, recouvr, True)
	TwindowAccelX = transform(windowAccelX, diff)
	N = len(TwindowAccelX[0])
	fe = N/(diff)
	f = np.arange(-fe/2.0, +fe/2.0, fe/N)
	FwindowAccelXShifted = fft.fftshift(TwindowAccelX)
	liste = np.arange(0,len(TwindowAccelX)/2,0.5)
	freq = np.arange(0,fe,fe/N)
	X, Y = np.meshgrid(liste, freq)
	plt.figure()
	plt.contourf(X,Y,np.log(TwindowAccelX.T), 128, cmap='jet' , vmin=-10, vmax=5)
	plt.ylabel('Frequency (Hz)')
	plt.xlabel('Time (s)')
	plt.title('Accel X ' + exo + " " + type)
	plt.ylim(0,64)
	plt.figure()
	plt.plot(f, np.log(np.mean(np.absolute(FwindowAccelXShifted.T), axis=1)))
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Amplitude')
	plt.title('Accel X ' + exo + " " + type)
	
	#AccelY
	windowAccelY = window(kf['Accel Y'], 128, recouvr, True)
	TwindowAccelY = transform(windowAccelY, diff)
	N = len(TwindowAccelY[0])
	fe = N/(diff)
	f = np.arange(-fe/2.0, +fe/2.0, fe/N)
	FwindowAccelYShifted = fft.fftshift(TwindowAccelY)
	liste = np.arange(0,len(TwindowAccelY)/2,0.5)
	freq = np.arange(0,fe,fe/N)
	X, Y = np.meshgrid(liste, freq)
	plt.figure()
	plt.contourf(X,Y,np.log(TwindowAccelY.T), 128, cmap='jet' , vmin=-10, vmax=5)
	plt.ylabel('Frequency (Hz)')
	plt.xlabel('Time (s)')
	plt.title('Accel Y ' + exo + " " + type)
	plt.ylim(0,64)
	plt.figure()
	plt.plot(f, np.log(np.mean(np.absolute(FwindowAccelYShifted.T), axis=1)))
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Amplitude')
	plt.title('Accel Y ' + exo + " " + type)
	
	#AccelZ
	windowAccelZ = window(kf['Accel Z'], 128, recouvr, True)
	TwindowAccelZ = transform(windowAccelZ, diff)
	N = len(TwindowAccelZ[0])
	fe = N/(diff)
	f = np.arange(-fe/2.0, +fe/2.0, fe/N)
	FwindowAccelZShifted = fft.fftshift(TwindowAccelZ)
	liste = np.arange(0,len(TwindowAccelZ)/2,0.5)
	freq = np.arange(0,fe,fe/N)
	X, Y = np.meshgrid(liste, freq)
	plt.figure()
	plt.contourf(X,Y,np.log(TwindowAccelZ.T), 128, cmap='jet' , vmin=-10, vmax=5)
	plt.ylabel('Frequency (Hz)')
	plt.xlabel('Time (s)')
	plt.title('Accel Z ' + exo + " " + type)
	plt.ylim(0,64)
	plt.figure()
	plt.plot(f, np.log(np.mean(np.absolute(FwindowAccelZShifted.T), axis=1)))
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Amplitude')
	plt.title('Accel Z ' + exo + " " + type)
	
	#GyroX
	windowGyroX = window(kf['Gyro X'], 128, recouvr, True)
	TwindowGyroX = transform(windowGyroX, diff)
	N = len(TwindowGyroX[0])
	fe = N/(diff)
	f = np.arange(-fe/2.0, +fe/2.0, fe/N)
	FwindowGyroXShifted = fft.fftshift(TwindowGyroX)
	liste = np.arange(0,len(TwindowGyroX)/2,0.5)
	freq = np.arange(0,fe,fe/N)
	X, Y = np.meshgrid(liste, freq)
	plt.figure()
	plt.contourf(X,Y,np.log(TwindowGyroX.T), 128, cmap='jet' , vmin=-2, vmax=8)
	plt.ylabel('Frequency (Hz)')
	plt.xlabel('Time (s)')
	plt.title('Gyro X ' + exo + " " + type)
	plt.ylim(0,64)
	plt.figure()
	plt.plot(f, np.log(np.mean(np.absolute(FwindowGyroXShifted.T), axis=1)))
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Amplitude')
	plt.title('Gyro X ' + exo + " " + type)
	
	#GyroY
	windowGyroY = window(kf['Gyro Y'], 128, recouvr, True)
	TwindowGyroY = transform(windowGyroY, diff)
	N = len(TwindowGyroY[0])
	fe = N/(diff)
	f = np.arange(-fe/2.0, +fe/2.0, fe/N)
	FwindowGyroYShifted = fft.fftshift(TwindowGyroY)
	liste = np.arange(0,len(TwindowGyroY)/2,0.5)
	freq = np.arange(0,fe,fe/N)
	X, Y = np.meshgrid(liste, freq)
	plt.figure()
	plt.contourf(X,Y,np.log(TwindowGyroY.T), 128, cmap='jet' , vmin=-2, vmax=8)
	plt.ylabel('Frequency (Hz)')
	plt.xlabel('Time (s)')
	plt.title('Gyro Y ' + exo + " " + type)
	plt.ylim(0,64)
	plt.figure()
	plt.plot(f, np.log(np.mean(np.absolute(FwindowGyroYShifted.T), axis=1)))
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Amplitude')
	plt.title('Gyro Y ' + exo + " " + type)
	
	#GyroZ
	windowGyroZ = window(kf['Gyro Z'], 128, recouvr, True)
	TwindowGyroZ = transform(windowGyroZ, diff)
	N = len(TwindowGyroZ[0])
	fe = N/(diff)
	f = np.arange(-fe/2.0, +fe/2.0, fe/N)
	FwindowGyroZShifted = fft.fftshift(TwindowGyroZ)
	liste = np.arange(0,len(TwindowGyroZ)/2,0.5)
	freq = np.arange(0,fe,fe/N)
	X, Y = np.meshgrid(liste, freq)
	plt.figure()
	plt.contourf(X,Y,np.log(TwindowGyroZ.T), 128, cmap='jet' , vmin=-2, vmax=8)
	plt.ylabel('Frequency (Hz)')
	plt.xlabel('Time (s)')
	plt.title('Gyro Z ' + exo + " " + type)
	plt.ylim(0,64)
	plt.figure()
	plt.plot(f, np.log(np.mean(np.absolute(FwindowGyroZShifted.T), axis=1)))
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Amplitude')
	plt.title('Gyro Z ' + exo + " " + type)
	
	#normAccel
	windowAccelNorm = window(df['normAccel'], 128, recouvr, True)
	TwindowAccelNorm = transform(windowAccelNorm, diff)
	N = len(TwindowAccelNorm[0])
	fe = N/(diff)
	f = np.arange(-fe/2.0, +fe/2.0, fe/N)
	FwindowAccelNormShifted = fft.fftshift(TwindowAccelNorm)
	liste = np.arange(0,len(TwindowAccelNorm)/2,0.5)
	freq = np.arange(0,fe,fe/N)
	X, Y = np.meshgrid(liste, freq)
	plt.figure()
	plt.contourf(X,Y,np.log(TwindowAccelNorm.T), 128, cmap='jet' , vmin=-10, vmax=5)
	plt.ylabel('Frequency (Hz)')
	plt.xlabel('Time (s)')
	plt.title('Norm Accel ' + exo + " " + type)
	plt.ylim(0,64)
	plt.figure()
	plt.plot(f, np.log(np.mean(np.absolute(FwindowAccelNormShifted.T), axis=1)))
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Amplitude')
	plt.title('Norm Accel ' + exo + " " + type)
	
	return FwindowAccelXShifted, FwindowAccelYShifted, FwindowAccelZShifted, FwindowAccelNormShifted
	
	
def normAccel(df):
	df['kfX'] = kalmanFilter(df,'Accel X', -0.5,0.5,0.01)
	df['kfY'] = kalmanFilter(df,'Accel Y', -0.5,0.5,0.01)
	df['kfZ'] = kalmanFilter(df,'Accel Z', -0.5,0.5,0.01)
	df['normAccel'] = (df['kfX']**2 + df['kfY']**2 + df['kfZ']**2).apply(math.sqrt)
	return df

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import scipy as sp
from scipy import constants     # for "g"
from scipy.integrate import cumtrapz
import re

# The following construct is required since I want to run the module as a script
# inside the skinematics-directory
import os
import sys

import vector, quat, rotmat

def analytical(R_initialOrientation=np.eye(3),
               omega=np.zeros((5,3)),
               initialPosition=np.zeros(3),
               accMeasured=np.column_stack((np.zeros((5,2)), 9.81*np.ones(5))),
               rate=100):
    ''' Reconstruct position and orientation with an analytical solution,
    from angular velocity and linear acceleration.
    Assumes a start in a stationary position. No compensation for drift.
    Parameters
    ----------
    R_initialOrientation: ndarray(3,3)
        Rotation matrix describing the initial orientation of the sensor,
        except a mis-orienation with respect to gravity
    omega : ndarray(N,3)
        Angular velocity, in [rad/s]
    initialPosition : ndarray(3,)
        initial Position, in [m]
    accMeasured : ndarray(N,3)
        Linear acceleration, in [m/s^2]
    rate : float
        sampling rate, in [Hz]
    Returns
    -------
    q : ndarray(N,3)
        Orientation, expressed as a quaternion vector
    pos : ndarray(N,3)
        Position in space [m]
    vel : ndarray(N,3)
        Velocity in space [m/s]
    Example
    -------
     
    >>> q1, pos1 = analytical(R_initialOrientation, omega, initialPosition, acc, rate)
    '''

    if omega.ndim == 1:
        raise ValueError('The input to "analytical" requires matrix inputs.')
        
    # Transform recordings to angVel/acceleration in space --------------

    # Orientation of \vec{g} with the sensor in the "R_initialOrientation"
    g = constants.g
    g0 = np.linalg.inv(R_initialOrientation).dot(np.r_[0,0,g])

    # for the remaining deviation, assume the shortest rotation to there
    q0 = vector.q_shortest_rotation(accMeasured[0], g0)    
    
    q_initial = rotmat.convert(R_initialOrientation, to='quat')
    
    # combine the two, to form a reference orientation. Note that the sequence
    # is very important!
    q_ref = quat.q_mult(q_initial, q0)
    
    # Calculate orientation q by "integrating" omega -----------------
    q = quat.calc_quat(omega, q_ref, rate, 'bf')

    # Acceleration, velocity, and position ----------------------------
    # From q and the measured acceleration, get the \frac{d^2x}{dt^2}
    g_v = np.r_[0, 0, g] 
    accReSensor = accMeasured - vector.rotate_vector(g_v, quat.q_inv(q))
    accReSpace = vector.rotate_vector(accReSensor, q)

    # Make the first position the reference position
    q = quat.q_mult(q, quat.q_inv(q[0]))

    # compensate for drift
    #drift = np.mean(accReSpace, 0)
    #accReSpace -= drift*0.7

    # Position and Velocity through integration, assuming 0-velocity at t=0
    vel = np.nan*np.ones_like(accReSpace)
    pos = np.nan*np.ones_like(accReSpace)

    for ii in range(accReSpace.shape[1]):
        vel[:,ii] = cumtrapz(accReSpace[:,ii], dx=1./rate, initial=0)
        pos[:,ii] = cumtrapz(vel[:,ii],        dx=1./rate, initial=initialPosition[ii])

    return (q, pos, vel)
	
def acceleration(df, lenCalib, stdAccel, stdGyro):

	meanGyroX = df['Gyro X'][0:lenCalib].mean(axis=0)
	meanGyroY = df['Gyro Y'][0:lenCalib].mean(axis=0)
	meanGyroZ = df['Gyro Z'][0:lenCalib].mean(axis=0)
	df['Gyro X'] = df['Gyro X'] - meanGyroX
	df['Gyro Y'] = df['Gyro Y'] - meanGyroY
	df['Gyro Z'] = df['Gyro Z'] - meanGyroZ
	
	meanAccelX = df['Accel X'][0:lenCalib].mean(axis=0)
	meanAccelY = df['Accel Y'][0:lenCalib].mean(axis=0)
	meanAccelZ = (df['Accel Z'][0:lenCalib] - 1).mean(axis=0)
	df['Accel X'] = df['Accel X'] - meanAccelX
	df['Accel Y'] = df['Accel Y'] - meanAccelY
	df['Accel Z'] = df['Accel Z'] - meanAccelZ 
	
	kf = pd.DataFrame()
	kf['Accel X'] = kalmanFilter(df,'Accel X', -2,2,stdAccel)
	kf['Accel Y'] = kalmanFilter(df, 'Accel Y', -2,2,stdAccel)
	kf['Accel Z'] = kalmanFilter(df,'Accel Z', -2,2,stdAccel)
	kf['Gyro X'] = kalmanFilter(df,'Gyro X', -5,5,stdGyro)
	kf['Gyro Y'] = kalmanFilter(df,'Gyro Y', -5,5,stdGyro)
	kf['Gyro Z'] = kalmanFilter(df,'Gyro Z', -5,5,stdGyro)
	
	gyroDf = pd.DataFrame({'Gyro X' : kf['Gyro X'].apply(math.radians), 'Gyro Y' : kf['Gyro Y'].apply(math.radians), 'Gyro Z' : kf['Gyro Z'].apply(math.radians)})
	gyroDf.index = df['Time']
	gyro = gyroDf.values
	
	accelDf = pd.DataFrame({'Accel X' : kf['Accel X']*9.80665, 'Accel Y' : kf['Accel Y']*9.80665, 'Accel Z' : kf['Accel Z']*9.80665})
	accelDf.index = df['Time']
	accel = accelDf.values
	
	q, vit, pos = analytical(R_initialOrientation = np.array([[1,0,0],[0,1,0],[0,0,1]]), omega = gyro, initialPosition = (0,0,0), accMeasured = accel, rate = 128)
	
	rotmatrix = quat.convert(q)
	
	rotation = []
	for i in range(len(rotmatrix)):
		rotation.append(np.reshape(rotmatrix[i],(3,3)))
		
	trueAccel =[]
	for i in range(len(df)):
		trueAccel.append(np.dot(rotation[i],accel[i]))
	
	return trueAccel