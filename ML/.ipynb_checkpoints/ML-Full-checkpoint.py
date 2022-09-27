
####----------------------------------------------------------------------------

#%%LESSON 1
#####################
#RULES       Traditional    ---->answers
#DATA        Programing

#burada kuralı ne koyarsanız çıktıyı o verir 
#örk:data=2 ve 3 rules=add  answers=5 olur 


#ANSWERS      Machine    ------------>rules 
#DATA         Learning 

#Algoritma üzerinden başka bir algoritma üretir
#örk: answers:9 7 8 data:6 3 - 4 2 - 4 4  data ve aswers a bakarak add  rule unu bulur 


#What is Machine Learning?
#*Recommendation engines
#*Customer Churn
#*New Pricing Models 
#*Email Spam Filtering 
#*Material and Stock Estimates 
#*Pattern and Image Recognıtion
#*Predictive iventory Plan
#*Purchasing Trends 
#*Credit Scoring 


#independent Variables(bağımsız değişken "X")Features
#Dependent Variables(bağımlı değişken "y")Target-Label 

#test:test için seçilecek satırlar(gözlem)
#train:eğitim için seçilecek satırlar 
#train ve test datası randomly ayrılır 


#   supervised learning overview
#supervised learning(denetimli öğrenme):data başka bir machine learning algoritması 
#tarafından yada kişiler tarafınfan label lanmış.
#feature ları kullanarak algoritmayı çalıştırır ve ortaya bir rule koyar 
#rule data gibi bir satır verildiğinde  o datanın ne olduğu çıktısını verir 

#*it is the process of learning from labeled observations 
#*labels teach the algorithm how to label the observations 


#   supervised learning overview

#unsupervised learning(denetimsiz öğrenme):
#*label yoktur verilen datayı çıktı olarak kümelere ayırır 
#*in unsupervised learning the machine learns from unlabeled data 
#*there is no training data for unsupervised learning 


#Correlation:İki nicel değişken arasındaki ilişkinin gücünü verir.
#r ile gösterilir.-1 ile +1 arasında gösterilir 

#güçlü      ilişki yok       güçlü  
# -1           0               1

#Correlation yoksa lineer regresyonda olmaz 
  
#*the correlation summarizes the direction of the association between 
#two quantitative variables and the strenght of its linear trend  
#Regression:Bit feature da bi değişiklik olurken diğer feature ın 
#nasıl etkilendiğine bakar

#Correlation(r)
#*iki veri arasındaki ilişkiye bakar 
#*variable lar birlikte hareket eder(scatterplot ortaya çıkar)
#*data -1 ile +1 arasında bir rakamla ifade edilir
#*is there a relationship between X and y

#Regression 
#*Bir feature ın diğer feature ı nasıl etkilediğine bakar 
#*Cause and effect üzerine çalışır(etki-tepki)
#*data bir line ile ifade edilir 
#*What is the relationship between X and y

#Linear Regression:istatiksel regresyon modellerini kullanarak tahmin analizi yapar
#*statistical regression method used for predictive analysis 
#*shows the relationship between the independent varianle and the dependet variable 
#*contiunes variable üzerinden çalışır.

#Linear Regression Theory
# Y=bı+b1X (regression equation)
 
#ei=Yi-^Yi  random error 
#bi=slope eğri eğimi  
#b0=intercept x in sıfır olduğu yerde y nin alacağı değer 
#*amaç bütün noktalara en yakın geçecek çizgiyi bulmak

#method of least squares:
#Σei^2=Σ(y−yˆ)^2

#Gradient Descent:
#*Gradient descent is an algorithm that finds best fit line for given training 
#dataset
#*Error leri 0 a yaklaştırmaya çalışır.En küçük olduğu yerde durur ve line ı çizer 

#The Coefficient of Determination (R^2):
#*Elimizdeki data ile tahmin etmek istediğimizin ne kadarını karşılıyoruz 
#0-hiç tanımlamaz 
#1-çok iyi tanımlar 

#*modelimiz iyi mi kötü mü ? bunun için kullanılır 

#R^2=1-()/(Σ(yi−y)^2)

#yiˆ :y lerin tahmini 
#y   :y lerin ortalaması 

#simple linear regression: Y=bı+b1X
#multiple linear regression: y=bo+b1*x1+b2*x2+.....+bn*xn
#y->dependent variable 
#x1..Xn -->independent variables 

#Regression Error Metriks:

#1)Mean Absulate Error(MAE):
#    1/nΣ =|yi-yi^|
#dezavantajı:Outlier hatalarını minimize etmez(cezalandırmaz)

#2)Mean Squared Error(MSE):Hatanın varyansı 
    
# 1/nΣ(yi-yi^)^2
 
#*hataların karesi alındığı için cezalandırması iyi 
#dezavantajı:açıklamsı zor 

#3)Root Mean Squared Error(RMSE):Hatanın standart sapması 
#√(1/nΣ(yi-yi^)^2)
#*hem hatayı cezalandırır hem açıklaması kolaydır 

### Scikit-Learn Kütüphanesi:
#*Supervised ve unsupervised Learning i destekler. 
#-model fitting 
#-data preprocessing 
#-model selection 
#-evaluation  
#aşamalarında kullanılır 

# ML Aşamaları 
 
#1)import aşamaları 

#2)Splitting Data (datayı bölme)
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=30,random_state=42)

#3)Fitting Data(Model Kurma)

#model.fit(X_train,y_train)
#logmodel=LogisticRegression() ---->önce model oluşturulur 
#logmodel.fit(X_train,y_train)---->sonra bu modelin içine X_train,y_train atılır

#4)Predicting Data(Fit te öğrendiklerine göre tahmin yapar)
#    predictions=model.predict(X_test)
    
#5)Probability of Predicted Data 
#    model.predict_proba(X_test)
#*bu adım ,classification problemlerinde olasılık istendiğinde kullanılır 

#6)Evaluation(Değerlendirme)
#*MAE,MES,RMSE kullanılarak evaluate işlemi yapılır """

#%%%
#**********NOTEBOOK 1*****************
# Elimizdeki data çok büyükse Deep Learning ile çalışılır
# ML de structured datalarla çalışılır. DL de farklı datalarla da çalışılır
# Correlation & Linearity
# Korelasyon : Iki sayısal değişken arasındaki ilişki(Yönü(pozitif ve negatif) ve kuvveti(weak and strong) var)
# .. -1, +1 arasında değerler alır. Simgesi r dir
# Linearity  : iki feature arasında lineer bir ilişkiden bahsetmek için bunu bir doğru
# .. üzerinde ifade edebilmemiz lazım.(y=ax+b)
# .. Lineer ilişki
# r =  1 kuvvetli ve pozitif ilişki
# r = -1 kuvvetli ve negatif
# r = 0.6 ilişki var ama çok değil
# r = -0.4 ilişki var ama düşük
# r = 0 ilişki yok

# correlation ilişki ve variables move together and data represent in single point,
# regression bir değişkenin diğer değişkeni ne kadar etkilediği, neden ve etki var regresyon da, data represent by line
#Simple Linear Regression - Supervised Model
# Parametrik algoritmadır. (Matematiksel bir denklem ile ifade edilen)
# Continuous variable lar tahmin edilmeye çalışılır
# Simple ve Multi olarak ikiye ayrılır
# b0: intercept        : y eksenini kestiği nokta
# b1: slope            : Eğim
# ei = Yi - y(head)i   : Hata
# Y(head) = b0 + b1*X  : Regression Equation
# Salary = b0 + b1*Experience

# method of Least Squares : TOPLAM(ei(kare)) = TOPLAM(Yi-Y(head)i)(kare)
# NOT: Best fit line ı hata kareler toplamını minimize ederek çizeriz. Yani
# .. least square metodu kullanılır algoritma olarak gradient descent i kullanırız
# NOT: Bütün hataların toplamı best fit line için sıfır olur. Bu yüzden kareler alınır
# NOT: gradient descent kullanırken karelerin türevini almak daha kolay olduğundan karalaeri almak daha kullanışlıdır

# Gradient Descent: Rasgele line lar çizer hataları hesaplar. Başka line çizer tekrar hesaplar
# .. böyle devam ediyor. En son hata bir yerde artıyor. O anda duruyor ve bir önceki benim
# .. best fit line ım diyor

# R(kare):Bağımsız değişkenlerin bağımlı değişkeni açıklama oranı.
# .. Orion hoca : Elimizdeki featurelar ile targetın varyansını ne kadar açıkladığını gösterir
# .. 0 ile 1 arasında değişir.
# .. Tv Ads --> Cars örneği
"""
week   Number of Tv ads    Number of cars sold
1       3                   13
2       6                   31
3       4                   19
4       5                   27
5       6                   23
6       3                   19
R(kare) = 0.7
# Tv reklamları araba satışlarını %70 oranında açıklıyor(Açıklanabilir varyans %70)
# Kalan %30
"""

# Simple Linear Regression   : y = b0+b1*x1
# Multiple Linear Regression : y = b0+ b1*x1 + b2*x2 + ... + bn*xn


#Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings('ignore')



#Read Dataset

df = pd.read_csv("Advertising.csv")
df

#independent variable = feature = estimator = attribute = input
#dependent variable = target = label = output
#rows = observation, sample
#features = TV + radio + newspaper
#target = sales

df.info()

 #   Column     Non-Null Count  Dtype  
#---  ------     --------------  -----  
 #0   TV         200 non-null    float64
 #1   radio      200 non-null    float64
 #2   newspaper  200 non-null    float64
 #3   sales      200 non-null    float64
#dtypes: float64(4)
#memory usage: 6.4 KB

df.describe().T
#           count      mean        std  min     25%     50%      75%    max
#TV         200.0  147.0425  85.854236  0.7  74.375  149.75  218.825  296.4
#radio      200.0   23.2640  14.846809  0.0   9.975   22.90   36.525   49.6
#newspaper  200.0   30.5540  21.778621  0.3  12.750   25.75   45.100  114.0
#sales      200.0   14.0225   5.217457  1.6  10.375   12.90   17.400   27.0

#mean ve std degerleri birbirine yakinsa yada std mean den buyukse outlier olabilir 
#min ile 0.25 lik deger ve 0.75 ve max arasinda ciddi bir fark varsaoutlier olabilir


#Create new independent variable (feature)

# Simple lineer regression ı göstermek için feature engineering yapalım

#modelden en iyi sonuc alinabilmesi icin eldeki feature larla yeni feature olusturldu.(feature engineering)
#tatal_spend in corr u incelenecek,corr en iyi olan fuature alinacak(simple lineer regression icin(tek feature))
df["total_spend"] = df["TV"] + df["radio"] + df["newspaper"]

df

#        TV  radio  newspaper  sales  total_spend
#0    230.1   37.8       69.2   22.1        337.1
#1     44.5   39.3       45.1   10.4        128.9
#2     17.2   45.9       69.3    9.3        132.4
#3    151.5   41.3       58.5   18.5        251.3
#4    180.8   10.8       58.4   12.9        250.0
#..     ...    ...        ...    ...          ...
#195   38.2    3.7       13.8    7.6         55.7
#196   94.2    4.9        8.1    9.7        107.2
#197  177.0    9.3        6.4   12.8        192.7
#198  283.6   42.0       66.2   25.5        391.8
#199  232.1    8.6        8.7   13.4        249.4


#okunmasi kolay olmasi icin tarket sona alindi.bu bir kural degil 
df = df.iloc[:,[0,1,2,4,3]]
df.head()
#      TV  radio  newspaper  total_spend  sales
#0  230.1   37.8       69.2        337.1   22.1
#1   44.5   39.3       45.1        128.9   10.4
#2   17.2   45.9       69.3        132.4    9.3
#3  151.5   41.3       58.5        251.3   18.5
#4  180.8   10.8       58.4        250.0   12.9

sns.pairplot(df)
#en yuksek corr un total_spend ile target(sales) arasinda oldugunu gorduk

#Which feature is more suitable for linear regression?

# We will check correlation for answer.

for i in df.drop(columns ="sales"):
    print(f"corr between sales and {i:<12}:  {df.sales.corr(df[i])}")

#corr between sales and TV          :  0.7822244248616061
#corr between sales and radio       :  0.5762225745710551
#corr between sales and newspaper   :  0.22829902637616528
#corr between sales and total_spend :  0.8677123027017427

sns.heatmap(df.corr(), annot=True)

#corr en yuksek total_spend feature oldugu icin onla devem edilecek
#independent variable:tota_spend ,dependent variable=sales
#simple lineer regression oldugu icin tek feature la devam edildi 
df = df[["total_spend", "sales"]]
df.head()


#Plotting the relationship between independent variable and dependent variable

sns.scatterplot(x ="total_spend", y = "sales", data=df)
#pozitif yonlu kuvvetli iliski
#Correlation between independent variable and dependent variable

corr = df["sales"].corr(df["total_spend"])
corr
#0.8677123027017427
df["total_spend"].corr(df["sales"])
#0.8677123027017427


#Coefficient of determination (R^2)

R2_score = corr**2  # Sadece simple linear regression özelinde olan bir formül. Multi de böyle değil
R2_score  # Feature ekleyerek R^2 yükseltilebilir(Eğer data lineer bir özellik gösteriyorsa)
#0.7529246402599608   elimizdeki ver target i tahmin edebilme icin yuzde 75 oraninda yeterli
#Bağımsız değişkenlerin bağımlı değişkeni açıklama oranı
#target label i ni dogru tahmin edebilmek icin elimde yeterli verinin ne kadari var
#Linear Regression

sns.regplot(x="total_spend", y="sales", data=df, ci=None) #eldeki data icin best fit i cizer
#best fit in altinda ve ustunde olan residual degerleri topladigimizda her zaman sifi cikar 

#Splitting the dataset into X(independent variables) and y (dependent variable)

# y_pred = b1X + b0   b1:slope   b0:intercept(x  0 iken y nin aldigi deger)

X= df["total_spend"]
y= df["sales"]


#Determination of coefficients (slope and intercept)
np.polyfit(X, y, deg=1) #bagimli ve bagimsiz degiskenler ve derecesi verildiginde katsayilari dondurur 
#array([0.04868788, 4.24302822])
slope, intercept = np.polyfit(X, y, deg=1)
print("slope    :", slope)
print("intercept:", intercept)
#slope    : 0.048687879319048145
#intercept: 4.2430282160363255

#Why do we use the least squares error method to find the regression line that best fits the data?

b1, b0 = np.polyfit(X, y, deg=1) # degree=1, linear (doğrusal) model

print("b1 :", b1)
print("b0 :", b0)
#b1 : 0.048687879319048145
#b0 : 4.2430282160363255
y_pred = b1*X + b0

values = {"actual": y, "predicted": y_pred, "residual":y-y_pred, "LSE": (y-y_pred)**2}
df_2 = pd.DataFrame(values)
df_2
#     actual  predicted  residual        LSE
#0      22.1  20.655712  1.444288   2.085967
#1      10.4  10.518896 -0.118896   0.014136
#2       9.3  10.689303 -1.389303   1.930164
#3      18.5  16.478292  2.021708   4.087302
#4      12.9  16.414998 -3.514998  12.355211
df_2.residual.sum().round()   #hatalarin(residual) toplami sifir geldi 

#Prediction with simple linear regression

potential_spend = np.linspace(0, 500, 100)
potential_spend

predicted_sales_lin = b1* potential_spend + b0
predicted_sales_lin

sns.regplot(x="total_spend", y="sales", data=df, ci=None)  
plt.plot(potential_spend, predicted_sales_lin)


# NOT: Modeli eğittiğiniz range önemli 1-5 odalı evler için fiyat tahmini yapıyorsanız,
# .. 8 odalıyı tahmin ederken büyük ihtimalle yanlış sonuç alırsınız

#%%%
################LESSON 2##########################

# Not: Lineer Regression 1. Assumption: Güçlü bir lineer ilişkisi olması 

# There are four assumptions associated with a linear regression model:
# 1.Linearity: The relationship between X and the mean of Y is linear.
# 2.Homoscedasticity: The variance of residual is the same for any value of X.
# 3.Independence: Observations are independent of each other.
# 4.Normality: For any fixed value of X, Y is normally distributed.


# Residual: Hata = ei = yi-y(head)i
# Not: Hataların toplamı sıfırdır
# 2. Assumption : Hatalar normal dağılım sergilerse bu data lineer regression a uygundur
# Hatalar origin etrafında homojen dağılmalı

# Regression Error Metrics
    # 1.Mean absolute error    : MAE : Ares hoca: Hataları çok cezalandırmıyor
    # 2.Mean Square Error      : MSE : Karesini aldığımızdan dolayı hataları fazla cezalandırıyor.
        # .. o yüzden bunu yorumlamakta da zorlanıyoruz.
    # 3.Root Mean Square Error : RMSE : Büyük hataları büyük, küçük hataları küçük cezalandırıyor.
        # .. o yüzden bunu yorumlamak daha kolay diyebiliriz.
# Genelde RMSE yi kullanacağız. Orion Hoca: Hepsini kullanıp duruma göre değerlendirme yapacağız ancak genelde
# MAE küçük, RMSE ve MSE büyük gelir. Önemli olan bunları yorumlamak
# Orion Hoca: Modeliniz de outlierlar var ise linear regresyon için MAE kötü bir seçim olabilir

# Scikit-Learn Library
# Model fitting, data processing, model selection, evalution vs gibi şeyler yapılabilir

# Machine Learning with Python
    # 1.EDA
    # 2.Splitting Data
        # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=42)
    # 3.Model Building
        # model.fit(X_train, y_train)
            # logmodel= LogisticRegression()
            # logmodel.fit(X_train, y_train)
    # 4.Predicting Data
        # prediction = model.predict(X_test)
            # prediction = logmodel.predict(X_test)
    # 5.Evaluation of the Model
        # R2_score = r2_score(actual,pred)
        
# Orion Hoca : Model oluşturma süreci basitçe :
# 1.EDA
# 2.Train Test Split
# 3.Preprocess (Scale,onehot encodin vb.)
# 4.model building (Linear regresion .......vb)
# 5.model evaluation (erros metricleri üzerinden modelin değerlendirilmesi)


#%%%
##############NOTEBOOK 2###################

#Multiple Linear Regression and Regression Error Metrics


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("Advertising.csv")
df
#        TV  radio  newspaper  sales
#0    230.1   37.8       69.2   22.1
#1     44.5   39.3       45.1   10.4
#2     17.2   45.9       69.3    9.3
#3    151.5   41.3       58.5   18.5
#4    180.8   10.8       58.4   12.9


df.shape
#(200, 4)

df.info()
 #   Column     Non-Null Count  Dtype  
#---  ------     --------------  -----  
# 0   TV         200 non-null    float64
# 1   radio      200 non-null    float64
# 2   newspaper  200 non-null    float64
# 3   sales      200 non-null    float64
#dtypes: float64(4)

sns.pairplot(df)

df.corr()

#                 TV     radio  newspaper     sales
#TV         1.000000  0.054809   0.056648  0.782224
#radio      0.054809  1.000000   0.354104  0.576223
#newspaper  0.056648  0.354104   1.000000  0.228299
#sales      0.782224  0.576223   0.228299  1.000000

sns.heatmap(df.corr(),annot=True)


# Korelasyonlar düşük diye EDA aşamasında bu değişkenler atılmamalı. Çünkü;
# .. bu değişkenler kullanılıp feature engineering yapılabilir ve o yeni oluşan
# .. feature lar işimize yarayabilir ya da feature lar bazen öğrenmede çok işe
# .. yarayabilir(korelasyonu düşük olmasına rağmen)
# Orion Hoca: Önce tüm değişkenleri atmadan modele sokup daha sonra feature çıkartmak daha iyi


#Train-Test Split

#verilerin hepsinin sayısal olması gerekiyor 


X=df.drop(columns="sales")
y=df["sales"]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
df.sample(15)

#test_size:datanın ne kadarının test e ayrılacağına karar verir.
#amaç train setine olabildiğince max gözlem verebilmek

print("Train features shape : ", X_train.shape)
print("Train target shape   : ", y_train.shape)
print("Test features shape  : ", X_test.shape)
print("Test target shape    : ", y_test.shape)
# 200 gözlemim vardı 0.80 ini(160 tanesini) train e ayırdı. 0.20 sini test e ayırdı train_test_split()

#Train features shape :  (160, 3)
#Train target shape   :  (160,)
#Test features shape  :  (40, 3)
#Test target shape    :  (40,)


X_train
X_test
y_train

#x_train y_train deki değerleri baz alarak kendini eğitir,kuralları belirler
#yani katsayıları belirler  
#eğitim bittikten sonra modele x_tes verilir,y_test verilmez,aldığı eğitime göre
#x_test deki değerlere tahminde bulunur(y_pred )
#y_pred=X_test sonucunda alınan tahminlerdir
#sonrasında y_pred ile y_test karşılaştırılır
#karşılaştırma sonucunda da r2,mae,rmse skorları alınacak 


#Model Fitting and Compare Actual and Predicted Labels

from sklearn.linear_model import LinearRegression
model=LinearRegression() #---->önce model oluşturulur 
model.fit(X_train,y_train)#---->sonra bu modelin içine X_train,y_train atılır,eğitime verilir
#amaç y trainde belirtilen gerçek değerlere en yakın tahminleri elde edebilmek
y_pred=model.predict(X_test) #eğitim sonucu y_pred değerleri bulunur.tahmin yapıyor
y_pred #modelin döndürdüğü tahminler


model.coef_ #modelin bütün featurelar için belirlediği katsayıları verir
 # b1,b2,b3 : katsayılar(Öğrenilen kısım diyebiliriz)
#array([0.04472952, 0.18919505, 0.00276111])
model.intercept_ # b0: y eksenini kestiği nokta

# y_pred = b3 * TV + b2 * radio + b3 * newspaper + b0


X_test.loc[95] # Rasgele bir değer aldık tahminde bulunmak için

#TV           163.3
#radio         31.6
#newspaper     52.9
#Name: 95, dtype: float64

sum(X_test.loc[95] * model.coef_) + model.intercept_ 
#y_pred in arka planda ne yaptığını göstermek için manuel hesaplandı

#16.408024203228628  X_test.loc[95] için tahmin sonucu 


my_dict = {"Actual": y_test, "Pred": y_pred, "Residual":y_test-y_pred}

comparing = pd.DataFrame(my_dict)
comparing.head(5)
#     Actual       Pred  Residual
#95     16.9  16.408024  0.491976
#15     22.4  20.889882  1.510118
#30     21.4  21.553843 -0.153843
#158     7.3  10.608503 -3.308503
#128    24.7  22.112373  2.587627

result_sample = comparing.head(25) # Grafik çizdireceğimiz için 25 tane gözlem alalım
result_sample

result_sample.plot(kind ="bar", figsize=(15,9))
plt.show()


#ERROR METRICS

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

R2_score = r2_score(y_test, y_pred) # y_pred ile y_test i karşılaştırıyoruz
R2_score # Sales i Tv,Radio ve newspaper yüzde 89 oranında açıklıyor.(Ares Hoca: Fena değil)
# Açıklanamayan kısım için feature engineering yapabiliriz.
#0.8994380241009121

mae=mean_absolute_error(y_test,y_pred)
mae# Hataları fazla cezalandırmıyor.

#1.4607567168117597

mse = mean_squared_error(y_test, y_pred)
mse   # Hataları fazla cezalandırıyor
#3.1740973539761015

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse # Yorumlaması daha kolay. Tercih ettiğimiz metric
#1.7815996615334502

sales_mean = df["sales"].mean()
sales_mean
#14.022500000000003

mae / sales_mean
#0.10417234564533852 #modelim ortalaa yüzde 10 civarında yanlış tahminler yapıyor
#yapılan kötü tahminleri gözz ardı eder

rmse / sales_mean
#0.12705292647769298

#Adjusted R2 score

def adj_r2(y_test, y_pred, X):
    r2 = r2_score(y_test, y_pred)
    n = X.shape[0]   # number of observations  
    p = X.shape[1]-1 # number of independent variables  
    adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)
    return adj_r2
adj_r2(y_test, y_pred, X)
#0.8984170903354392
#Eğer dataya yeni feature lar ekliyorsak,datanın yeni bilgiler öğrenebilmesi için
#yeni gözlemler ilave etmemeiz gerekir ki data yeni şeyler öğrenebilsin.
#Feature sayısı arttığında gözlem sayısını arttırmazsak R2 score da yalancı bir iyileşme olur ama MAE ve 
#RMSE de herhangi bir değişiklik olmaz veya daha çok kötüleşebilir.Bunun önüne geçmek için
#"Asjusted r2 score" fonksiyonu kullanıyoruz.Bu score gözle sayısı ile feature sayısını
#dengeler
# Biz feature engineering de feature sayısını arttırdıkça r2 de yalancı bir iyileşme oluyor
# Bunu engellemek adın r2_adj score u kullanıyoruz yorumlarken/değerlendirme yaparken
# Orion Hoca: Feature eklerseniz r square her zaman iyileşir(O yüzden gözlemde eklemeliyiz ki r2 bizi yanıltmasın)
# Feature sayınız ile gözlem oranınız aynı oranda artmalı ki r2 score umuz dengeli olsun
# .. r2_adj bunu engelliyor
# Interview larda karşılaşabilirsiniz diye bunu anlatıyoruz
# Interview soru: Lineer regresyonda feature ekleyerek doğruluğum arttı. Bunu nasıl teyit ederiz
# Cevap: r2_adj score a bakarız deriz


#What is the concept of punishment for RMSE error metric?

variables = {"Actual": [1000, 2000, 3000, 4000, 5000], "pred": [1100, 2200, 3200, 4200, 5300]}  # 6000
df_2 = pd.DataFrame(variables)
df_2

#   Actual  pred
#0    1000  1100
#1    2000  2200
#2    3000  3200
#3    4000  4200
#4    5000  5300

df_2["residual"] = abs(df_2.Actual - df_2.pred)
df_2
#   Actual  pred  residual
#0    1000  1100       100
#1    2000  2200       200
#2    3000  3200       200
#3    4000  4200       200
#4    5000  5300       300


#mae
df_2.residual.sum()/5 # 5: gözlem sayısı
#200 # 5300 yerine 6000 yazarsak sonuç 340.00 çıkacak


#rmse
((df_2.residual**2).sum()/5)**0.5  # 0.5 : Karekök almak yerine 0.5 ile çarpılmış
# 5300 yerine 6000 yazarsak bu 475.39 çıkacak

#209.76176963403032


def eval_metric(actual, pred):
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    R2_score = r2_score(actual, pred)
    print("Model testing performance:")
    print("--------------------------")
    print(f"R2_score \t: {R2_score}")
    print(f"MAE \t\t: {mae}")
    print(f"MSE \t\t: {mse}")
    print(f"RMSE \t\t: {rmse}")
# Her seferinde yazmak yerine burada bunları bir fonksiyona topladık

eval_metric(y_test, y_pred) # Modelin görmediği data

#Model testing performance:
#--------------------------
#R2_score 	: 0.8994380241009121
#MAE 		: 1.4607567168117597
#MSE 		: 3.1740973539761015
#RMSE 		: 1.7815996615334495

#y_pred=model.predict(X_test)
#sadece test setinde aldığımız skorlarla yetinmiyoruz
#test setinde aldığımız skorların gerçekten genelleme yapılabilecek
#skorlar olduğunu anlamak için mutlaka eğitim yaptımız train setindende tahmin alırız.
y_train_pred = model.predict(X_train)
#train stinde skorlar genelde daha yüksek olur çünkü eğütimi yaptığı yer train 

eval_metric(y_train, y_train_pred) # Modelin gördüğü data
# test ile train score larımın(r2 özellikle) yakın olması gerekiyor. Yoksa
# .. overfitting ve underfitting durumları oluşuyor(Daha sonra anlatılacak)

#Model testing performance:
#--------------------------
#R2_score 	: 0.8957008271017817
#MAE 		: 1.1984678961500141
#MSE 		: 2.7051294230814147
#RMSE 		: 1.6447277656443375


#Is data suitable for linear regression?

residuals = y_test-y_pred "1"
plt.figure(figsize = (10,6))
sns.scatterplot(x = y_test, y = residuals) #-residuals
plt.axhline(y = 0, color ="r", linestyle = "--")
plt.ylabel("residuals")
plt.show()

sns.kdeplot(residuals); # Residular normal dağılım sergiliyor mu diye bakıyoruz
# Ares hoca: Normal dağılıma yakın bir dağılıma benziyor

stats.probplot(residuals, dist ="norm", plot =plt); # Q-Q plot
# residularım bu doğruya yakın olup sarılırsa datam normal dağılım sergiliyor diyebiliriz

#https://stats.stackexchange.com/questions/12262/what-if-residuals-are-normally-distributed-but-y-is-not

from scipy.stats import skew
skew(residuals) # 3. method : -0.5 ile +0.5 aralığında residuların normal dağılım sergilediğini gösteriyor
# -1 ile 1 olursa normal dağılımdan uzaklaştığını söyleyebiliriz
#pip install -U yellowbrick


from yellowbrick.regressor import ResidualsPlot # 4. method

# Instantiate the linear model and visualizer
model = LinearRegression()
visualizer = ResidualsPlot(model)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show();        # Finalize and render the figure



#Prediction Error for LinearRegression

from yellowbrick.regressor import PredictionError
# Instantiate the linear model and visualizer
model = LinearRegression()
visualizer = PredictionError(model)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show();    

#Retraining Model on Full Data

final_model = LinearRegression()
final_model.fit(X, y)

#Coefficients

final_model.coef_
#array([ 0.04576465,  0.18853002, -0.00103749])

final_model.intercept_
#2.9388893694594103

df.head()

#    TV	    radio	newspaper	sales
#0	230.1	37.8	69.2	    22.1
#1	44.5	39.3	45.1	    10.4
#2	17.2	45.9	69.3	    9.3
#3	151.5	41.3	58.5	    18.5
#4	180.8	10.8	58.4	    12.9

coeff_df = pd.DataFrame(final_model.coef_, index = X.columns, columns = ["Coefficient"] )

coeff_df
#           Coefficient
#TV            0.045765
#radio         0.188530
#newspaper    -0.001037

#Prediction on New Data

adv = [[150, 20, 15]] # 2D # Modeller 2 boyutlu olmalı o yüzden çift parantez kullanıldı
adv
#[[150, 20, 15]]

final_model.predict(adv)
#array([13.55862413])
final_model.coef_ # Bu coefficient lar yanıltıcı olabilir korelasyonla karşılaştırıldığında
# .. Burada sanki radio sanki 0.18 ile Tv(0.045) den daha önemli gibi görünüyor. Ama bu "scaling" yapmadığımız için oldu
#array([ 0.04576465,  0.18853002, -0.00103749])
final_model.intercept_
#2.9388893694594103
sum(final_model.coef_ * [150, 20, 15]) + final_model.intercept_
#13.558624130495994

adv_2 = [[150, 20, 15], [160, 20, 15]] # tv yi 10 birimlik arttırdık

final_model.predict(adv_2)    # tahmin 13.55 den 14.01 e gelmiş.
#array([13.55862413, 14.01627059])

14.01627059 - 13.55862413
# tv deki 10 birimlik artış. Benim label ımda 0.45 lik bir artışa neden oldu.
# Yani tv nin katsayısı oldu

# Biz varsayımları vs irdeledik burada ama biz her zaman bunlara bakmayıp
# .. modeli alıp fit edip vs ilerleyeceğiz


#%%

#####################LESSON 3####################### 

#bıas:Modelin dataları çok fazla genellemesidir.

#UNDERFIT             IDEAL                   OVERFIT
#Az paramete          low                  çok parametre 
#High Bias         low variance              Low bias
#Low varyans                                high varyans
#data->simple                               data-->complex


#UNDERFIT:modelin data eğitimine ihtiyacı var
#    *Datayı çok fazla geneller
#    *Hem eğitim hemde test datasında büyük hatalar yapar 
#    *Bu yüzden varyansı düşük çıkar
#    *Model hiçbir şey öğrenmemiş olur

#OVERFIT:
#    *çok parametre olunca bütün noktalarına girer
#    *datayı ezberler
#    *train datada varyans --->düşük 
#    *test datasında varyans ---<yüksek    çünkü ezberledi


#HOW TO RECOGNIZE UNDERFITTING-OVERFITTING ?

#underfitting:High training error,high test error
#overfitting:low training error,high test error(en çok bununla karşılaşacağız) 

#*hem train hem test datasında düşük hatalar alınması beklenir 

#Variance:Train ile test datası arasındaki fark 

#Comlexity çok düşük olursa model underfit(çok karşılaşmayız),çok yüksek olursa 
#overfit olur.Biz ne çok ne de çok düşük olsun istiyoruz 

#Bias: train setindeki gerçek değerle tahmin edilen değer arasındaki fark

#Underfitting ve Overfitting ile mücadele için;
#underfittig-->feature eklenir(Comlexity artar)
#Overfitting-->Feature azaltılır(Comlexity düşer)
#datanın azlığından kaynaklanıyorsa data arttırılır yada derece düşürülür

#*overfitting de daha iyi bir sonuç alır mıyım diye cross validation yapılabilir
#*yada regularization(lassa-ridge)yapılabilir
 
#                    Underfitting          Overfitting
#Bias                   high                   low
#Variance               low                    high
#Complexity             low                    high
#Flexibillity           low                    high
#Generalizability       high                   low

#Types of regression models 
#*Simple linear regression   y=b0+b1x
#*Multi linear regression 
#*Polynomial Regression      y=b0+b1x+b2x^2...

#Smple Linear-->1 feature, 1 label 
#Multi linear-->birden fazla feature,1 label
#Polynomial-->Multinin özel biçimi
#*Non-linear datalarda iyi çalışır 
#*Polynomial degresini seçmek çok önemli 
#*degree=1-->Linear, degree=düşük-->underfit ,degree=yüksek-->overfit 

#*Polynomial regressionda Bias Variance Trade-off(Bias-Variance degresi)ni 
#ayarlaMak çok önemli 

#Regularization --->overfitting ve Multicallinearity ile ücadele 
#regularization-->*performansı yüksektir 
#*trainde testde makul sonuçlar verir 

#*Multicallinearity, Linear modellerde yaşanan bir sorundur. 
#*Advance modellerde bunların arkada çalışan paraetreleri var ama linear 
#modellerde yok 

#%%
################NOTEBOOK 3#####################################

# Lineer Regression ın devamı. Polynomlarla bu regressipn ı çözmeye çalışacağız
# Underfitting ve Overfitting göreceğiz bu gün

# ÖNCEKİ DERSIN ÖZETİ
# Residuals
    # Sum and mean of the Residuals are always Zero
    # Residuals are normally distributed for suitable Linear Regression
# Regression Error Metrics
    # MAE, MSE, RMSE
# Scikit-learn Library and ML
    # 5 steps: import, split data, model building and fit, prediction, evaluation

# Konular
    # Introduction to Bias-Variance Trade-off
    # Underfitting and Overfitting Problems
    # Training Errors vs Validation Error(Test Error)
    # Polynomial Regression

# Introduction to Bias-Variance Trade-off
# varyans hataların dağlımı demek 
# Bias, bir eşik/aralık şeklinde tanımlayabiliriz. Varyans demek değişim demek
# .. Heterojen yapıda değişim çok fazla olur. Homojen yapıda değişim çok az olur
# .. Bias, Data Scienceda da bu residual e karşılık geliyor.
# 3 kavrama bakacağız
    # Underfitted      : Models with too few parameters may not fit the data well(high bias)
        # .. but are consisten accross different training sets(low variance)
        # Residual erin ya da bias ın burada çok yüksek olduğunu söyleriz, varyans düşük(dümdüz doğru)
        # -- Simple Model Underfit
        # Bu durumda model veriyi iyi öğrenememiş. Simple bir model ortaya çıkarmış complex bir data için
        # Buna underfitting denir
        # Yeterli feature olmadığında ya da model complexity düşük olduğunda kaynaklanır
        # Veriyi arttırmakla, Ya da complexity yi arttırmakla çözülebilir
    # Good Fit/Robust  : bias ve variance ideal değerlerde(Low bias & low variance)
    # Overfitted       : Models with too many parameters may fit the training data well(low bias)
        # but are sensitive to choice of training set(high variance) -- Complex Model-- OverFIT
        # Burada varyans çok yüksek, bias çok düşük
        # Bu durumda model veriyi ezberlemiş. Hata 0 a yakın. Training sonucu %100 çıkar ama
        # .. test(görmediği) datası geldiğinde başarısının düşük olduğunu görürüz
        # NOT: Bundan dolayı train hatası ile test hatası birbirine yakın olmalı
        # Model complex olmasından kaynaklanır. Çözmek için model complexity düşürülebilir
        # Parametre azaltılması, More training data/ Cross/validation ile, Regularization(Lasso&Ridge) gibi şeylerle çözülür
    # Orion hoca: 
        # low bias low variance(underfittig)   modelin trainden aldığı scorelar ile testten aldığı scoreların ikisininde  çok düşük olması ile karşımıza çıkar.
        # low bias high varyans(overfitting) ise model train datası üzerinde çok çok iyi score alırken test datasına gelince kötü scorelar olarak karşımıza çıkar.

# Model complexity
    # Train ve test hataları ile alakalı bir konu Train hatası ile test hatasının belli bir oranda
    # .. olması gerekir. Grafiklere bakmak daha anlamlı burada. Ayrıca altta da bahsedilecek
    # Orion Hoca : bizim istediğimiz model train datası üzerinden bir genelleme yapsın bazı hataları olsun bunun karşılığında da test datası üzerinde hata makul seviyede kalsın.


# Kurs boyunca 2 anayol var: Regression ve Classification
# Regression bir değer buluyoruz(Salary, Amount vs)
# Classification da sınıflandırma yapıyoruz(0 mı 1 mi, sıcak mı soğuk mu, Hayatta mı değil mi ... gibi)
# Alttaki classification da grafikte sağ üst underfitting. Sağ alt overfitting


#Polynomial Regression¶
#Polynomial Regression is a form of regression analysis in which the relationship between the independent variables and dependent variables are modeled in the nth degree polynomial.

#Polinom Regresyon , bağımsız değişkenler ile bağımlı değişkenler arasındaki ilişkinin n'inci derece polinomda modellendiği bir regresyon analizi şeklidir.

#Types of polinomials

#1st degree ---> linear b1x + b0

#2nd degree ---> Quadratic b2x**2 + b1x + b0

#3rd degree ---> Cubic b3x3 + b2x2 + b1x + b0 (third order equation)

#interview sorusu:Bir datanın lineer regresyona uygun olup olmadığını nasıl anlıyorsunuz?
#*target datasındaki gerçek değer ve tahmin değerlerinin residual larının 
#normal dağılıma uyup uymadığını kontrol ediyoruz.
#*bunlar mutlaka random dağılacak 

#Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (10,6)

#Polynomial Features
from sklearn.preprocessing import PolynomialFeatures

# içerisine datayı verdiğimiz zaman verdiğimiz dereceye göre feature sayısını arttırıyor
data = np.array([[2, 3, 4]])   # Çalışma mantığını anlatmak için array ürettik
print(data)

trans=PolynomialFeatures(degree=2,include_bias=False)# Datayı polinomial Featurelara dönüştürüyoruz
# include_bias = False : İlerde neden False olduğundan bahsedilecek(basitçe formülde b0 ın olmaması) 

trans.fit(data) # 2, 3, 4 , 2x3, 2x4, 3x4, 2**2, 3**2, 4**2  # Veriyi öğrenme/tanıma aşaması

# NOT: Orion hoca: fit kalıbını çıkar, transform o kalıbı uygula demek
# 2, 3, 4, 2x3, 2x4, 3x4, 2**2, 3**2, 4**2, 2x3x4, 3x2**2, 4x2**2, 2x3**2, 4x3**2, 2x4**2, 3x4**2, 2**3, 3**3, 4**3

trans.transform(data)  # Veriyi fit ettikten sonra dönüştürüyor. Ancak alttaki komut tek aşamada yapıyor
#array([[ 2.,  3.,  4.,  4.,  6.,  8.,  9., 12., 16.]])

trans.fit_transform(data) # combining method # Daha kullanışlı
# Orion hoca: Eldeki feature lar ile sentetik feature lar ürettik diyebiliriz
#array([[ 2.,  3.,  4.,  4.,  6.,  8.,  9., 12., 16.]])

df = pd.read_csvpd.read_csv("Advertising.csv")
df.head()

#      TV  radio  newspaper  sales
#0  230.1   37.8       69.2   22.1
#1   44.5   39.3       45.1   10.4
#2   17.2   45.9       69.3    9.3
#3  151.5   41.3       58.5   18.5
#4  180.8   10.8       58.4   12.9


#Polynomial Converter

X=df.drop(columns="sales",axis=1)# sales haricindeki diğer değişkenleri seçiyoruz(Independent variables)
y=df.sales

polynomial_converter=PolynomialFeatures(degree=2,include_bias=False)
polynomial_converter.fit(X)

poly_features=polynomial_converter.transform(X)
poly_features


#array([[ 230.1 ,   37.8 ,   69.2 , ..., 1428.84, 2615.76, 4788.64],
#       [  44.5 ,   39.3 ,   45.1 , ..., 1544.49, 1772.43, 2034.01],
#       [  17.2 ,   45.9 ,   69.3 , ..., 2106.81, 3180.87, 4802.49],
#       ...,
#       [ 177.  ,    9.3 ,    6.4 , ...,   86.49,   59.52,   40.96],
#       [ 283.6 ,   42.  ,   66.2 , ..., 1764.  , 2780.4 , 4382.44],
#       [ 232.1 ,    8.6 ,    8.7 , ...,   73.96,   74.82,   75.69]])

#polynomial_converter.fit_transform(X)

poly_features.shape # Shape de columns 3 dü, 9 a çıktı(2.dereceden olduğu için)
#(200, 9)


Train | Test Split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(poly_features,y,test_size=0.3,random_state=101)
# test_size    : train datasının ve test datasının oranını belirleme. Test oranı:%30, Train oranı:%70
# random_state : Her seferinde aynı sonuçları almak için kullanılır(Çalışırken hocayla aynı sonuçları almak için)

X_train.shape
#(140, 9)
from sklearn.linear_model import LinearRegression
model_poly=LinearRegression()
model_poly.fit(X_train,y_train)

my_dict = {"Actual": y_test, "pred": y_pred, "residual": y_test-y_pred}
compare = pd.DataFrame(my_dict)
compare.head(5)
#     Actual       pred  residual
#37     14.7  13.948562  0.751438
#109    19.8  19.334803  0.465197
#31     11.9  12.319282 -0.419282
#89     16.7  16.762863 -0.062863
#66      9.5   7.902109  1.597891

#tahminler iyi gözüküyor 

compare.head(20).plot(kind='bar',figsize=(15,9))
plt.show();


#Poly Coefficients


model_poly.coef_
#array([ 5.17095811e-02,  1.30848864e-02,  1.20000085e-02, -1.10892474e-04,
#        1.14212673e-03, -5.24100082e-05,  3.34919737e-05,  1.46380310e-04,
#       -3.04715806e-05])


df_coef = pd.DataFrame(model_poly.coef_, index = ["TV", "radio", "newspaper", "TV^2", "TV&Radio", \
                                   "TV&Newspaper", "Radio^2", "Radio&newspaper", "Newspaper^2"], columns = ["coef"])

df_coef 
# John Hoca: Standartlaştırma yapmadığımız için alttaki feature ların önemlerini sıralamak çok doğru değil.
# .. Bunu Ridge ve Lasso da detaylı göreceğiz

#                     coef
#TV               0.051710
#radio            0.013085
#newspaper        0.012000
#TV^2            -0.000111
#TV&Radio         0.001142
#TV&Newspaper    -0.000052
#Radio^2          0.000033
#Radio&newspaper  0.000146
#Newspaper^2     -0.000030


model_poly.predict([[2.301000e+02, 3.780000e+01, 6.920000e+01, 5.294601e+04,
       8.697780e+03, 1.592292e+04, 1.428840e+03, 2.615760e+03,
       4.788640e+03]])

array([21.86190699])

#Evaluation on the Test Set

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

y_pred = model_poly.predict(X_test) # Tahminlerimizi yaptık.(Dikkat: X_test ile)
# Şimdi metriklerimize bakarak tahminlerin ne kadar iyi olduğuna bakalım


def eval_metric(actual, pred):
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    R2_score = r2_score(actual, pred)
    print("Model performance:")
    print("--------------------------")
    print(f"R2_score \t: {R2_score}")
    print(f"MAE \t\t: {mae}")
    print(f"MSE \t\t: {mse}")
    print(f"RMSE \t\t: {rmse}")


eval_metric(y_test, y_pred)

#Model performance:
#--------------------------
#R2_score 	: 0.9843529333146795
#MAE 		: 0.48967980448035575
#MSE 		: 0.44175055104033906
#RMSE 		: 0.6646431757269001

y_train_pred = model_poly.predict(X_train) # Tahminlerimizi yaptık.(Dikkat: X_train ile)

eval_metric(y_train, y_train_pred) 
# John Hoca: X_test ile X_train metric leri arasında çok fark olmadığı için
# .. underfitting, overfitting durumu yok diyebiliriz

#Model performance:
#--------------------------
#R2_score 	: 0.9868638137712757
#MAE 		: 0.4049248139151643
#MSE 		: 0.34569391424439977
#RMSE 		: 0.5879574085292231

#Simple Linear Regression:
#MAE : 1.213
#RMSE : 1.516
#r2_score : 0.8609

#Polynomial 2-degree:
#MAE : 0.48
#RMSE : 0.66
#r2_score : 0.9868

# Şu anda 2. dereceden bir polynomun(lineer regression a göre) daha iyi sonuçlar ürettiğini söyleyebiliriz
# Acaba derece artarsa daha mı iyi olur sonuçlar. Buna bakacağız altta
# NOT: Polynomial regression un dezavantajının bu olduğu söylenmişti

#Let's find optimal degree of poly

def poly(d):
    
    train_rmse_errors = []
    test_rmse_errors = []
    number_of_features = []
    degrees=[]
    
    for i in range(1, d):
        polynomial_converter = PolynomialFeatures(degree = i, include_bias =False)
        poly_features = polynomial_converter.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)
        
        model = LinearRegression(fit_intercept=True)
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_RMSE = np.sqrt(mean_squared_error(y_train,train_pred))
        test_RMSE = np.sqrt(mean_squared_error(y_test,test_pred))
        
        train_rmse_errors.append(train_RMSE)
        test_rmse_errors.append(test_RMSE)
        
        number_of_features.append(poly_features.shape[1])
        degrees.append(i)
        
    return pd.DataFrame({"train_rmse_errors": train_rmse_errors, "test_rmse_errors":test_rmse_errors, "Degree":degrees}, 
                        index=range(1,d))



poly(10) # Test hatası belli bir seviye sonra veriyi ezberlemiş olduğunu görüyoruz
# .. Çünkü veriyi o kadar ezberlemiş ki train hatasında hata neredeyse hiç yok ama
# .. hiç görmediği veri ile(test verisi) karşılaştırınca ezberlediği için tahminler doğru gelmemiş


#   train_rmse_errors  test_rmse_errors  Degree
#1           1.734594          1.516152       1
#2           0.587957          0.664643       2
#3           0.433934          0.580329       3
#4           0.351708          0.507774       4
#5           0.250934          2.575837       5
#6           0.194567          4.214027       6
#7           5.423737       1374.957405       7
#8           0.141681       4344.727851       8
#9           0.170935      93796.026718       9

#5 ten itibaren overfitting başladı(train seti ile test seti arasında fark açıla durumu )

# Derece 1 den 9 a kadar
plt.plot(range(1,10), poly(10)["train_rmse_errors"], label = "TRAIN")
plt.plot(range(1,10), poly(10)["test_rmse_errors"], label = "TEST")
plt.xlabel("Polynamial Complex")
plt.ylabel("RMSE")
plt.legend();



# Derece 1 den 5 e kadar olan kısmı inceleyelim
# Derece 4-5 arası ezberleme yapmış
# derece 1 de de underfitting var
plt.plot(range(1,6), poly(6)["train_rmse_errors"], label = "TRAIN")
plt.plot(range(1,6), poly(6)["test_rmse_errors"], label = "TEST")
plt.xlabel("Polynamial Complex")
plt.ylabel("RMSE")
plt.legend();

# Hangi noktayı seçeceğiz optimal değeri seçmek için
# Orion Hoca: Sınır değerlerden(2.0 ve 4.0) uzak durulmalı(underfitting ve overfitting e gitmeye meyilli)
# .. O yüzden orta yolu seçmek daha anlamlı(yani 3.0)

#Finalizing Model Choice

final_poly_converter = PolynomialFeatures(degree = 3, include_bias=False)

final_model = LinearRegression()

final_model.fit(final_poly_converter.fit_transform(X), y) 
# Burada datayı bölmedik artık çünkü optimal noktayı belirlediğimiz için tüm datayla eğitimi yapalım ki 
# .. en ideal tahminleri elde edelim


#Predictions
new_record = [[150, 20, 15]]

new_record_poly = final_poly_converter.fit_transform(new_record) 
# Burada new_record_poly formatını değiştiriyorum

new_record_poly

#array([[1.500e+02, 2.000e+01, 1.500e+01, 2.250e+04, 3.000e+03, 2.250e+03,
#        4.000e+02, 3.000e+02, 2.250e+02, 3.375e+06, 4.500e+05, 3.375e+05,
#        6.000e+04, 4.500e+04, 3.375e+04, 8.000e+03, 6.000e+03, 4.500e+03,
#        3.375e+03]])

final_model.predict(new_record_poly) # Sonuç: 14.24
# Kulanıcı derse ki eğer benim tv değerim 150, radio :20, newspaper:20 olursa sonucum(sales) ne olur?? --> 14.24

#array([14.24950844])

#Overfitting

# Dereceyi 5 seçerek overfitting durumunu gözlemleyelim
over_poly_converter = PolynomialFeatures(degree =5, include_bias =False)

over_model=LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(over_poly_converter.fit_transform(X), y, test_size=0.3, random_state=101)
over_model.fit(X_train, y_train)

y_pred_over = over_model.predict(X_test)
eval_metric(y_test, y_pred_over)


#Model performance:
#--------------------------
#R2_score 	: 0.7649916528404768
#MAE 		: 0.6659637641421313
#MSE 		: 6.634794172264552
#RMSE 		: 2.575809420796607

y_train_over = over_model.predict(X_train)
eval_metric(y_train, y_train_over)
# Gördüğümüz gibi metricler arasındaki değerlerde kayda değer(örneğin R_2 0.99-0.76) fark var
# RMSE ye bakarsak train datasında hata çok çok düşük(0.25) ama hiç görmediği test te 2.575 e çıkmış

#Model performance:
#--------------------------
#R2_score 	: 0.9976072484167179
#MAE 		: 0.1862092141111129
#MSE 		: 0.06296802178630591
#RMSE 		: 0.2509342977480478



#Underfitting

#Testing data performance:
#--------------------------
#R2_score 	: 0.8609466508230367
#MAE 		: 1.5116692224549084
#MSE 		: 3.796797236715222
#RMSE 		: 1.9485372043446392


#Training data performance:
#--------------------------
#R2_score 	: 0.9055159502227753
#MAE 		: 1.1581502948072524
#MSE 		: 2.4793551789057866
#RMSE 		: 1.574596830590544

 
# Gördüğümüz gibi metricler arasındaki değerlerde fark var
# r2 de performance düşmüş(0.90 dan, 0.86 ya). Çok düşmemiş ama bu data için ideal
# .. olan değerler daha yüksek olmalıydı
# RMSE de de fark var

# NOT: Ares hoca: R2 score negatif olması aşırı ezberi ve best fit line nın ters çizildiğini gösterir

# NOT: Orion hoca: fit kalıbını çıkar. Transform o kalıbı uygula

#%%


#####################LESSON 4-5########################################## 

#Multicollinearity:Bir modeldeki değişkenlerin en az ikisinin birbiriyle çok 
#içli dışlı olması durumu .
#Bu durum önem sırasının karışmasına neden olur.Modelin kafası karışır.Herhangi 
#birisi atılabilir.Atılmazsa da regularization yapılır.

#Regularization: (regularizasyon)

#Σei^2=Σ(yi−yiˆ) (Least Squared method un cost function u)

#Cost function a penalty(ceza) eklenerek regularization yapılır 

#Σei^2=Σ(yi−yiˆ)^2+Penalty

#Regularization için 3 yöntem var
 
#*Ridge regression(L2)<----Penalty---->Lasso regression (L1)
#                            |
#                      Elastic-Net

#Ridge regression(L2):λ.coef in (slope) karesi kadar hata eklenir 
#Lasso regression (L1):λ.coef in (slope) mutlak değeri kadar hata eklenir 

#λ:Lambda (Hyper-parameter):Cezanın seviyesini belirler 
#*Modelin içinde sabit olmayan,elle değiştirebildiğimiz,modelin kalitesini,metriklerini
#iyileştirdiğimiz parametredir.

#*Lamda arttıkça iyileşme görülür.Ama bir noktadan sonra tekra kötüleşme başlar 


#1-Ridge regression:

#Datamız train datasında çok iyi sonuç verip test datasında çok kötü sonuç 
#vermiş olsun(Overfit-ezber).Biz train datasına ridge regression ile penalty 
#eklersek line ımız train ve test datasında makul bir noktaya gelir.
#(Train datasına hata ekleyince test datasındaki hata düşer,bias ekledik variance düştü)

#Ridge Regression Avantajları:
    
#1-Eğer feature fazla,data az ise;Regularization ile bazı feature ların etkisini
#azaltacağından complexity i aşağı çeker.(yani featureları azaltır).Az feature 
#az row gerektirir.Böylece model daha düzgün çalışır 

#*çünkü modelin düzgün çalışması için;data sağa doğru büyükse(feature) aşağı(observation)
#doğru da büyük olmalı.Bu durumu değiştiremiyorsak Ridge regression ile 
#bazı feature ları öne çıkarırım böylece observation feature dengelenmiş olur 

#2-Multicollinearity için çok iyi.Featurelar arasında önem sırası yapar,en iyi 
#feature ları seçer 

#3-Modele bias(hata) ekleriz,bunun karşılığında variance düşer.Böylece regression
#metriclerinde iyileşme olur 

#Ridge Regression Dezavantajı: 
#Featue selection yapamayız.İyiyi kötüyü gösterir ama kullanmadığımız featureları 
#atamayız.Mesela 100 feature dan ilk 20 si sıfıra yakın değerler çıktı(çok iyi),
#geri kalan 80 i atarsak stability bozulur.Çünkü geri kalan feature larında 
#toplamda mutlaka bir etkisi vardır 

#2-Lasso Regression:
#Çalışma mantığı Ridge ile aynı.Cost function a hata eklenir.Ama karesi değil mutlak 
#değeri eklenir 

#ridge regresion-->Featureları etki sırasına göre sıralar,ama atmaz.
#(x ve y eksenindeki değerler 0 a yaklaşsa bile hiçbir zaman 0 a düşmez )
#-0 a yaklaşır 

#lasso regression-->Önemli feature ları alır geri kalanı direk atar 
#(x eksenindeki değer 0 )-0 yapar 

#Lasso Regression Avantajları:
#1-Feature fazla,data az ise önemsiz feature ları direk atar.Complexity azalır.Ama 
#hata artabilir 
#2-Feature selection yapılabilir 
#3-Bias(hata) eklenir,variance düşer.Regression metric leri iyileşir 

#Lasso Regregression Dezavantajı:
#Yüksek corr ilişkili olan feature lardan birini alıp diğerini attığı için gruplama 
#yapılamaz.
#Çünkü feature ları yok eder 
#Feature azalacağı içi bu durum performansa da etki eder ve R2 skorları biraz düşer 
#Lasso feature sayımı düşürdükten sonra biz daha fazla feature düşürmek istersek :
#    30 feature-->r2=%93 ,RMSI=2 
#    lasson ile 
#    20 feature-->r2=%90 ,RMSI=3
#    lassondan sonra kendimiz düşürürsek performansımız çok düşer 
#    10 feature -->R2=%80 RMSI=7 
    
#SUMMURY:
#    *OLS(Ordinary Least Squares) hatasız bir model ortaya koymaya çalışır 
#    *Ridge&Lasso öne çıkarmak istedikleri featurlara bias eklerler 
#    *Ridge&Lasso overfitting ile mücadele için kullanılır 
#    *Multicollinearity den kurtulmak için kullanılır
#    Ridge-->Group selection için kullanılır 
#    -Featureların hepsi önemlliyse,atamıyorsak bu yöntem 
#    Lasso-->Eliminating predictors için kullanılır(iyileri al,kötüleri at)
#    -Feature sayısı düşer 
    
#3-Elastik Net:
#    Ne Ridge olsun ne de Lasso olsun.ikisinden de olsun 
#    λ yani alpha-->0 ise Ridge gibi davranır 
#                -->1 ise Lasso gibi davranır 
#Ama çoğunlukla Lasso yu seçer 
#Elastic net i pek kullanmayacağız

#Regularization yapabilmek için 2 şey yapılmalı :

#    1-Feature Scaling(Bütün coefficent lları aynı seviyeye getirir)
#    2-Cross Validation and Grid Search (λ yı bulmak için)


#1-Feature Scaling: *Standardization,*Normalization  
#    Ridge ve Lasso uygulamadan önce coeff leri aynı aralığa çekmemiz lazım ki
#coefficient ları aynı kefeye koyabilelim 
#    Sayılar çok büyükse coeff küçük,
#    Düşükse coef büyük olur.Büyük olan da her zaman kazanır.Bunu istemiyoruz.
#Aynı range de olsunlar ki kıyaslama yapabileyim.
#    *Scale sadece train datasına uygulanır      
#    *Regularization dan önce scale riz ki coefleri karşılaştırabilelim
#    Range ler aynı aralıkta mı değil mi karar veremiyorsak scale uygularız .(Standardization*Normalization)
#not:target a kesinlikle scaling uygulanmaz,çünkü target daki değerler asıl ulaşamız gereken hedefler 
#olduğu için değişmemesi lazım 
#Standardization(z-score)
#mean=0,std=1   --->feature değerlerini bu formata gelecek şekilde dağıtır 
#(-3,3) aralığı 
#Xchanged=(x-μ)/σ

#Normalization:Feature değerlerini 0-1 aralığına getirir.
#En küçük değere 0,en büyük değere 1 verir aralığı buna göre belirler 
#Xchanged=(X-Xmin)/(Xmax-Xmin)

#*hangisi iyi sonuç verirse onu seçeceğiz 
#not:Feature scalin sadece train datasına uygulanır.Tüm dataya uygulanırsa 
#"Data Leakage" ortaya çıkar.(data leakage:test datasının kopya çekmesi)


#Data Leakage:
#    Test datasının görmemesi gereken yerleri görüp kopya çekmesidir.Bu durumda
#overfitting ortaya çıkar.Model güvenirliliği ortadan kalkar .



#2-Cross Validation and Grid Search:
#*Train için acaba datanın doğru yerini mi aldım?
#*Her seferinde datanın farklı yerlerini train ve test olarak ayırır

#k_fold Cross Validation:
#    iteration 1 accuracy=%92
#    iteration 2 accuracy=%95
#    iteration 3 accuracy=%89  --->default k=5
#    .
#    .
#    iteration k accuracy=%90
#    Final accuracy=tüm iterationların ortalaması 
    
#*Datanın her yerinde modelim düzgün çalışıyor mu?Bunu bulmuş oluyoruz 


#LOO Cross Validation:(Leave One Out)
#LOO Cross Validation:
#    iteration 1 accuracy=%92
#    iteration 2 accuracy=%95
#    iteration 3 accuracy=%89  
#    .
#    .
#    iteration NOO accuracy=%90 (NOO:Number of  Observations)
#    Final accuracy=tüm iterationların ortalaması 
    
#    Bir datanın sadece 1 observation ını teste ayırır.Geri kalan hepsini train e ayırır.
#    Bunu her satır için tek tek yapar.Satır sayısı kadar iteration olur.
#    İş yükü çoktur.Bilgisayar çok iyi değilse çok uzun sürer 
    
#Croos Validation uyguladık ve acaba Data Leakage oldu mu bundan şüpheleniyoruz.
#O zaman "Hold Out Test" yapılır. 

#Data--->Test :Hold out test uygulanır  -dataya 2 kere train_test_split uygulanmış oldu
#    --->Train:Train datası tekrar bölünür.Cross validation buraya uygulanır 
#    Sonra daha önce hiç görmediği test datasına hold out test uygulanır.
#    Sonuçlar iyiyse yola devam edilir 
    
#Grid Search :
#    Modeli test ettikten sonra "hyper parameter" ların en optimumunu bulmak için
#    yapılır. 
#    Scikit learn;verdiğimiz her hyper_parameterı iterasyon üzerinden deneyen(grid search)
#    hem de cross validation yapan bir fonksiyona sahiptir.
#    Buna "grid search cross validation" denir
#grid search cross validation-->Hem iterasyon yapar hem hyper metrelerş tek tek dener 
#best model best paraeter ortaya çıkar  


#Scaling yapılmadığı zaman;
#1-Model buna çok daha fazla ağırlık verir.Bu da feature önemsiz bile olsa ön plana çıkarır.
#2-Gradient Descen Tabanlı(Linear Reg.Lojistic Reg)modeller çok daha hızlı  çalışır 
# Distance tabanlı(mesafe tabanlı) modellerde büyük olan dataya çok daha büyük ağırlık vermez 
 

#!!!!!!!!!!!!!!!!!!ROAD MAP!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#--->Exploratory Data Analysis and Visualization 
#--->Machine Learning 
#    *Train|Test Split 
#     X_train,X_test,y_train,y_test=train_test_split()
    
#    *Scalling(if needed)
#     scaler=scaler_name()
#     scaler.fit_transform(X_train)
#     scaler.transform(X_test)
    
#    *Modelling 
#     model=model_name().fit(X_train,y_train)
#     y_pred=model.predict(X_test)
#     y_pred_proba=model.predict_proba(X_Test)
     
#    *Model Performance 
#        Regression --->r2_score,MAE,MSE,RMSE 
#        Classification--->accuracy,recall,precision,f1_score(confusion_matrix,classification_report)
#        Cross Validate--->cross_val_score,cross_validate 
    
#    *Tunning(if needed)
#     grid_param={}
#     GridsearchCV(grid_param)
     
#    *Final Model
#     model=model_name().fit(X,y)
    
#--->Model Deployment 
    
#*What is regularization?
#*Why is it usefull?
    
    
#%%    
#######################NOTEBOOK 4-5######################################    
    
# Geçen dersin özeti
# Bias in data science da karşılığı residual lerdir.
# Underfitting : Özetle, veriyi tam olarak fit edememiş anlamındaydı(high bias, low variance))
    # Data ya da featurelar azdır ya da model complexity yetersizdir. O yüzden oluşur.
    # Çözümü : feature arttırmak, model complexity yi arttırmak vs
# Overfitting  : Özetle, veriyi ezberlemiş anlamındaydı(high variance, low bias)
    # Model çok komplex olabilir(3. derece yeterken 8. dereceden bir polinomla fit etmeye çalışmak)
    # Çözümü: Complexity yi düşürmek, Regularization yapmak(Ridge, Lasso ile bakacağız)
# John Hoca: Teknik sınavlarda ve mülakatlarda bunları mutlaka soruyorlar(Underfitting nedir
# .. overfitting nedir, neden oluşur, çözümü nedir, bunların bias ve variance terimleriyle açıklamaları vs) )   
    
 # multicollinearity: Featurelar arasında çok yüksek ilişki varsa bu yüksek ilişkiye denir.
# Neden istemeyen durumdur peki? Estimator ımızın unstable olmasına sebep olur. Çünkü bu
# .. feature lar arasındaki yüksek korelasyon birbirini etkiler ve estimator unstable olur. 
# .. Korelasyonun -(eksi) olması bir sorun olmaz ama -(eksi) yönde yüksek ilişki varsa bu da sıkıntı
# Bunu aşmak için regularization bize fayda sağlıyor. Bunu ilerleyen kısımlarda göreceğiz   
 # OLS ile biz best line ı buluyorduk. Buradaki regularization ı yaparken yapacağımız işlem
# .. bir penalty(ceza) ekliyoruz. Overfitting durumunda penalty ekleyerek overfitting den kurtarmaya çalışıyoruz
# Birazdan detaylı bakacağız   
    
  # Mülakat sorusu: Overfitting i nasıl çözersin? (Cevap: ridge ile, lasso ile)
# Mülakat sorusu: Ridge ve lasso nedir?     # .. gibi sorular gelebilir.
    # "residual" diye bahsedilen konu "the residual sum" zaten yer
    # Ridge residual a penalty ekliyor. Bu da -->  "lambda * Eğimin kare toplamı
        # "residual" diye bahsedilen konu "the residual sum" zaten yer
    # Lasso residual a penalty ekliyor Bu da  --> "lamda * Eğimin mutlak değer toplamı"
    # l0: scaler bir sayı
    # Lasso Regression(L1): mutlak değer
    # Ridge Regression(L2): square
    
# Elastic net: Ridge ve lassonun combine hali. Belli oranlarda Ridge ve Lassodan alınarak uygulanır
# John Hoca: Tam bir model olarak adlandırılmayabilir  
    
# 2 nokta olduğunu düşünelim. Modele bunu öğren dedik. Model bir line çizmiş(soldaki grafik)
# Test e gidince model öğrenememiş olacak test hatası yüksek olacak(sağdaki grafik)   
    
# Biz o çizgiyi aşağı doğru çekiyoruz ve train de bir hatam oluşacak ama(bunu göze alıyoruz)
# .. test durumuna gidince de hatam azalmış oluyor. Bu da bizim için overfitting e 
# .. göre daha ideal bir durum.
# Bunun ayarlamasını formüldeki "lambdayı" değiştirerek ayarlayacağız
# Sonuç olarak bir ifade ekleyerek çizgiyi oynatmış olup test hatasını azaltıyoruz.
# Ridge bu şekilde çalışır
# Normalde hata bizden kaynaklanıyor. Mesela komplex bir model seçmisizdir ya da featurelarda
# .. sıkıntı vardır ezbere gitmiştir vs

# Orion Hoca: regresyon katsayısı = coef = slope .. bunu değiştiriyoruz    
# Kafamıza göre bir şey eklemiyoruz. Burada slope un bilgisini kullanıyoruz.    
# Sonuç olarak lambdanın değişik değerleri için eğim değişiyor. En minimum hatayı veren lambdayı belirleyeceğiz
# Lambdanın nasıl belirlendiğini birazdan göreceğiz.
# Ridge ve Lasso aynı işlemi yapıyor sadece biri formülde kare alıyor biri mutlak değer   
    
# Ridge ve Lasso nun artıları ve eksileri var
# Ridge ile ilgili konuşalım
# NOT: Feature selection için uygun değil demek; featureların katsayılarını ya çok yüksek verir
# .. ya da 0 a yaklaşır katsayılar. Çünkü formülde karesini aldığı için. 
# .. Sonuç olarak feature selection yapmakta zorlanırız
# Lasso da bazı featureları siler. Lasso da feature selection yapabilirim    
    
# Lassoya geçelim
# Ridge de katsayı 0 a hiç bir zaman gitmiyor
# Lasso da katsayı 0 olabiliyor
# Katsayılar bize hangi feature un daha önemli olduğunu gösteriyor(Scaling yaptıktan sonra)
# Lasso da 50 feature ı 5 feature a indirebiliyor. Ridge de 50 feature 50 olarak kalıyor    
    
# Lasso avantaj ve dezavantajları
# Not: Dezavantaj açıklaması olarak Feature larımın çok yüksek korelasyonu varsa birini tutup 
# .. diğerlerini atabiliyor lasso    
    
# Ridge de featureları atmadığı için yüksek değerleri featureları gruplayabilirsiniz
# Lasso da featureları attığı için kalan featurelardan seçilebilir
# Orion hoca: feature selection yapmak istiyorsanız lassoyu kullanacağız    
    
    
# Elastic-NET: Ridge ve Lassonun toplamı(kombinasyonu). "fi" diye bir katsayı var.
# Bu katsayının değerine göre hangi modelden ne kadar kullanacağını belirleyeceğiz
# NOT: Altta düzeltilen yer --> (1-"fi")/2
# fi=1 olursa, sadece ridge i kullanacak
# fi=0 olursa sadece lasso gibi çalışacak
# fi=0.5 olursa iki yöntemi eşit oranda kullanmış olacak
# grid search kullanarak bu değeri belirleyeceğiz    
    
# REGULARIZATION
# Regularization için ridge ve lasso dedik(Elastic nette kullanabiliriz dedik)
# Alttakiler yapılarak da regularization yapılır    
    
    
# Feature Scaling nedir?
# Data science da çok kullanılan yöntem. Kullanma sebeplerimiz;

# 1.Gradient descent
# Biz ML de katsayıları optimize edip best line ı bulmaya çalışıyoruz.
# Katsayıları optimize etmek için ilerde gradient descent kullanacağız
# Gradient descent ile türev bilgisini kullanarak katsayıları optimize ediyoruz
# Türev = Eğim/Değişim. Eğrinin ne kadar değiştiğini söylüyor bize. Eğimden teğet
# .. geçirerek eğimi hesaplıyorduk
# Discrete deki karşılığı farktır(çıkartma). a1-a2
# Türeve bakarak adımlarda değişiklik yapıyor gradient descent
# Gradient descent te türev olduğu için, örneğin auto-scoutta arabanın fiyatları
# .. için tahmin ile gerçek değerler arasında fark çoksa gradient descent ayarlamayı uzun
# .. yapıyor. O yüzden scaling yaparak gradient descent in hızlı çalışmasını sağlıyoruz

# 2.Algoritmamız uzaklık tabanlı(temelli) ise
# Uzayda aynı ölçekte değilse. Bir değişkende çok büyük değişiklikler başkasında küçük değişiklikler
# .. olacak scale o yüzden scaling gerekli

# 3. Model coefficient da scaling yapmazsak coefficientlar bizim için anlamlı olmaz
# .. o yüzden scaling yapmadan coefficientlar hakkında yorum yapmak iyi değildir    
    
# Scaling yapmakta fayda var yapmazsak performance düşebilir
# 2 türlü scaling var.
# Deviation =1, mean=0 olacak şekilde veriyi dönüştürüyor = standart scaling (z score normalization, genellikle(-3, 3))
# Değerler 0-1 arasında olacak şekilde veriyi dönüştürüyor =

# NOT:class chatteki bir soru: label(Target/Bağımlı) degere(değişkene) scaling yapilimiyor?
# Orion Hoca: ML de label kutsal. mümkünse dokunmuyoruz    

# Fit i ve transform u train e yapıyoruz
# test e transform u uygulayabiliriz. fit i uygulamıyoruz.
# John Hoca: Burada bunu bilsek yeterli

# Cross Validation

# Elindeki datayı %30 test, %70 test olarak ayırıyor ve bir sonuç üretiyor
# Buradaki sorun. Bu %30 u başka yerlerden seçseydi? Seçtiğimiz %30 kötü bir yerden seçildiyse?
# .. ya da değerlerin dengesiz olduğu bir şekilde seçildiyse?? Çözüm için;
# Bunu biz farklı farklı yerlerden böleyim ve hepsi için sonuç üreteyim diyoruz
# En son bunların ortalamasını alalım diyoruz
# K-fold : k değerinin kaç olacağını belirtiyoruz. Genelde 5 veya 10 seçilir
# 5 olursa 4ü(yani %80) traine 1 i test e atarak yapar. 10 olursa; 9 a 1 şeklinde

# Diğer bir yaklaşım. LOO(Leave one out)
# Bu bilgisayarı çok yoruyor. Büyük datada sıkıntı. Çok tercih edilmiyor
# Bütün hepsini alıyor train ediyor- tek değerle test ediyor
# sonra diğer 1 i ayırıyor test e kalanıyla test.. bu şekilde devam ediyor

# Genelde modelde train_test_split yapıyoruz. Tahmin yapıyoruz vs
# Biz o kısmı validation için kullanıyoruz burada. YANİ;
# veriyi %80 , %20 ayırdık sonra
# %80 lik ayırdığımız kısmı bir daha %80 ve %20 diye ayırıyorum ve bu şekilde
# .. denemeler yapıyoruz. En son başta ayırdığımız test ile modelimizin performansını görüyoruz
# John Hoca: Notebookta detaylı anlatacağız

# Bir diğer konu grid search
# hyperparameters: Her modelin parametreleri vardır. Biz parametre için hangi değerin
# .. daha iyi olduğunu belirlemek için modelin performansını etkileyen parametrelerdir
# .. İlerde farklı yöntemlerde farklı parametreleri göreceğiz.
# Biz bu hyperparametreleri optimize etmek için grid search kullanıyoruz
# Grid search bütün kombinasyonları deniyor
# Orion Hoca: Modelde bizim ayarladığımız parametrelere hyperparametre diyoruz.

# Mülakat sorusu: Regularization nedir? Ne için faydalıdır?
# Cevap: ML de kullandığımız bir yaklaşımdır. Overfitting sorununu çözeriz. En büyük faydası bu diyebiliriz




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (10,6)
pd.set_option('display.max_columns', 100)
#Polinomial Regression

df=pd.read_csv("Advertising.csv")
df.head()
#      TV  radio  newspaper  sales
#0  230.1   37.8       69.2   22.1
#1   44.5   39.3       45.1   10.4
#2   17.2   45.9       69.3    9.3
#3  151.5   41.3       58.5   18.5
#4  180.8   10.8       58.4   12.9

df.shape
#(200, 4)
X=df.drop("sales",axis=1)
y=df["sales"]
X.head()

#      TV  radio  newspaper
#0  230.1   37.8       69.2
#1   44.5   39.3       45.1
#2   17.2   45.9       69.3
#3  151.5   41.3       58.5
#4  180.8   10.8       58.4


#Polynomial Conversion

from sklearn.preprocessing import PolynomialFeatures
#5.dereceden sentetik bir overfitting durumu oluşturduk.Bunu yapmamızın sebebi 
#overfitting nasıl düzlelteceğimizi ridge ve lasso kullanarak görecez 
# Polynomial uygulamamızın sebebi dereceyi değiştirerek overfitting sağlamak
# Sonra bunu ridge ve lasso ile çözeceğiz
polynomial_converter=PolynomialFeatures(degree=5,include_bias=False)
# Featurelarımızı 5. dereceden polynomial featurelara dönüştürüyoruz
poly_features=polynomial_converter.fit_transform(X)
# fit ve transform ediyoruz oluşturduğumuz nesneyi
# Lineer regression da x ve y yi kullanıyorduk. Burada poly_features ları kullandık

poly_features.shape 
#(200, 55)
# 3 feature du 55 feature a çıktı 5. dereceden olunca
# Ares Hoca: bu polyfeatures ile oluşturduğumuz modelin overfitting olduğunu biliyoruz 

#Train | Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)
# test_size    : Datanın ne kadarının test e ayrılacağına karar verir.
# random_state : Kod her çalıştığında aynı sonuçlar alınmak isteniyorsa kullanılır

#Scaling the Data
# Bir çok scaling yöntemi var ama burada en çok kullanılan 3 tanesini import ettik
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler # Robustscaler is used when outlier could be present
scaler = StandardScaler()   # Burada bir instance alıyoruz.

scaler.fit(X_train)  # Apply just for X_tarin not for X_test (Kalıp çıkarma)
# X_train datamızın kalıbını çıkartıyoruz fit te.(Transforma kesip biçiyoruz)
#burda aldığımız

X_train_scaled = scaler.transform(X_train) # Apply transform according to fit
X_train_scaled
# Transformda uygulamış olduk. Bütün değerler değişti öncekine göre

# Orion hoca: scaling bir data preprocessing işlemidir.modele sokmadan önce yapılan bir işlem

# Class chat ten: Algoritma distance tabanlıysa feature scaling yapılır.
# .. Model coefficient ları da aynı ölçeğe getirirsek katsayılar hangi feature ın etkisinin çok olduğunu gösterir.

# Class chat soru: Hocam scaling i neden poly_features a değil de X_train e uyguladık
# Cevap: X_train ve y_train i polyfuture dan üretildi zaten. O yüzden polyfeature tarih oldu

# Class chat soru: scaling yapilmis data ile sokulmamis data ayni sonucumu verir hocam
# Cevap Orion Hoca: prediction için birşey değişmez ama feature importance durumu değişir.linear regresyon özelinde


# Train için kalıp çıkartıp(fit yapıp) ayrıca test içinde kalıp çıkartırsak sonuçlar yanlış oluyor. Yani
# Test i de fit yaparsak domainler değişmiş oluyor ve bu da data leak e sebep oluyor
# Bundan dolayı test e sadece transform yapıyoruz
# Sonuç: 1 fit, 2 transform
X_test_scaled = scaler.transform(X_test)
X_test_scaled
#ortalama ve standat sapma bilgisi scaler.fit(X_train) ile sadece X_trainden alındı
#scaler.transform(X_test) burada donusum X Train den alınan bilgilerle yapılır 
#c_test e fit uygulamamamızın sebebi lealking i önlemek 
#data şu anda x_test den hiçbirşey öğrenemedi 
#checking std = 1 and mean = 0
#this gives us the z-scores. so it's also called z-score scaling
#These values show where in the normal distribution they correspond to the z score.

pd.DataFrame(X_train_scaled).agg(["mean", "std"]).round() #Applying aggregation across all the columns, mean and std will be found for each column in the dataframe
# Scaling yaptık ancak bunu bir kontrol edelim her bir sütun için. Mean i gerçekten 0, std sapması 1 mi diye

pd.DataFrame(X_test_scaled).agg(["mean", "std"]).round()

#Polinomial Regression Model Building

from sklearn.linear_model import LinearRegression
lm = LinearRegression()          # Modeli kuruyoruz

lm.fit(X_train_scaled, y_train)  # Modeli eğitiyoruz burada

y_pred = lm.predict(X_test_scaled)          # Tahmin yapıyoruz
y_train_pred = lm.predict(X_train_scaled)   # Bunu neden yaptık? train datasının da performansına
# .. bakıp test datası ile karşılaştıracağız ki overfitting olup olmadığını anlayacağız

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




# Fonksiyonumuza y_train, y_train_pred, y_test, y_pred i veriyoruz. İsmi de 'linear' olsun
ls =train_val(y_train, y_train_pred, y_test, y_pred, "linear") # Evaluate the result. Overfitting?
ls
#      linear_train  linear_test
#R2        0.997607     0.764990
#mae       0.186213     0.665959
#mse       0.062968     6.634847
#rmse      0.250934     2.575820
# Performanslara bakalım
# r2: 0.99 a 0.76,    .. Bağımsız değişkenlerin bağımlı değişkeni açıklama oranı(Açıklanan varyans) çok düştü
# rmse : 0.25 , 2.57  .. Hata 10 katına çıkmış
# Bunları görünce diyoruz ki overfitting var burada(Train i ezberlemiş ve hata az. Görmediği datada(Test te) hata artmış)


#Multicollinearity
#If there is a strong correlation between the independent variables, this situation is called ** Multicollinearity.
#Multicollinearity prevents my model from detecting important features.

# Multicollinearity: Featurelar arasında çok yüksek ilişki varsa bu yüksek ilişkiye denir.
# 1.Bu ilişki modelin kararlılığını, tutarlılığını bozuyor.
# 2.Feature importance ı belirlerken bunda zorlanıyoruz. Bu da istenilen bir şey değil
# Bundan kurtulmak istiyoruz. Ridge ve Lasso bu işi yapıyordu(Ridge ve Lasso overfitting den de kurtarıyordu)
# Jason Hoca: Feature'lar arasında fazlalık (redundance) vardır gibi yorumlayabiliriz

def color_red(val):
    if val > 0.90 and val < 0.99:
        color = 'red'
    else:
        color = 'black'
    return f'color: {color}'
# Yüksek korelasyonlu olanları kırmızı renkli diğerlerini siyah renkli yapacak. Altta uygulayalım

pd.DataFrame(poly_features).corr().style.applymap(color_red)


#CROSS VALIDATION

from sklearn.metrics import SCORERS
list(SCORERS.keys())
# Hangi score ları kullanabileceğimizi görebiliriz bu kod ile
from sklearn.model_selection import cross_validate

model = LinearRegression()
scores = cross_validate(model, X_train_scaled, y_train, scoring = ['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv = 5)
#cv datayı kaça bölmek istediğinizi belirler.Normalde cv ne kadar fazla olursa sonuç o kadar 
#güvenilir olur.alcağımız değer sonuçların hassasiyetine bağlıdır.5 üstü değerler için
#güçlü bilgisayarlar gerekir
scores  
pd.DataFrame(scores, index = range(1,6))
# Her bir satırda datanın farklı yerlerden alınan sonuçları görüyoruz
# Bunların ortalamalarını görerim altta
#   fit_time  score_time   test_r2  test_neg_mean_absolute_error  \
#1  0.010254    0.002393  0.367902                     -1.271247   
#2  0.001554    0.001791  0.879658                     -0.710463   
#3  0.002002    0.000998  0.977560                     -0.395033   
#4  0.002001    0.001000  0.989369                     -0.418977   
#5  0.001001    0.002001  0.166596                     -1.474093   

#   test_neg_mean_squared_error  test_neg_root_mean_squared_error  
#1                   -19.204259                         -4.382266  
#2                    -2.713292                         -1.647207  
#3                    -0.622901                         -0.789240  
#4                    -0.289361                         -0.537923  
#5                   -15.409856                         -3.925539  

scores = pd.DataFrame(scores, index=range(1,6))
scores.iloc[:, 2:].mean()
#üsteki ilgilendiğimiz değerlerin ortalamalarını aldık
# "fit_time" ve "score_time" bizi ilgilendirmediği için diğerlerini aldık
# Fold lara göre datayı taradığı için bu sonuç daha güvenilir
#test_r2                             0.676217
#test_neg_mean_absolute_error       -0.853963
#test_neg_mean_squared_error        -7.647934
#test_neg_root_mean_squared_error   -2.256435
train_val(y_train, y_train_pred, y_test, y_pred, "linear")
#      linear_train  linear_test
#R2        0.997607     0.764990
#mae       0.186213     0.665959
#mse       0.062968     6.634847
#rmse      0.250934     2.575820

#cross val e sokmadan önce aldığımız test değerleri ile kıyasladık
#değerlerde tutarsızlık var 
sns.lineplot(data = scores.iloc[:, 2:])

lm.coef_

lm_df = pd.DataFrame(lm.coef_, columns = ["lm_coef"])
lm_df.head()
#     lm_coef
#0  13.942084
#1  -3.147845
#2   0.622117
#3 -58.348614
#4  17.227695
# Bu coefficientlardan 3 ü bizim gerçek feature umuz. Diğerleri sentetik
# Coefficientlar scaling yaptığımız için hangi coefficient ın bizim için önemli olduğunu görebiliriz

#Ridge Regression

from sklearn.linear_model import Ridge 

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ridge_model = Ridge(alpha=1, random_state=42)  # alpha=1 default değeri zaten

ridge_model.fit(X_train_scaled, y_train)

y_pred = ridge_model.predict(X_test_scaled)
y_train_pred = ridge_model.predict(X_train_scaled)

rs = train_val(y_train, y_train_pred, y_test, y_pred, "ridge")
rs

#      ridge_train  ridge_test
#R2       0.988677    0.982511
#mae      0.338149    0.482446
#mse      0.297970    0.493743
#rmse     0.545866    0.702669
# Alttaki Comparision kısmı eski sonuçlar
# Bizim çıktımız yeni sonuçlar
# Sonuçta Ridge kullandığımız için overfitting den kurtulduğumuzu söyleyebiliriz
# Burada bias de düşük, varyansta düşük. Good-fitting oldu
# Acaba sonuçlarım doğru mu diye cross-validation a bakalım bir de

#Comparision

pd.concat([ls, rs], axis=1)

#    linear_train	linear_test	ridge_train	ridge_test
# R2	0.997607	0.764990	0.988677	0.982511
# mae	0.186213	0.665959	0.338149	0.482446
# mse	0.062968	6.634847	0.297970	0.493743
# rmse  0.250934	2.575820	0.545866	0.702669

#ridge ile train setine hata vererek test setindeki skorların iyileştiğini gördük 

#For Ridge Regression CV with alpha:1

model=Ridge(alpha=1,random_state=42)
scores = cross_validate(model, X_train_scaled, y_train,
                    scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv=5)

pd.DataFrame(scores,index=range(1,6))
#   fit_time  score_time   test_r2  test_neg_mean_absolute_error  \
#1  0.021240    0.002999  0.988913                     -0.405789   
#2  0.001000    0.001000  0.969951                     -0.551469   
#3  0.001000    0.001000  0.950978                     -0.552320   
#4  0.001001    0.000999  0.991259                     -0.365106   
#5  0.001013    0.000987  0.986854                     -0.343011   

#   test_neg_mean_squared_error  test_neg_root_mean_squared_error  
#1                    -0.336840                         -0.580379  
#2                    -0.677493                         -0.823099  
#3                    -1.360795                         -1.166531  
#4                    -0.237919                         -0.487769  
#5                    -0.243072                         -0.493023  

scores = pd.DataFrame(scores, index = range(1, 6))
scores.iloc[:,2:].mean()
#test_r2                             0.977591
#test_neg_mean_absolute_error       -0.443539
#test_neg_mean_squared_error        -0.571224
#test_neg_root_mean_squared_error   -0.710160
# Yukarda 0.98 di, burada 0.97 bulduk r2 yi. Yani sonuçlar tutarlı
# CV ile sonucumuzu doğrulamış olduk.
# John Hoca: Patron sorarsa gerçek sonuç nedir diye. 0.97 deriz.
# Çok hassas bir şekilde sonuç isterseniz CV kullanılmalı

train_val(y_train, y_train_pred, y_test, y_pred, "ridge")
#      ridge_train  ridge_test
#R2       0.988677    0.982511
#mae      0.338149    0.482446
#mse      0.297970    0.493743
#rmse     0.545866    0.702669

sns.lineplot(data = scores.iloc[:,2:])

ridge_model.coef_

rm_df = pd.DataFrame(ridge_model.coef_, columns = ["ridge_coef_1"])

pd.concat([lm_df,rm_df],axis=1).head()

#   lm_coef      ridge_coef_1
#0  13.942084      3.428822
#1  -3.147845      0.689086
#2   0.622117      0.170305
#3 -58.348614     -0.910850
#4  17.227695      2.723077


#Choosing best alpha value by RidgeCV

# Hiperparametreyi optimize edeceğiz gridsearch ile
# En uygun hyperparametreyi seçmek için kullanılan parametredir grid search
# Burada RidgeCV diye bir metod var ona bakacağız önce. En uygun alpha yı CV yaparak buluyor
from sklearn.linear_model import RidgeCV

alpha_space = np.linspace(0.01, 1, 100)  # Alpha kümesi oluşturalım
alpha_space

ridge_cv_model = RidgeCV(alphas=alpha_space, cv = 5, scoring= "neg_root_mean_squared_error")
# alphas=alpha_space : Bu değerlerde ara
# CV= 5 olsun
# her bir alpha degeri icin 5 kere skor aliyor sonra o skorun ortalamasini donduruyor
# scoring= "neg_root_mean_squared_error": Bu scora göre bak. NOT: Buraya 1 score yazabiliyoruz.
# .. maximize etmek için negatif aldık(neg_root_mean_squared_error)

ridge_cv_model.fit(X_train_scaled, y_train) # Eğitim yapıyoruz

ridge_cv_model.alpha_  # En uygun alpha değeri 0.02 olarak bulunmuş
#0.02

#rmse for ridge with CV
ridge_cv_model.best_score_   
#-0.6530406519552931
# alpha=0.02 için bulunan en iyi score = -0.65(cv=5 di yani alpha 0.02 icin 5 degerin ortalamasi 0.65)

y_pred = ridge_cv_model.predict(X_test_scaled)   # Bu scaled ile tahmin yapalım şimdi
y_train_pred = ridge_cv_model.predict(X_train_scaled)

rcs = train_val(y_train, y_train_pred, y_test, y_pred, "ridge_cv")
rcs

#      ridge_cv_train  ridge_cv_test
#R2          0.994491       0.983643
#mae         0.244842       0.442087
#mse         0.144977       0.461803
#rmse        0.380758       0.679561

#ekstradan cv yapmamiza gerek yok cunku ridge_cv_model.best_score_ bize en iyi scoru dondurdu
#bu cv sonucunda aldigimiz tek seferlik skor test setinin rmse si ile karsilastirildigi zaman 
#birbirlerine cok yakin oldugunu goruyoruz(-0.6530406519552931-0.679561)
#buna bakarak modelimiz tutarli diyebiliriz 

pd.concat([ls,rs, rcs], axis = 1) # R2 çok az artmış oldu 0.982511(ridge_test-r2) --> 0.983643(ridge_cv_test-r2)
#      linear_train  linear_test  ridge_train  ridge_test  ridge_cv_train  \
#R2        0.997607     0.764990     0.988677    0.982511        0.994491   
#mae       0.186213     0.665959     0.338149    0.482446        0.244842   
#mse       0.062968     6.634847     0.297970    0.493743        0.144977   
#rmse      0.250934     2.575820     0.545866    0.702669        0.380758   
#
#      ridge_cv_test  
#R2         0.983643  
#mae        0.442087  
#mse        0.461803  
#rmse       0.679561  

#alpha yi degistirince skorlarin biraz daha iyilestigini gorduk
#ork rmse (0.702669-0.679561 )

ridge_cv_model.coef_
rcm_df=pd.DataFrame(ridge_cv_model.coef_,columns=['ridge_cv_coef_0.02'])

pd.concat([lm_df,rm_df,rcm_df],axis=1).head()
#     lm_coef  ridge_coef_1  ridge_cv_coef_0.02
#0  13.942084      3.428822            6.383047
#1  -3.147845      0.689086            0.709275
#2   0.622117      0.170305            0.478838
#3 -58.348614     -0.910850           -8.015483
#4  17.227695      2.723077            3.997813

#LASSO REGRESSION 

# Aynı şeyleri Lasso ile yapalım
from sklearn.linear_model import Lasso, LassoCV

lasso_model=Lasso(alpha=1,random_state=42)

lasso_model.fit(X_train_scaled,y_train)

y_pred=lasso_model.predict(X_test_scaled)
y_train_pred=lasso_model.predict(X_train_scaled)

lss=train_val(y_train,y_train_pred,y_test,y_pred,'lasso')
lss# Bu scorumuz doğru mu diye CV ile test edeceğiz
#      lasso_train  lasso_test
#R2       0.919650    0.918590
#mae      1.018355    1.017959
#mse      2.114491    2.298390
#rmse     1.454129    1.516044


pd.concat([ls,rs,rcs,lss],axis=1)
#      linear_train  linear_test  ridge_train  ridge_test  ridge_cv_train  \
#R2        0.997607     0.764990     0.988677    0.982511        0.994491   
#mae       0.186213     0.665959     0.338149    0.482446        0.244842   
#mse       0.062968     6.634847     0.297970    0.493743        0.144977   
#rmse      0.250934     2.575820     0.545866    0.702669        0.380758   

#      ridge_cv_test  lasso_train  lasso_test  
#R2         0.983643     0.919650    0.918590  
#mae        0.442087     1.018355    1.017959  
#mse        0.461803     2.114491    2.298390  
#rmse       0.679561     1.454129    1.516044 

#diger modellere baktigimizda bu modelde underfitting sonucu aldigimizi gorduk
#ork ridge_cv_train r2-0.994491,lasso_train r2-0.919650 
#degerlere bakarak alpha degerini degistirmemiz gerektigi yorumunu yaptik 

#For Lasso CV with Default Alpha:1
model = Lasso(alpha=1, random_state=42)
scores = cross_validate(model, X_train_scaled, y_train,
                        scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv=5)

pd.DataFrame(scores, index = range(1, 6))
#   fit_time  score_time   test_r2  test_neg_mean_absolute_error  \
#1  0.005245    0.002131  0.924246                     -1.155268   
#2  0.001197    0.000280  0.955624                     -0.656358   
#3  0.000496    0.001124  0.888327                     -1.072936   
#4  0.001012    0.000411  0.896199                     -1.255165   
#5  0.001539    0.001020  0.903545                     -1.056538   

#   test_neg_mean_squared_error  test_neg_root_mean_squared_error  
#1                    -2.301550                         -1.517086  
#2                    -1.000530                         -1.000265  
#3                    -3.099952                         -1.760668  
#4                    -2.825429                         -1.680901  
#5                    -1.783470                         -1.335466  

scores = pd.DataFrame(scores, index = range(1, 6))
scores.iloc[:,2:].mean()   # 0.918 di önceki, şimdi 0.913. Değerler birbirine yakın
#test_r2                             0.913588
#test_neg_mean_absolute_error       -1.039253
#test_neg_mean_squared_error        -2.202186
#test_neg_root_mean_squared_error   -1.458877

train_val(y_train,y_train_pred,y_test,y_pred,'lasso')
#      lasso_train  lasso_test
#R2       0.919650    0.918590
#mae      1.018355    1.017959
#mse      2.114491    2.298390
#rmse     1.454129    1.516044

#lasso test 0.918590 di,cv sonunda 0.913588 cok fark yok,modelin tutarli oldugunu gorduk
#fakat diger modellere gore dusuk degerler verdi 
#bunu alpha ile oynayarak duzeltmeye calisacaz 

#tunning islemi en iyi parametreyi secmeye calisacaz
#ridge ve lasso icin kendi modellerinin kullandik diger butun modeller icin grid_search kullanilacak 

sns.lineplot(data = scores.iloc[:,2:])

lasso_model.coef_ # Lasso bazı katsayıları 0 a çekiyordu. Sadece 2 tanesini muhafaza etmiş
# Bunu alpha = 1 için yaptık. Bu ideal alpha değeri değil şu an. O yüzden bu sağlıklı bir sonuç değil
#2 feature kaldi 
#55 feature la ridge yuzde 98 li bir skor mu yoksa 
#2 feature la yuzde 91 li skor mu m,bu tercih duruma gore degisir 
#ork:araba fiyati tahmin etme sitesi olsun 55 feature girdisi 
#hizmet alacaklari zorlayabilecegi icin 2 feature li lasso daha 
#kullanisli olabilir 
#ML nin amaci min feature la max skoru elde etmektir 
#lasso sonucu modelin Comlexity si azaldi(2 feature)
#feature selection icin lasso kullanilir

lsm_df = pd.DataFrame(lasso_model.coef_, columns = ["lasso_coef_1"])

pd.concat([lm_df,rm_df,rcm_df, lsm_df], axis = 1).head()

#     lm_coef  ridge_coef_1  ridge_cv_coef_0.02  lasso_coef_1
#0  13.942084      3.428822            6.383047      0.696016
#1  -3.147845      0.689086            0.709275      0.000000
#2   0.622117      0.170305            0.478838      0.000000
#3 -58.348614     -0.910850           -8.015483      0.000000
#4  17.227695      2.723077            3.997813      3.490946


#Choosing best alpha value by LassoCV
lasso_cv_model = LassoCV(alphas = alpha_space, cv = 5, max_iter=100000, random_state=42)
#default da mse skorunu dondurur,skoru kendimiz secemiyoruz burada 
lasso_cv_model.fit(X_train_scaled, y_train)

lasso_cv_model.alpha_
#0.01
y_pred = lasso_cv_model.predict(X_test_scaled)   #Lasso(alpha =0.01)
y_train_pred = lasso_cv_model.predict(X_train_scaled)

alpha_space[::-1]
lasso_cv_model.mse_path_[-1].mean() #mse skor for cv #son index i aldik cunku 0 a yakin degerler oralar 
#0.4118332980702656
lcs = train_val(y_train, y_train_pred, y_test, y_pred, "lasso_cv")
lcs
#      lasso_cv_train  lasso_cv_test
#R2          0.988824       0.986295
#mae         0.339065       0.440975
#mse         0.294098       0.386919
#rmse        0.542308       0.622028

pd.concat([rs, rcs, lss, lcs], axis = 1)


# Lasso_cv_test r2 sonucu, ridge_cv_test_r2 ye göre daha iyi sonuç verdi
# Bu data için Lassonun daha uygun olduğu söylenebilir

#      ridge_train  ridge_test  ridge_cv_train  ridge_cv_test  lasso_train  \
#R2       0.988677    0.982511        0.994491       0.983643     0.919650   
#mae      0.338149    0.482446        0.244842       0.442087     1.018355   
#mse      0.297970    0.493743        0.144977       0.461803     2.114491   
#rmse     0.545866    0.702669        0.380758       0.679561     1.454129   

#      lasso_test  lasso_cv_train  lasso_cv_test  
#R2      0.918590        0.988824       0.986295  
#mae     1.017959        0.339065       0.440975  
#mse     2.298390        0.294098       0.386919  
#rmse    1.516044        0.542308       0.622028 

lasso_cv_model.coef_
# Orion Hoca: 0 demek o feature yok demek o yüzden lassoyu feature selectionda kullanıyoruz
#buradan 10 feature geldi 
#2 feature la 91 lik 10 feature la 98 lik skor  geldi 
#cv sonunda daha iyi skor dondurdu  

lcm_df = pd.DataFrame(lasso_cv_model.coef_, columns = ["lasso_cv_coef_0.01"])

pd.concat([rm_df, lsm_df, lcm_df], axis = 1).head(10)
#   ridge_coef_1  lasso_coef_1  lasso_cv_coef_0.01
#0      3.428822      0.696016            4.270214
#1      0.689086      0.000000            0.180374
#2      0.170305      0.000000            0.139659
#3     -0.910850      0.000000           -3.146055
#4      2.723077      3.490946            4.153400
#5      0.112909      0.000000           -0.000000
#6     -0.404686      0.000000            0.000000
#7      0.287950      0.000000            0.062497
#8     -0.091937      0.000000            0.000000
#9     -1.014535      0.000000           -0.000000

#alpha katsayisi modelin agresifligini gosteriyordu,1 den 0.01 e 
#dustugu zaman 

from yellowbrick.model_selection import FeatureImportances
from yellowbrick.features import RadViz

viz = FeatureImportances(ridge_cv_model, labels=pd.DataFrame(X_train).columns) # modelimizi FeatureImportances içine veriyoruz
visualizer = RadViz(size=(720, 3000))
viz.fit(X_train, y_train)
viz.show()

# Çıktıda göre 0. feature, 4. feature, 20. feature vs diye önem sırasına göre sıralanmış
# Şeklin en altında ters bir orantı var ama 3. feature, 40. feature vs de önemli

from yellowbrick.model_selection import FeatureImportances
from yellowbrick.features import RadViz

viz = FeatureImportances(ridge_cv_model, labels=pd.DataFrame(X_train).columns) # modelimizi FeatureImportances içine veriyoruz
visualizer = RadViz(size=(720, 3000))
viz.fit(X_train, y_train)
viz.show()

Feature importances with Ridge

# Çıktıda göre 0. feature, 4. feature, 20. feature vs diye önem sırasına göre sıralanmış
# Şeklin en altında ters bir orantı var ama 3. feature, 40. feature vs de önemli

Feature importances with Lasso

from yellowbrick.model_selection import FeatureImportances
from yellowbrick.features import RadViz

viz = FeatureImportances(lasso_cv_model, labels=pd.DataFrame(X_train).columns)
visualizer = RadViz(size=(720, 3000))
viz.fit(X_train, y_train)
viz.show()

# Çıktıda göre 0. feature, 4. feature, 34. feature vs diye önem sırasına göre sıralanmış



#%% LAB-2
# CEMENT_SLUMP_SOLUTION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (10,6)
pd.set_option('display.max_columns', 100)
df = pd.read_csv("cement_slump.csv")
df.head()
# Cement	Slag	Fly ash	Water	SP	Coarse Aggr.	Fine Aggr.:1 kg betonda olan bileşenler 
# SLUMP(cm)	FLOW(cm) : Betonun kıvamını ölçen metrikler
df.info()
df.shape
df.describe().T   # std > mean ise outlier olabilir diyoruz
df.corr()['Compressive Strength (28-day)(Mpa)']
# Cement, Fly Ash, Slag.. Bunların korelasyonları yüksek target ile

plt.figure(figsize=(20,10))
sns.heatmap(df.corr(), annot = True, vmin=-1, vmax=1);
# Multicollinarity var SLUMP ve FLOW arasında : 0.91 .. Bu featureların birbirini baskılama durumu var
# Lineer regression bunu halledemiyordu. Ridge ve Lasso bunları hallediyordu
sns.pairplot(df) # Flow ve slump arasındaki korelasyonu görmüş oluyoruz

plt.figure(figsize =(20,10))
df.boxplot()
# Outlier değerleri describe da değerlendirmiştik. Burada herhangi bir outlier görmüyoruz
# Genel inside elde ettik. Şimdi modellemeye geçelim

X = df.drop("Compressive Strength (28-day)(Mpa)", axis =1)
y = df["Compressive Strength (28-day)(Mpa)"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

########## Scaling the Data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
scaler = StandardScaler() # will be used in pipeline later
# if you don't use pipeline, you can use scaler directly
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train) 
X_test_scaled = scaler.transform(X_test)

# Pipeline
"""
What happens can be described as follows:
Step 1: The data are split into TRAINING data and TEST data according to ratio of train_test_split
Step 2: the scaler is fitted on the TRAINING data
Step 3: the scaler transforms TRAINING data
Step 4: the models are fitted/trained using the transformed TRAINING data
Step 5: the scaler is used to transform the TEST data
Step 6: the trained models predict using the transformed TEST data

pipe.fit(X_train, y_train)--> scaler.fit_transform(X_train) --> lm.fit(scaled_X_train, y_train)
pipe.predict(X_test) --> scaler.transform(X_test) --> lm.predict(scaled_X_test)
"""

########### Linear Regression
from sklearn.pipeline import Pipeline # pipeline is used to combine scaler and model
from sklearn.linear_model import LinearRegression

lm = LinearRegression() # will be used in pipeline later
pipe_lm = Pipeline([("scaler", scaler), ("lm", lm)]) # pipeline is used to combine scaler and model
pipe_lm.fit(X_train, y_train)

y_pred = pipe_lm.predict(X_test) # predict on test data
y_train_pred = pipe_lm.predict(X_train) # predict on train data

ls = train_val(y_train, y_train_pred, y_test, y_pred, "linear") # train and test scores
ls 
# Test scorum tek seferlik 0.91 yani yüksek çıkmış olabilir. Bunu test edeceğiz

############# Cross Validate
#from sklearn.metrics import SCORERS
#list(SCORERS.keys())
from sklearn.model_selection import cross_validate, cross_val_score
model = Pipeline([("scaler", scaler), ("lm", lm)]) # Modelimizi 0 ladık tekrar
scores = cross_validate(model, X_train, y_train, scoring = ['r2', 'neg_mean_absolute_error','neg_mean_squared_error', \
                                                            'neg_root_mean_squared_error'], cv = 5)
# CV= maksimizasyon algoritması.. Orion hoca...

pd.DataFrame(scores, index = range(1,6))

scores = pd.DataFrame(scores, index=range(1,6))
scores.iloc[:, 2:].mean()
# Asıl scorumuz 0.84(test) , bunu train hatası ile karşılaştırıyoruz(0.90)

train_val(y_train, y_train_pred, y_test, y_pred, "linear")

print("train RMSE:", 2.423698/df["Compressive Strength (28-day)(Mpa)"].mean())
print("CV RMSE:", 2.737927/df["Compressive Strength (28-day)(Mpa)"].mean())

pipe_lm["lm"].coef_

lm_df = pd.DataFrame(pipe_lm["lm"].coef_, columns = ["lm_coef"])
lm_df

############# Ridge Regression
from sklearn.linear_model import Ridge
# Modele hata ekleyip bias ve varyans arasındaki dengeyi sağlıyordu
ridge_model = Ridge(alpha=1, random_state=42) # will be used in pipeline later

pipe_ridge = Pipeline([("scaler", scaler), ("ridge", ridge_model)]) # pipeline is used to combine scaler and model
# Pipeline sizin için scaling yapıyor. Modelin kuruyor
# Test setine transform yapıyor ve transform yapılmışı model içerisine koymuş oluyor

pipe_ridge.fit(X_train, y_train)

y_pred = pipe_ridge.predict(X_test)
y_train_pred = pipe_ridge.predict(X_train)

rs = train_val(y_train, y_train_pred, y_test, y_pred, "ridge")
rs
# Score lar birbirine yakın görünüyor ama cross validation yapmalıyız

pd.concat([ls, rs], axis=1)  # combine train and test scores to compare

############# For Ridge Regression CV with alpha : 1
model = Pipeline([("scaler", scaler), ("ridge", ridge_model)]) # Pipeline ı kurduk tekrar
scores = cross_validate(model, X_train, y_train,
                    scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv=5)
pd.DataFrame(scores, index = range(1, 6))

scores = pd.DataFrame(scores, index = range(1, 6))
scores.iloc[:,2:].mean()

train_val(y_train, y_train_pred, y_test, y_pred, "ridge")

pipe_ridge["ridge"].coef_

rm_df = pd.DataFrame(pipe_ridge["ridge"].coef_, columns = ["ridge_coef_1"])

pd.concat([lm_df,rm_df], axis = 1) # Diğer modellerin skorlarıyla karşılaştırma yapayım
# lm_coef: Lineer regression coefficientlar
# ridge_coef_1: Alphası 1 olan coefficientlar
# --flow -2.188378,-1.954987, slump: 1.465706,1.133500
# Lineer regression da flow daha baskılamış gibi(-2.188378 daha yüksek 1.46 ya göre yani baskılamış)
# ridge_coef_1 de değerlerin oransal değeri birbirine yaklaşmış. Ridge çözmüş sorunu
# Ridge de -1.954987 , 1.133500 birbirlerine nispeten yakınlaşmışlar yani ridge multicollinerity yi çözmüş

############## Choosing best alpha value with Cross-Validation
from sklearn.linear_model import RidgeCV # Burada gridsearchCV de kullanabilirdik
alpha_space = np.linspace(0.1, 1, 100) # Hyperparametreyi tanımlıyorum. 100 adet değer ürettik
alpha_space
# Not: Parametre sayısı arttıkça bu kadar fazla örneklem alırsanız bilgisayarı çok kasacaktır

ridge_cv_model = RidgeCV(alphas=alpha_space, cv = 10, scoring= "neg_root_mean_squared_error") # will be used in pipeline later
# alphas=alpha_space parametreleri verdik. Grid search yapıyoruz en ideal parametreyi bulmak için

pipe_ridgecv = Pipeline([("scaler", scaler), ("ridgecv", ridge_cv_model)]) # pipeline is used to combine scaler and model

pipe_ridgecv.fit(X_train, y_train) # Eğitim yapıyorum

pipe_ridgecv["ridgecv"].alpha_ # En iyi alpha buymuş 0.91

# Ridge( alpha = 0.91)
y_pred = pipe_ridgecv.predict(X_test)
y_train_pred = pipe_ridgecv.predict(X_train)  

rcs = train_val(y_train, y_train_pred, y_test, y_pred, "ridge_cv") 
rcs
# Lineer regressionda ve ridge aldığım scora yakın görünüyor
# Bunun içinde bir cross validation yapmayacağız. Üstte yaptık zaten

pd.concat([ls, rs, rcs], axis = 1) # Diğer modellerin skorlarıyla karşılaştırma yapayım
# linear_train rmse 2.423698 den ridge_train rmse 2.433509 e çıkmış hata yani modele hata eklemiş ridge
# ridge_train	ridge_test : alpha = 0.1 iken
# Ridge_cv_train	ridge_cv_test = 0.9 --> hatayı milimetrik düşürmüş(2.432414)
# Hata ekleyip ...

pipe_ridgecv["ridgecv"].coef_

rcm_df = pd.DataFrame(pipe_ridgecv["ridgecv"].coef_, columns=["ridge_cv_coef_0.91"])

pd.concat([lm_df,rm_df, rcm_df], axis = 1) 

############# LASSO
from sklearn.linear_model import Lasso, LassoCV
lasso_model = Lasso(alpha=1, random_state=42)
# Bütün parametreler default ise vanilla parameter deniyor buna
# Büyük alpha büyük hata ya da büyük regularization demek(1 büyük diyebiliriz burada)

pipe_lasso = Pipeline([("scaler", scaler), ("lasso", lasso_model)]) # pipeline is used to combine scaler and model
pipe_lasso.fit(X_train, y_train)

y_pred = pipe_lasso.predict(X_test)
y_train_pred = pipe_lasso.predict(X_train)

lss = train_val(y_train, y_train_pred, y_test, y_pred, "lasso") 
lss
# Genelde train yüksek olur teste göre
# Burada lasso bazı featureları 0 ladığı için böyle oldu

pd.concat([ls, rs, rcs, lss], axis = 1)
# linear_train rmse 2.423698 den ridge_train rmse 2.433509 e çıkmış hata yani modele hata eklemiş ridge
# ridge_train	ridge_test : alpha = 0.1 iken
# Ridge_cv_train	ridge_cv_test = 0.9 --> hatayı milimetrik düşürmüş(2.432414)
# Hata ekleyip ...
# Bir cross validation yapalım

########### For Lasso CV with Default Alpha : 1
model = Pipeline([("scaler", scaler), ("lasso", lasso_model)]) # CV öncesi modeli sıfırlamak lazım
scores = cross_validate(model, X_train, y_train,
                        scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv=5)

pd.DataFrame(scores, index = range(1, 6))

scores = pd.DataFrame(scores, index = range(1, 6))
scores.iloc[:,2:].mean()
# Asıl scorum 0.80 değilmiş 0.68 miş
# Train le karşılaştırınca(0.77) dengelenmiş oldu
# Burada  underfitting durum diyebiliriz(Alpha büyük iken(1 iken))
# Burada ideal alpha belirleyip çözmemiz lazım

train_val(y_train, y_train_pred, y_test, y_pred, "lasso")

model["lasso"].coef_ # 4 feature ı lasso model sıfırlamış
# Bakalım az feature la skorları yukarı çekebilecekmiyiz

lsm_df = pd.DataFrame(model["lasso"].coef_, columns = ["lasso_coef_1"])

pd.concat([lm_df, rm_df, rcm_df, lsm_df], axis = 1)
# Lasso da multicollinearity varsa. Lasso bi tanesini otomatik atıyor
# 7. index lasso -0.710631 , 8. index -0.000000
# Şimdi best alphayı bulup skorlarımızı yükseltelim

############ Choosing best alpha value with Cross-Validation
lasso_cv_model = LassoCV(alphas = alpha_space, cv = 10, max_iter=100000, random_state=42) # will be used in pipeline later
#  max_iter=100000 : Orion H: gradient descent in attığı adımları yükseltiyoruz yeterli olmadığı için

pipe_lassocv = Pipeline([("scaler", scaler), ("lassocv", lasso_cv_model)]) # pipeline is used to combine scaler and model

pipe_lassocv.fit(X_train, y_train)

pipe_lassocv["lassocv"].alpha_ # Best alpha for lasso 0.1

# Lasso(alpha =0.1)
y_pred = pipe_lassocv.predict(X_test)   
y_train_pred = pipe_lassocv.predict(X_train)

lcs = train_val(y_train, y_train_pred, y_test, y_pred, "lasso_cv")
lcs
# CV yapmıştım skorlarımı yorumlayabilirim

pd.concat([ls,rs, rcs, lss, lcs], axis = 1)
# 3.760858 dan hata 2.509041 düştü. Çünkü alphayı düşürmüştük. Böylelikle
# Skorlar birbirine yakınlaştı(Train ve test korları)

pipe_lassocv["lassocv"].coef_ # 0.90 lık skoru 7 feature ile almışız(0.900491)
# Ridge de 9 feature ile 0.90 lık skor mu(0.906476)

lcm_df = pd.DataFrame(pipe_lassocv["lassocv"].coef_, columns = ["lasso_cv_coef_0.1"])

pd.concat([lm_df, rm_df, rcm_df, lsm_df, lcm_df], axis = 1) # (7 fearures ile) test_r2 = 0.90
# Alpha 1 iken çok fazla regularization yapmıştık(4 feature atmıştı)
# lasso_cv_coef_0.1: Daha az bir regularization yaptı ve skorlarımda iyileşme oldu(0.90 a çıktı)

########### Elastic net
from sklearn.linear_model import ElasticNet, ElasticNetCV
elastic_model = ElasticNet(alpha=1, l1_ratio=0.5, random_state=42) # l1_ratio is used to control the amount of L1 and L2 regularization
# alpha=1 , l1_ratio=0.5 : default
pipe_elastic = Pipeline([("scaler", scaler), ("elastic", elastic_model)]) # pipeline is used to combine scaler and model
pipe_elastic.fit(X_train, y_train) # Pipelinedan sonra modelimi eğitim alpha=1 iken

y_pred = pipe_elastic.predict(X_test)
y_train_pred = pipe_elastic.predict(X_train)

es = train_val(y_train, y_train_pred, y_test, y_pred, "elastic")
es

pd.concat([ls,rs, rcs, lss, lcs, es], axis = 1)
# Lasso default 3.760858	3.168025
# Elastic feault 4.793925	4.297257
# Underfitting var diyebiliriz skorlara bakarak

########### For Elastic_net CV with Default alpha = 1 and l1_ratio=0.5
model = Pipeline([("scaler", scaler), ("elastic", ElasticNet(alpha=1, l1_ratio=0.5, random_state=42))])

scores = cross_validate(model, X_train, y_train,
                        scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv=5)

scores = pd.DataFrame(scores, index = range(1, 6))
scores.iloc[:,2:]

scores = pd.DataFrame(scores, index = range(1, 11))
scores.iloc[:,2:].mean() # Asıl skorlarımı 0.54 müş
# underfitting açık bir şekilde görünüyor

train_val(y_train, y_train_pred, y_test, y_pred, "elastic")

pipe_elastic["elastic"].coef_ # 2 feature ı kesmiş Elastic model(Lassodan dolayı)

em_df = pd.DataFrame(pipe_elastic["elastic"].coef_, columns=["elastic_coef_(alp:1, L1:0.5)"])

pd.concat([lm_df, rm_df, rcm_df, lsm_df, lcm_df, em_df], axis = 1)

########### Grid Search for ElasticNet
from sklearn.model_selection import GridSearchCV 
# Elastic net kendine ait Elasticnet cv var ama burada gridsearchcv kullanacağız
elastic_model = ElasticNet(max_iter=10000, random_state=42) 

pipe_elastic = Pipeline([("scaler", scaler), ("elastic", elastic_model)]) # pipeline is used to combine scaler and model

param_grid = {"elastic__alpha":alpha_space,
            "elastic__l1_ratio":[0.1, 0.5, 0.7,0.9, 0.95, 1]}

grid_model = GridSearchCV(estimator = pipe_elastic, param_grid = param_grid, scoring = 'neg_root_mean_squared_error',
                         cv =10, verbose =2)
# verbose : çıktıdakileri yazdırması için

grid_model.fit(X_train, y_train)

grid_model.best_params_
# 'elastic__l1_ratio': 1 : direk lassoyu seçti hatalar aynı olacak concat kısmında göreceğiz

y_pred = grid_model.predict(X_test)
y_train_pred = grid_model.predict(X_train)

gm = train_val(y_train, y_train_pred, y_test, y_pred, "elastic_grid")
gm

pd.concat([ls,rs, rcs, lss, lcs, es, gm], axis = 1)

########### Feature importances with Ridge
ridge_cv_model.alpha_
from yellowbrick.model_selection import FeatureImportances 

model = Ridge(alpha=pipe_ridgecv["ridgecv"].alpha_)  # ridge_cv_model.alpha_ = 0.91
viz = FeatureImportances(model,labels=list(X.columns),relative=False)
viz.fit(X_train_scaled,y_train)
viz.show()

########### Feature importances with Lasso
pipe_lassocv["lassocv"].alpha_
from yellowbrick.model_selection import FeatureImportances

model = Lasso(alpha=pipe_lassocv["lassocv"].alpha_)  # lasso_cv_model.alpha_ = 0.1
viz = FeatureImportances(model,labels=list(X.columns),relative=False)
viz.fit(X_train_scaled,y_train)
viz.show()

## rcm_df = pd.DataFrame(pipe_ridgecv["ridgecv"].coef_, columns=["ridge_cv_coef_0.91"],index = X.columns)

#%% ML-6-Session
# Tasks
    # 1. Import Modules, Load Data and Data Review
    # 2. Data Pre-Processing
    # 3. Implement Linear Regression
    # 4. Implement Ridge Regression
    # 5. Implement Lasso Regression
    # 6. Implement Elastic-Net
    # 7. Visually Compare Models Performance In a Graph

########### 1. Import Modules, Load Data and Data Review
import pandas as pd      
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import skew
from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings('ignore')
plt.rcParams["figure.figsize"] = (10,6)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

df = pd.read_csv("final_scout_not_dummy.csv")
df.head()
df.info()
df.describe()
df.isnull().sum()

df.make_model
df2 = df.copy()

############ Feature Engineering
df_object = df.select_dtypes(include ="object").head()
df_object

for col in df_object:
    print(f"{col:<20}:", df[col].nunique())

df.make_model.value_counts()

ax = df.make_model.value_counts().plot(kind ="bar")
ax.bar_label(ax.containers[0]);

df[df.make_model=="Audi A2"] # Tek değer("Audi A2")

df.drop(index=[2614], inplace =True)
df.shape

sns.histplot(df.price, bins=50, kde=True)

skew(df.price)

df_numeric = df.select_dtypes(include ="number")
df_numeric

sns.heatmap(df_numeric.corr(), annot =True)

########### Multicollinearity control
df_numeric.corr()[(df_numeric.corr()>= 0.9) & (df_numeric.corr() < 1)].any().any()
df_numeric.corr()[(df_numeric.corr()<= -0.9) & (df_numeric.corr() > -1)].any().any()

sns.boxplot(df.price)

plt.figure(figsize=(16,6))
sns.boxplot(x="make_model", y="price", data=df, whis=1.5)
plt.show()

df[df["make_model"]== "Audi A1"]["price"]

total_outliers = []
for model in df.make_model.unique(): 
    car_prices = df[df["make_model"]== model]["price"]
    Q1 = car_prices.quantile(0.25)
    Q3 = car_prices.quantile(0.75)
    IQR = Q3-Q1
    lower_lim = Q1-1.5*IQR
    upper_lim = Q3+1.5*IQR
    count_of_outliers = (car_prices[(car_prices < lower_lim) | (car_prices > upper_lim)]).count()
    total_outliers.append(count_of_outliers)
    print(f" The count of outlier for {model:<15} : {count_of_outliers:<5}, \
          The rate of outliers : {(count_of_outliers/len(df[df['make_model']== model])).round(3)}")
print()    
print("Total_outliers : ",sum(total_outliers), "The rate of total outliers :", (sum(total_outliers)/len(df)).round(3))

# Her bir unique değerdeki outlier sayısı ve oranı

############## 2. Data Pre-Processing
X= df.drop(columns="price")
y= df.price

def trans_1(X, y, test_size = 0.2, random_state=101):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train = X_train.join(X_train["Comfort_Convenience"].str.get_dummies(sep = ",").add_prefix("cc_"))
    X_train = X_train.join(X_train["Entertainment_Media"].str.get_dummies(sep = ",").add_prefix("em_"))
    X_train = X_train.join(X_train["Extras"].str.get_dummies(sep = ",").add_prefix("ex_"))
    X_train = X_train.join(X_train["Safety_Security"].str.get_dummies(sep = ",").add_prefix("ss_"))
        
    X_test = X_test.join(X_test["Comfort_Convenience"].str.get_dummies(sep = ",").add_prefix("cc_"))
    X_test = X_test.join(X_test["Entertainment_Media"].str.get_dummies(sep = ",").add_prefix("em_"))
    X_test = X_test.join(X_test["Extras"].str.get_dummies(sep = ",").add_prefix("ex_"))
    X_test = X_test.join(X_test["Safety_Security"].str.get_dummies(sep = ",").add_prefix("ss_"))
    
    X_test = X_test.reindex(columns = X_train.columns, fill_value=0) # "0"
     
    X_train.drop(columns=["Comfort_Convenience","Entertainment_Media","Extras","Safety_Security"], inplace = True)
    X_test.drop(columns=["Comfort_Convenience","Entertainment_Media","Extras","Safety_Security"], inplace = True)
    
    return X_train, X_test, y_train, y_test

# Bu dataya özel bu fonksiyon kullanılacak. Bu sütunların başına prefix ekleyecek

train = {"a": [1, 2], "b": [2,3], "c":[1,4], "d":[2,4], "e":[5,6]}
test = {"e": [1, 2], "c": [2,3], "a":[1,4], "d":[2,4]}
train = pd.DataFrame(train)
test = pd.DataFrame(test)

train
test

test.reindex(columns = train.columns, fill_value=0)
X_train, X_test, y_train, y_test = trans_1(X, y)

X_train.head()
X_test.head()

################# OneHotEncoder
# Tüm dataya get_dummy yapılırsa data leakage oluyor. O yüzden ayrı ayrı get_dummy yapıyoruz
# Önce tüm dataya ait dataya yapılırsa. Test datası train datasındaki verileri(kendinde olmayan sütunları) görmüş olduğu için 
# yalancı bir iyileşme oluyor
# .. Yani sonuç olarak data leakage oluyor
# Orion Hoca: her iki tarafta aynı feature gerçek hayatta olamayabiliri. bizimde bunu simüle etmemiz gerek
# .. o yüzden direk tüm dataya herhangi bir encoding ML de uygulanmaz

# Johnson Hoca: dummies yapmak için model tüm datayı dolaşır ve tüm unique categoric verileri tepit eder. 
# .. Eger  tüm dataya dummies uygulayıp sonradan train ve test setini ayırırsanız train seti test setindeki 
# .. gözlemerde hangi categorik veriler var bilgisine sahip olur. Buda skorlarımda yalancı iyileştirmeler yapabilir.
# .. Bunun önüne geçmek için train ve test setine ayrı ayrı dummies uyguluyoruz
# .. Yani model daha önce görmediği bir feature özelliği görürse onu hesaba katmadan tahmin yapacak değil mi? --> Evet

# Kategorik sütunumuz olduğu zaman onehotencoder yapacağız
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
# sparse=False : Bir dizi dönsün istiyorum
# sparse=True  : Matris dönüyor
# handle_unknown="ignore": Birazdan açıklanacak

data =pd.DataFrame(pd.Series(['good','bad','worst','good', 'good', 'bad', 'bed'])) 
new_data= pd.DataFrame(pd.Series(['bad','worst','good', 'good', 'bad', "bed", "resume", "car"]))
# data : eski data olsun, new_data da da 2 farklı yeni gözlemim olsun(resume,car)
# new_data has two values that data does not have.
data
new_data

enc.fit_transform(data[[0]])  # Eğitim yaptığı datayı fit_transform yapıyoruz # Dizi döndü

enc.transform(new_data[[0]]) # new_datayı da transform yapıyoruz.
# Üstte OneHotEncoder içine yazılan
# handle_unknown="ignore": Eğitim yaptığım model haricinde bir değişken gördüğünde ignore ediyor ve onları 0 yapıyor
# .. Yani new_datada son 2 satırda "resume", "car" vardı. Onları 0 yaptı

enc.get_feature_names(["0"])
# "0" ön ek getirdik sütun isimlerine

pd.DataFrame(enc.fit_transform(data[[0]]), columns = enc.get_feature_names(["0"]))
# Df e çevirdik(data)

pd.DataFrame(enc.transform(new_data[[0]]), columns = enc.get_feature_names(["0"]))
# ff e çevirdik(new_data). Son indexin(resume ve car) değerleri 0 geldi

################ OneHotEncoder for X_train and X_test
# Orion Hoca
    # get dummy pandas
    # one hot encoder =sklearn.
# Class chat: one hot encoder ın get dummy den bir farkı var mı?
    # Johnson Hoca: OneHotEncoder içindeki drop hyperparametresini none yerine drop olarak değiştirirseniz
    # fazla feature ignore eder

cat = X_train.select_dtypes("object").columns
cat
# train üzerinden onehotencoder yapacağız. O yüzden traindeki kategorik sütunları çağırıyoruz

cat = list(cat) # Bunları listeye çevirelim
cat 

enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
X_train_cat = pd.DataFrame(enc.fit_transform(X_train[cat]), index = X_train.index, 
                           columns = enc.get_feature_names(cat))
# OneHotEncoder fonk. kullanalım
# enc.fit_transform(X_train[cat] : encoder tanımlayıp fit_transfotm yapıyorum
# index = X_train.index : sıra karışmasın diye train in indexsini veriyoruz

enc.fit_transform(X_train[cat])

enc.get_feature_names(cat)  # column isimleri başa geldi

X_train_cat = pd.DataFrame(enc.fit_transform(X_train[cat]), index = X_train.index, 
                           columns = enc.get_feature_names(cat))
X_train_cat
# One hot encoder yapılmış halini görüyoruz
# Orion Hoca: Onehotencoder : trainden öğrenip test e onu uygulayabiliyor. get_dummy de böyle bir şey yok. Yani;
# .. get_dummy de transform yapılamıyor

X_train.select_dtypes("number") # Nümerik featurelarımız

X_train_new = X_train_cat.join(X_train.select_dtypes("number"))
X_train_new # x_train kategorikleri ile  x_train nümeriklerini birleştirdik
# X_train datamız oluştu. Bunu test içinde yapacağız

X_test_cat = pd.DataFrame(enc.transform(X_test[cat]), index = X_test.index, columns = enc.get_feature_names(cat))
X_test_cat
# Encoder ı transform olarak çağırıyoruz burada(enc.transform(X_test[cat])
# Sonuç olarak Data leakage in önüne geçmiş olduk

X_test.select_dtypes("number") # X_test nümerik sütunlar

X_test_new = X_test_cat.join(X_test.select_dtypes("number"))
X_test_new 
# # X_test kategorik ve nümerik sütunlarını birleştiriyoruz
# Test datamızda oluştu

# Yukarda uzun uzun yaptığımız şeyi altta fonksiyona aktaralım

def trans_2(X_train, X_test):
    cat = X_train.select_dtypes("object").columns
    cat = list(cat)
    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    X_train_cat = pd.DataFrame(enc.fit_transform(X_train[cat]), index = X_train.index, 
                           columns = enc.get_feature_names(cat))
    X_test_cat  = pd.DataFrame(enc.transform(X_test[cat]), index = X_test.index, 
                               columns = enc.get_feature_names(cat))
    X_train = X_train_cat.join(X_train.select_dtypes("number"))
    X_test = X_test_cat.join(X_test.select_dtypes("number"))
    return X_train, X_test

X_train, X_test = trans_2(X_train, X_test) # Tek satırda X_train ve X_test onehotencoder uygulanmış datam döndü

X_train.head()
X_test.head()
# Datamız modellemeye hazır. Son olarak korelasyona bakalım

corr_by_price = X_train.join(y_train).corr()["price"].sort_values()[:-1]
corr_by_price

# Üstteki corr bilgisini görselleştirelim
plt.figure(figsize = (20,10))
sns.barplot(x = corr_by_price.index, y = corr_by_price)
plt.xticks(rotation=90)
plt.tight_layout();

############# 3. Implement Linear Regression
def train_val(model, X_train, y_train, X_test, y_test):
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    scores = {"train": {"R2" : r2_score(y_train, y_train_pred),
    "mae" : mean_absolute_error(y_train, y_train_pred),
    "mse" : mean_squared_error(y_train, y_train_pred),                          
    "rmse" : np.sqrt(mean_squared_error(y_train, y_train_pred))},
    
    "test": {"R2" : r2_score(y_test, y_pred),
    "mae" : mean_absolute_error(y_test, y_pred),
    "mse" : mean_squared_error(y_test, y_pred),
    "rmse" : np.sqrt(mean_squared_error(y_test, y_pred))}}
    return pd.DataFrame(scores)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

pd.options.display.float_format = '{:.3f}'.format # Sonuçlarda virgülden sonra 3 hane gelmesi için kod
# Environment settings: 
    # pd.set_option('display.float_format', lambda x: '%.4f' % x)

train_val(lm, X_train, y_train, X_test, y_test)
# Skorlar tutarlı görünüyor
# Ama 150 feature ımız var artık. O yüzden adjusted_r2 ye bakıyoruz
# Acaba skorlarımız gerçek mi değil mi diye

############# Adjusted R2 Score
def adj_r2(y_test, y_pred, X):
    r2 = r2_score(y_test, y_pred)
    n = X.shape[0]   # number of observations
    p = X.shape[1]   # number of independent variables 
    adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)
    return adj_r2

y_pred = lm.predict(X_test)
adj_r2(y_test, y_pred, X) # Karşılaştırdığımızda skorlarımız tutarlı
# Yani artan feature sayımı datam kaldırabiliyor

############# Cross Validate
model = LinearRegression() # normalize=True
scores = cross_validate(model, X_train, y_train, scoring=['r2', 
            'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv =10)
# CV yi Test skorumuzu teyit etmek için yapıyorduk...
# model = LinearRegression(normalize=True) : min, max değerlerine göre scale ediyor
# scores = cross_validate(model, X_train, y_train, scoring=['r2', 
#            'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv =10, return_train_score=True)

pd.DataFrame(scores)

pd.DataFrame(scores).iloc[:, 2:].mean()

train_val(lm, X_train, y_train, X_test, y_test) # Skorlarımız tutarlı görünüyor

2405/df.price.mean() # Hatalarımızın oransal değeri. price ı yüzde 13 hata ile tahmin edebiliyoruz

########### Prediction Error
from yellowbrick.regressor import PredictionError
from yellowbrick.features import RadViz
visualizer = RadViz(size=(720, 3000))
model = LinearRegression()
visualizer = PredictionError(model)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show();
# Silik olan çizgi(Ares hoca: mükemmel doğru): Tahmin edilen değerle gerçek değerin arasındaki fark 0
# Koyu olan çizgi: modelimin bulduğu best fit line doğrusu. Aşağı doğru çekilmiş. outlier değerlerden dolayı
# Bunları atsak skorlarımız iyileşir mi ? Aşağıda bakacağız

########## Residual Plot
plt.figure(figsize=(12,8))
residuals = y_test-y_pred

sns.scatterplot(x = y_test, y = -residuals) #-residuals
plt.axhline(y = 0, color ="r", linestyle = "--")
plt.ylabel("residuals")
plt.show()
# Hatalarımızın dağımında normal dağılım olduğunu görüyoruz genelde
# Tam anlaşılması için kdeplot a bakalım

sns.kdeplot(residuals) # Normal dağılım olduğunu görüyoruz

skew(residuals) # skew e bakarsak bu skordanda residual lerin normal dağılım sergilediğini görüyoruz

from yellowbrick.regressor import ResidualsPlot
visualizer = RadViz(size=(1000, 720))
model = LinearRegression()
visualizer = ResidualsPlot(model)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show();       
# Burada da genel anlamda normal dağılım olduğunu görüyoruz Train ve test in

########## Dropping outliers that worsen my predictions from the dataset
for model in df2.make_model.unique():
    car_prices = df2[df2["make_model"]== model]["price"]
    Q1 = car_prices.quantile(0.25)
    Q3 = car_prices.quantile(0.75)
    IQR = Q3-Q1
    lower_lim = Q1-1.5*IQR
    upper_lim = Q3+1.5*IQR
    drop_index = df2[df2["make_model"]== model][(car_prices < lower_lim) | (car_prices > upper_lim)].index
    df2.drop(index = drop_index, inplace=True)
df2

# Outlier içeren indexleri drop edelim. Bunu make_model bazında yapacağız

df2[df2.make_model=="Audi A2"] # df2 de Audi A2 görünüyordu. Onu tekrar burada drop edelim

df2.drop(index=[2614], inplace =True)
df2.reset_index(drop=True, inplace=True) # reindex yaptık
df2

df3 = df2.copy()  # df3 ü feature selection aşamasında kullanacağımız için df2 nin kopyasını bir değişkene ekledik

# Outlier drop edilmiş df ile çalışalım
X = df2.drop(columns = "price")
y = df2.price

X_train, X_test, y_train, y_test = trans_1(X, y) # Araba özelliklerini get_dummy yapmıştık
X_train, X_test = trans_2(X_train, X_test)       # trans_2 ile onehotencoder yapmış olduk

X_train.head() # Datamız modellemeye hazır
X_test.head()

lm2 = LinearRegression()
lm2.fit(X_train,y_train)

visualizer = RadViz(size=(720, 3000))
model = LinearRegression()
visualizer = PredictionError(model)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show();
# Outlierları attıktan sonra grafiğimize bakalım
# Koyu çizgi ile silik çizgi arasındaki aralık nispeten kapanmış öncekine göre

plt.figure(figsize=(12,8))
y_pred = lm2.predict(X_test)
residuals = y_test-y_pred
sns.scatterplot(x = y_test, y = -residuals) #-residuals
plt.axhline(y = 0, color ="r", linestyle = "--")
plt.ylabel("residuals")
plt.show()

train_val(lm2, X_train, y_train, X_test, y_test)
# Skorlarımız artmış oldu
# Hatamız da düşmüş

# Alttakiler eski skorlarımız
"""
     train	      test
R2	 0.890	      0.890
mae	 1705.452	  1705.217
mse	 6038122.231  5785150.711
rmse 2457.259	  2405.234
"""
2052/df2.price.mean() # Önceki hatamıza göre oransal hatamızda düştü
2405/df.price.mean() # Eski hatamız

model = LinearRegression()
scores = cross_validate(model, X_train, y_train,
                        scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv=10)

scores = pd.DataFrame(scores, index = range(1, 11))
scores.iloc[:,2:]

scores = pd.DataFrame(scores, index = range(1, 11))
scores.iloc[:,2:].mean()

train_val(lm2, X_train, y_train, X_test, y_test) # cv den sonra skorların tutarlı olduğunu görüyoruz

# Görselleştirme için değerlerimizi tanımlayalım
y_pred = lm2.predict(X_test)
lm_R2 = r2_score(y_test, y_pred)
lm_mae = mean_absolute_error(y_test, y_pred)
lm_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

my_dict = { 'Actual': y_test, 'Pred': y_pred, 'Residual': y_test-y_pred }
compare = pd.DataFrame(my_dict)

comp_sample = compare.sample(20)
comp_sample

comp_sample.plot(kind='bar',figsize=(15,9))
plt.show()

pd.DataFrame(lm2.coef_, index = X_train.columns, columns=["Coef"]).sort_values("Coef")
# Outlierların olmadığı coefficientlarımız
# Bunu daha sağlıklı yorumlamak için scaling yapılmalı

# LineerRegression ın arkasında OLS metodu
# Ridge ve Lassoda gradient descent metodu çalışıyor

# Orion Hoca:
    # bütün modellere bakmadık, daha bunun DT var RF i var boosting methodları var
    # onlard bu outlierları atmaya gerek kalmayacak
    # illa linear lasso ridge diyorsanız da datayı bölerek iki model kurabilirsiniz
    # bellii fiyatın üzerine bir model altına başka bir model
    # ya da marka bazında bir model vesaire vesaire
    # seçenek çok
    # yeterki modelin öğrenebileceği kadar sample olsun

######### 4. Implement Ridge Regression
######### Scaling
scaler = MinMaxScaler()   # dummy featurelar çok olduğu için MinMaxScaler() kullanmak mantıklı
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

######### Ridge
from sklearn.linear_model import Ridge
ridge_model = Ridge()
ridge_model.fit(X_train_scaled, y_train)
train_val(ridge_model, X_train_scaled, y_train, X_test_scaled, y_test)

######## Cross Validation
model = Ridge()
scores = cross_validate(model, X_train_scaled, y_train,
                        scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], 
                        cv=10)

scores = pd.DataFrame(scores, index = range(1, 11))
scores.iloc[:,2:].mean()
# Skorlar birbirine yakın(öncekiyle). Yani tutarlı

######## Finding best alpha for Ridge
from sklearn.model_selection import GridSearchCV # RidgeCV yerine bu gün GridSearchCV kullanalım
alpha_space = np.linspace(0.01, 100, 100)
alpha_space

ridge_model = Ridge()
param_grid = {'alpha':alpha_space}
ridge_grid_model = GridSearchCV(estimator=ridge_model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=10,
                          n_jobs = -1)
#  n_jobs = -1: işlemcileri çalıştırmak için yazılan parametre

ridge_grid_model.fit(X_train_scaled,y_train)

ridge_grid_model.best_params_ #ridge_grid_model.best_estimator_

pd.DataFrame(ridge_grid_model.cv_results_)

ridge_grid_model.best_index_

ridge_grid_model.best_score_ # minimize edilmiş RMSE

train_val(ridge_grid_model, X_train_scaled, y_train, X_test_scaled, y_test)

# Skorlarımızı alalım karşılaştırma için kullanacağız
y_pred = ridge_grid_model.predict(X_test_scaled)
rm_R2 = r2_score(y_test, y_pred)
rm_mae = mean_absolute_error(y_test, y_pred)
rm_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

ridge = Ridge(alpha=1.02).fit(X_train_scaled, y_train)
pd.DataFrame(ridge.coef_, index = X_train.columns, columns=["Coef"]).sort_values("Coef")

########## 5. Implement Lasso Regression¶
from sklearn.linear_model import Lasso
lasso_model = Lasso()
lasso_model.fit(X_train_scaled, y_train) # alpha=1

train_val(lasso_model, X_train_scaled, y_train, X_test_scaled, y_test)

######### Cross Validation
model = Lasso()
scores = cross_validate(model, X_train_scaled, y_train,
                        scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'],
                        cv=10)

scores = pd.DataFrame(scores, index = range(1, 11))
scores.iloc[:,2:].mean()
# Skorlar tutarlı

########### Finding best alpha for Lasso
lasso_model = Lasso()
param_grid = {'alpha':alpha_space}
lasso_grid_model = GridSearchCV(estimator=lasso_model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=10,
                          n_jobs = -1)

lasso_grid_model.fit(X_train_scaled,y_train)

lasso_grid_model.best_params_

lasso_grid_model.best_score_  # RMSE

train_val(lasso_grid_model, X_train_scaled, y_train, X_test_scaled, y_test)
# Default halinde alınan skorlarla aynı gibi.
# Virgülden sonra 3 basamak almıştık. Daha fazla alırsak fark olduğunu görebiliriz
# ridge lasso kullanınca;
    # train tes skorlarını yaklaştırmak
    # overfitting i çözmek
    # multicollinearity yi halletmek
# .. işlemleri hallediliyor

y_pred = lasso_grid_model.predict(X_test_scaled)
lasm_R2 = r2_score(y_test, y_pred)
lasm_mae = mean_absolute_error(y_test, y_pred)
lasm_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

lasso = Lasso(alpha=1.02).fit(X_train_scaled, y_train)
pd.DataFrame(lasso.coef_, index = X_train.columns, columns=["Coef"]).sort_values("Coef")
# Lasso bazı featureları atmış(0 yapmış katsayısını)
# Class chat soru : Peki teorik olarak belli değişkenler bizim için önemliyse lasso'nun kendi belirlediği faktörleri 0'laması sorun olmaz mı hocam? Şu değişken kalsın, multicollinearity halinde olduğu diğer değişkeni at diyebiliyor muyuz?
# Orion hoca: diyemiyoruz. o kendi karar verir.ama siz modele sokmadan datayı ayarlayabilirisiniz.herşeyide ML den beklemeyelim.o da gariban bir kütüphane

######## 6. Implement Elastic-Net
from sklearn.linear_model import ElasticNet
elastic_model = ElasticNet()
elastic_model.fit(X_train_scaled,y_train) # l1_ratio:0.5, alpha:1

train_val(elastic_model, X_train_scaled, y_train, X_test_scaled, y_test)
# underfitting durumunu söyleyebiliriz
# Skor çok düşük, hata çok yüksek
# Teyit için CV yapalım

######### Cross Validation
model = ElasticNet()
scores = cross_validate(model, X_train_scaled, y_train,
                        scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], 
                        cv=10)
# Skorlar tutarlı ama yukardaki modellerle kıyaslandığında underfitting durumu var.

scores = pd.DataFrame(scores, index = range(1, 11))
scores.iloc[:,2:].mean()

######### Finding best alpha and l1_ratio for ElasticNet
elastic_model = ElasticNet()

param_grid = {'alpha':[1.02, 2,  3, 4, 5, 7, 10, 11],
              'l1_ratio':[.5, .7, .9, .95, .99, 1]}
elastic_grid_model = GridSearchCV(estimator=elastic_model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=10,
                          n_jobs = -1)
# class chat soru: Hocam buradaki alphaları neye göre belirlediniz?
# Orion hoca: daha önceden denedi, o yüzden son halini görüyorsunuz.denemeleri burada yapmamak için

elastic_grid_model.fit(X_train_scaled,y_train)

elastic_grid_model.best_params_
# 'l1_ratio': 1 : Lasso

elastic_grid_model.best_score_

train_val(elastic_grid_model, X_train_scaled, y_train, X_test_scaled, y_test)

y_pred = elastic_grid_model.predict(X_test_scaled)
em_R2 = r2_score(y_test, y_pred)
em_mae = mean_absolute_error(y_test, y_pred)
em_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

########## Feature İmportance
from yellowbrick.model_selection import FeatureImportances
from yellowbrick.features import RadViz
model = Lasso(alpha=1.02)
viz = FeatureImportances(model, labels=X_train.columns)
visualizer = RadViz(size=(720, 3000))
viz.fit(X_train_scaled, y_train)
viz.show();

df_new = df3[["make_model", "hp_kW", "km","age", "price", "Gearing_Type", "Gears"]]
df_new
# Get_dummy yapmadığım dataframeden(df3) feature selection yapıyorum
# Bunlarla yeni model oluşturacağız
# Acaba 6 tane feature kullansak(target/label hariç) güzel bir skor alabilir miyiz diye deniyoruz
# 2 tane kategorik sütunum var. Bunlara onehotencoder yapacağız

X = df_new.drop(columns = ["price"])
y = df_new.price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

X_train, X_test = trans_2(X_train, X_test)  # Onehotencoder işlemini tek satırda yapıyoruz
# Data leakage olmasın diye ayrı ayrı yapıyoruz X_train ve X_test e
X_train
X_test

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

lasso_model = Lasso() # default haliyle çağırdık
lasso_model.fit(X_train_scaled, y_train)
train_val(lasso_model, X_train_scaled, y_train, X_test_scaled, y_test)
# Sadece 6 feature ile aldığımız skor 0.869. 141 feature ile 0.90 dı
# Amaç her zaman için az feature ile yüksek skorlar almak
# Bu skoru teyit edelim CV ile

########### Cross Validate
model = Lasso()
scores = cross_validate(model, X_train_scaled, y_train,
                        scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'],
                        cv=10)

scores = pd.DataFrame(scores, index = range(1, 11))
scores.iloc[:,2:].mean()
# Skorlar tutarlı
# Skorları iyileştirmek için gridsearch yapalım

########## Gridsearch
lasso_model = Lasso()
param_grid = {'alpha':alpha_space}
lasso_final_model = GridSearchCV(estimator=lasso_model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=10,
                          n_jobs = -1)

lasso_final_model.fit(X_train_scaled,y_train)

lasso_final_model.best_params_ # best alpha: 0.01. (Değer küçük yani çok regularization a gerek yok diyor)

lasso_final_model.best_score_

train_val(lasso_final_model, X_train_scaled, y_train, X_test_scaled, y_test)
# Skorlarda küsürat sonrası iyileşme olmuştur
# Sonuç olarak 6 feature ile yüzde 87 lik skor gayet iyi(141 feature kullanmaktansa)

2364/df_new.price.mean()

y_pred = lasso_final_model.predict(X_test_scaled)
fm_R2 = r2_score(y_test, y_pred)
fm_mae = mean_absolute_error(y_test, y_pred)
fm_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

########## 7. Visually Compare Models Performance In a Graph
# Aldığımız tüm skorları burada dictinary ye atıp df oluşturduk
scores = {"linear_m": {"r2_score": lm_R2 , 
 "mae": lm_mae, 
 "rmse": lm_rmse},

 "ridge_m": {"r2_score": rm_R2, 
 "mae": rm_mae,
 "rmse": rm_rmse},
    
 "lasso_m": {"r2_score": lasm_R2, 
 "mae": lasm_mae, 
 "rmse": lasm_rmse},

 "elastic_m": {"r2_score": em_R2, 
 "mae": em_mae, 
 "rmse": em_rmse},
         
 "final_m": {"r2_score": fm_R2, 
 "mae": fm_mae , 
 "rmse": fm_rmse}}
scores = pd.DataFrame(scores).T
scores
# Final model skoru biraz daha düşük ama 6 feature ile bu skoru aldık

for i, j in enumerate(scores):
    print(i, j)
    
#metrics = scores.columns
for i, j in enumerate(scores):
    plt.figure(i)
    if j == "r2_score":
        ascending = False
    else:
        ascending = True
    compare = scores.sort_values(by=j, ascending=ascending)
    ax = sns.barplot(x = compare[j] , y= compare.index)
    ax.bar_label(ax.containers[0], fmt="%.4f");
# Müşteriye sunarken görselleştirme kullanabiliriz
# Final modelde tahminlerde biraz fark var   
    
############################# Prediction new observation 
# Diyelim ki yeni gözlem geldi. Buna göre tahmin yapacağız diyelim    
X = df_new.drop(columns = ["price"])
y = df_new.price   
    
X.head()  
    
cat = X.select_dtypes("object").columns
cat = list(cat)
cat    
    
enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
X_cat = pd.DataFrame(enc.fit_transform(X[cat]), index = X.index,     # DIKKAT: X[cat] .. Bütün dataya one-hot encoding yapıyoruz
                           columns = enc.get_feature_names(cat))
X = X_cat.join(X)
X.drop(columns = cat, inplace=True)
X   
    
final_scaler = MinMaxScaler()
final_scaler.fit(X)
X_scaled = final_scaler.transform(X) # Final adımda tüm datamızı kullanıyoruz yine   
    
final_model = Lasso(alpha=0.01)
    
final_model.fit(X_scaled, y) # Şu an modelimizi inşa ettik. Deploy etmeye hazır    
    
# Yeni gözlem ekleyelim
my_dict = {
    "hp_kW": 66,
    "age": 2,
    "km": 17000,
    "Gears": 7,
    "make_model": 'Audi A3',
    "Gearing_Type": "Automatic"
}    
    
new_obs = pd.DataFrame([my_dict])
new_obs    
    
# Girilen gözlemlere sadece transform yapıyoruz. Çünkü datadan öğrendiğim bilgilerle
# .. burada sadece dönüşüm yapıyoruz
onehot = pd.DataFrame(enc.transform(new_obs[cat]), index=new_obs.index,
                           columns = enc.get_feature_names(cat))
new_obs = onehot.join(new_obs)
new_obs.drop(columns = cat, inplace=True)
new_obs
# Burada sıra çok önemli. Modeldeki sıra ile aynı olmalı. o yüzden altta reindex yapacağız    
    
new_obs = new_obs.reindex(columns=X.columns)
new_obs    
    
new_obs = final_scaler.transform(new_obs)  # Yeni gelen gözlemlerimi modelimizin anlaması için scaling yapıyoruz
new_obs    
    
final_model.predict(new_obs) # Tahmin alıyoruz
# Bunları 11-12 kod satırında yaptık. Şimdi pipeline a bakalım    

################# Pipeline  
X = df_new.drop(columns = ["price"])
y = df_new.price

cat = X.select_dtypes("object").columns
cat = list(cat)
cat

X.head(1)

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
column_trans = make_column_transformer((OneHotEncoder(handle_unknown="ignore", sparse=False), cat), 
                                       remainder=MinMaxScaler())
# Pipeline: Sıralı boru hattı diyebiliriz. Sırayla;;
# train ve test e ayrı ayrı onehotencoder yapıp sonra minmaxscaler yapıyor

from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
operations = [("OneHotEncoder", column_trans), ("Lasso", Lasso())]
# OneHotEncoder : Bunun yerine OneHotEncoder and scaling de yazabiliriz. Çünkü scalingde yaptı
pipe_model = Pipeline(steps=operations)
# Yapacağımız işlemleri koyuyoruz pipeline a
pipe_model.fit(X, y)  # Oluşturduğum pipeline modelimi veriyoruz ve tüm işlemler tamamlanıyor
# Modelimizi kurduk. Yeni tahminle bakalım şimdi

my_dict = {
    "hp_kW": 66,
    "age": 2,
    "km": 17000,
    "Gears": 7,
    "make_model": 'Audi A3',
    "Gearing_Type": "Automatic"  
}

new_obs = pd.DataFrame([my_dict])
new_obs

############### Cross Validate With Pipeline
# Yukardakileri tekrar ediyoruz
X = df_new.drop(columns = ["price"])
y = df_new.price

X.head()

cat = X.select_dtypes("object").columns
cat = list(cat)
cat

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

column_trans = make_column_transformer((OneHotEncoder(handle_unknown="ignore", sparse=False), cat), 
                                       remainder=MinMaxScaler())
operations = [("OneHotEncoder", column_trans), ("Lasso", Lasso())]
pipe_model = Pipeline(steps=operations)
pipe_model.fit(X_train, y_train)

train_val(pipe_model, X_train, y_train, X_test, y_test) # Skorları karşılaştırıyoruz

# Skorlarım gerçek mi diye cross validation yapıyoruz.
operations = [("OneHotEncoder", column_trans), ("Lasso", Lasso())]
pipe_model = Pipeline(steps=operations)
scores = cross_validate(pipe_model, X_train, y_train,
                        scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'],
                        cv=10)
scores = pd.DataFrame(scores, index = range(1, 11))
scores.iloc[:,2:].mean()

# Karşılaştırdığımız sonuçlar tamamen aynı 0.874(bu koddaki çıktı) e 0.874()
# class chat:hocam biz final olarak feature sayısını azaltarak tahmin yaptık ya; mesela şimdi siz örnek olarak bir dictionary üzerinden gelen verilerle ilgili en son fiyat tahmini yaptırdınız. Peki kullanıcı bizim tahmin yaparken kullandığımız feature'lardan farklı değerleri de girerse model bu sefer eksik tahmin yapmış olmaz mı?
# Orion hoca: Olur. Biz bunu yaptık bitti diye bir şey yok. Data tekrar toplanır eğitilir tekrar yeni model canlıya alınır
# .. Lassoyu bir kere seçtik sonra başka bir şey yapmayacağız diye bir şey yok.
# Kullanıcının sıklıkla girdiği yada girmek istediği feature ları modele ekleyip yeniden model tasarımı yapmamız lazım
# Johnson hoca: kurduğumuz bu pipeline işlem yaptığımız kategorik veriler haricinde veya numeric veriler haricinde farklı bir gözlem verilse bile farklı verilen gözlemler ignore edilerek tahmin verir.


#%%------------------------------------------

###############LESSON 8##########################

#Logistic Regression ( Lojistik Regresyon ) sınıflandırma 
#işlemi yapmaya yarayan bir regresyon yöntemidir.
#Kategorik veya sayısal verilerin sınıflandırılmasında
#kullanılır. Bağımlı değişkenin yani sonucun sadece kategorik veri olması durumunda çalışır. 
#( Evet / Hayır, Erkek / Kadın, Şişman / Zayıf vs. )


#Binary Classification :***,ooo
#Multi-Class Classification :***,ooo,---

#Regression deyince anladığımız continious değerdi ama 
#logistic regression classification yapar 

#log model:eksi sonsuz ile artı sonsuz arasındaki sayıları 0 ile 1 arasında 
#sıkıştırır 

# p|     . . . . .   o ->0
# a|                 . ->1
# s|o o o o o o 
# s|______________
# e 1  1  3  4 5
# d   study hour 

#Buraya linear line uygulanır ama mantıksız olur.Bu yüzden bu tür 
#verilerde logistic regression u kullanırız 

# p|    . . . . . .                          o ->0
# a|      .  ->logistig model                . ->1
# s|     .      
# s|o o o o o o 
# e|______________
# d 1  1  3  4 5
#    study hour 

#Aldığımız bütün değerler 0-1 arasında olacak yani olasılığını alacağız

#Logistic model:>Bütün değerleri 0-1 arasına getirir.Her bir değer için probability
#verir(bunu sigmoid function yapıyor)

#*Study hour u ne verirsek verelim sigmoid function bize 0 ile 1 arasında bi 
#değer döndürür 

#Logistic Regression:
#    *Her feature için coef değerlerini bulur 
#    *Sigmoid function u uygular,değeri 0-1 arasına sokar.Bize bir 
#    probability verir(örk:2 saat ders çalışanın geçme ihtimali %61)

#*bizim belirleyeceğimiz treshold a göre geçti kaldı der 
#treshold: >0.50 geçti,<0.5 kaldı gibi 

#SIGMOID FUNCTION:
    
#P(z)=1/(1+e^(-z)) 

#Probability bulmak için coef lere ihtiyacımız var.Bu yüzde yine Linear Regression ı 
#kullanmak zorundayız 

#income=bo+b1x
#  |
#p(income>4000)=1/(1+e^-(bo+b1x))
#     |                     |
#değeri 400 den           Linear Reg
#yüksek olanların 
#olasılığı 

#z=bo+b1x  -->Böylece bir intercept ve slope üzerinden çalışmış oluyoruz.Böylece 
#elimizde her bir değer için çarpacağımız bir değer(b1);sıfır noktası için 
#sabit bir değer(bo) oluyor,böylece Linear Regression ile çalışabiliyoruz 
#Linear regression dan Logistic regression a geçeceğiz 


#Linear Model:bo+b1x
#Logistic Model:P(z)=1/(1+e^(-z)) 

#*önce değerlerimizi probabilty(0-1 arası) çeviriyoruz 
#*1 gözlelerini + sonsuz a 
#*0 gözlelerini - sonsuz a koyuyoruz 

#*Residual lar sonsuz olduğu için residualları bulmak mümkün değil 
#*yani burada best fit line çizmek mümkün değil 
#*Bu yüzden rasthgele bir çizgi çiziyoruz ve iz düşülerin y 
#eksenindeki ln karşılığını alıyoruz 

#*ln karşılıkları hesaplanarak probability ler hesaplanır.

#p=e^ln(odds)/(1+e^ln(odds))
#*0-1 üzerindeki değerler lojistik model üzerine taşındı 

#*Likelihood:1 classına ait olma olasılığı.Bunu maximize etmek istiyoruz ki başarımız artsun 
#MLE:Maximum Likehood Estimation 

#Treshold ile belirlenen değerin üstündeki noktaların y eksenindeki değeri aynen alınır 
#Çizginin altındakiler diğer classa ait olduğu için 1 den çıkarılır 

#Likelihood değeri 1 e ne kadar yakın olursa model o kadar başarılı demek 
#Bunun anlamı :Linear Reg. ile logistic reg. çizgilerinin kesiştikleri noktanın 
#üstündeki değerlerin büyümesini,altındaki değerlerin küçülmesini istiyoruz ki 
#Likehood büyüsün 

#*Likelihood da değerlerimizi artırmak istiyoruz ki başarımız artsın.Ama eliizde 
#"Gradien descent algoritması" var.Bu algoritma değerleri minimize etmeye çalışır.
#Likehood da da değerleri maximize etmeye çalıştığımız için bu algoritma  Likelihood 
#için kullanılmaz 

#*Bu yüzden "cost funtion(log loss)" fonksiyonu oluşturulmuş.
#   Cost function minimize edilince Likehood maximize edilmiş olur 
#   Cost function her zaman negatiftir.Onu minimize edince en yüksek skor elde edilmiş olur 
   
#*Log Loss Function nasıl değişir?
#Gradien descent algoritması yeni bir line çizer.Bu line a göre logistic line da değişir 
#Böylece izdüşümler de(y eksenine karşılık gelen nokta )değişir.Böylece likelihood değerleri 
#de değişir 

#ÖZETLE:
    
#    *0-1 noktalarını probability e çevir(0-1 arasına)
#    *Değerleri log ile artı sonsuz ile eksi sonsuz arasına taşı 
#    *araya bir line çiz
#    *noktaların izdüşümlerini bul 
#    *0 ile 1 classının izdüşülerini tekrar probability e çevir 
#    *yani 0 ve 1 olan değerleri logistic line üzerine taşıdık 
#    *Line değerini iyileştirmek için Likelihood u maximize etmek
#    gerekir fakat Gradien descent algoritması minimize etmek üzere 
#    çalıştığı için "log loss(cost function)" u kullan 

#Regressin(Hava sıcaklığı ne olacak?):Sıcaklık tahmini yapılıp regression hesabı yapılır
#Classification(Hava sıcak mı soğuk mu):Treshold belirlenir(20 C üstü sıcak,altı soğuk gibi)

#Classification Error Metrics

#1)Confusion Matrix:
        
#             Predict Class 
#          +           -
#Actual + TP          FN     
#Class  - FP          TN 

#True_Positive:Gerçekte (+),doğru tahmin.Actual:1,Predict:1
#False_Positive:Gerçekte(-),yanlış tahmin,Actual:0,Predicted:1
#True_Negative:Gerçekte(-),Dogru tahmin,Actual:0,Predicted:0
#False_Negative :Gerçekte(+),yanlış tahmin,Actual:1,Predicted:0

#Confusion Matrix üzerinden tahmin ettiklerimiz:
#    1-Accuracy
#    2-Recall 
#    3-Precision
#    5-F1_Score
#->classları düzgün ayırabildi mi?ayıramadı mı?

#1)ACCURACY:
#    
#Actual    :1 1 0 1 0 0 0 1 1 0
#Predicted :0 1 1 1 0 0 0 1 1 1
#           x + x + + + + + + x

#Accuracy=(Doğru bilinen tahminler)/(tüm tahminler)
#=7/10*100=%70
#
#Accuracy çok tercih edilmez.Nedeni:
#    Mesela 63 tane hasta var.hastalardan 5 i kanser 58 i sağlıklı 
#    tahmin->60 doğru 3 yanlış 
#    accuracy=60/63*100=%95
    
# Datadaki kanser hastalarının sayısı çok az iken,sağlıklı bireylerin 
#sayısı çok fazla.Yani accuracy nin başarısı daha çok sağlıklı 
#bireyler üzerinden geliyor.Aa biz 5 kanser hastamızı tespit etmek istiyoruz 
# Yani datadaki classlardaki observation sayıları arasında çok fark varsa Accuracy
#bizi yanıltır.
#Yani kanser olan bireyler üzerinde iyi bir tahmin yapıyor diyemeyiz 
 
#2)RECALL:
#    Gerçekten kanser olan hastaların kaçını bildim?

#Actual    :1 1 0 0 1 0 0 1 0 0
#Predicted :0 1 1 0 1 0 0 0 1 0
#           x + x + + + + x x +
#gerçekte kanser olan hastalar 4 kişi,doğru tahmin 2 kişi 

#recall=TP/(TP+FN)=2/(2+2)=%50 sensitivity

#3-PRECISION:
#    Sadece pozitif tahminlere odaklanıyoruz.
#    "Yaptığımız pozitif tahminlerin ne kadarı doğru"
    
#Actual    :1 1 0 0 1 0 0 0 0 1
#Predicted :0 1 1 0 1 0 0 0 0 0
#           x + x + + + + + + -
#pozitif tahmin edilen 3 kişi,gerçekten pozitif olan 2 kişi

#Precision:TP/(TP+FP)=2/3=%67

#4-SPECIFITY:(Recall in tersi)
#   Kanser hastasi olmayanlarin kacini bildim?
#   
#   Actual    :1 1 0 0 1 0 0 0 0 1
#   Predicted :0 1 1 0 1 0 0 0 0 0
#              x + x + + + + + + -
#              
#    TN->5 ,FP->1 
#    Specifity=TN/(TN+FP)=5/(5+1)=%98
    
#Recall:Kanser hastalarina odaklandi(1 e bakar)
#Specifity:Kanser olmayanlara odaklandi (0 a bakar)

#5-F1 SCORE:
#    Precision ve Recall in harmonik ortalamasidir.Harmonik ort. alinir
#cunku eger esit dagilmayan bir veri seti varsa,sonuclardan biri 0 cikarsa 
#F1 score da 0 cikar ve modelin basarisizligi ortaya cikmis olur.
#Unbalance durumda yaniltici sonuc vermez 

#F1=2*(Precision*Recall)/(Precision+Recall)

# Actual    :1 1 0 0 1 0 0 0 0 1
# Predicted :0 1 1 0 1 0 0 0 0 0
#            x + x + + + + + + -

#Precision=TP/(TP+FP)=2/3
#Recall=TP/(TP+FN)=2/4 

#F1=%57

#F1-Score formulune gore Recall ve Precision ters orantilidir.Bu yuzden F1-Scoru 
#arttirmak icin hangisi iyiyse o yukari cekilir,digeride duser

#Model Karsilastirma Yontemleri:
    
#    1)ROC/AUC (Balanced datalarda)
#    2)Precision-Recall Curve (Imbalance datalarda)
    
#Receiver Over Characteristics->ROC
#Area Under Curve->AUC 

#ROC/AUC
#Model gucunu degerlendirmek icin kullanilir 
#True Positive Rate yuksek,False Positive Rate dusukse iyi tahmin yapar

#True Positive Rate(Sensitivity)=TP/(TP+TN)
#False Positive Rate(1-Specifity)=FP/(FP+TN)

#*Olusturdugumuz modellerin hangisi guclu?Bunun icin ROC/AUC kullanilir

#ROC/AUC Nasil Calisir?

#Treshold degeri defauld:0.5 
#Cesitli treshold degerlerini dener.Mesela(TH=0,TH=0.2,TH=0.8,TH=1)
#Treshold degerlerine gore yeni logistic line i cizer ve ona gore 
#TP,FP,FN,TN degerlerini hesaplar.

#Buldugu degerleri formulde yerine koyar ve grafigi cizer.Altta kalan alan 
#ne kadar buyukse model o kadar basarilidir.

#Imbalance datalarda(class sayilari uyumsuz),ROC/AUC a bakilmaz.
#Precision_Recall Curve bakilir 

#Precision-Recall Curve
#*ROC/AUC den farki,sag ust koseye dogru alan arttikca model iyilesiyor demektir

#Precision=TP/(TP+FP)
#Recall=TP/(TP+FN) 

#%%%

######################NOTEBOOK 8################################# 

# Supervised learningte 2 tane alan vardı--> Regression ve Classification
# Teknik sınavlarda
# Aşağıdakilerden hangisi bir classification yöntemidir diyor
# Regression deyince anladığımız continious değerdi ama burada 
# .. logistic regression classification yapar
# Bir çok yöntemin hem regression hem classification için kullanımı vardır
# Logistic i sadece classification için kullanıyoruz

# Regression: Datalarımız var(x1,x2,x3) ve bu katsayılarımızla çarpılıp best fit line bulunuyordur regressionda
# Classification: Datalarımız var(x1,x2,x3) ve bu katsayılarımızla çarpılıp sonuçlar discrete ya da categorical olarak çıkıyor
# .. (Good-bad , 0-1,)
# Mesela yarın dondurma satışı iyi mi olacak kötü mü olacak
# Modele diyoruz ki eğer şu değerler arasında olursa good, şu değerler arasında olursa "bad" de diyoruz
# Buna göre sınıflama yapıyor

# 2 sınıflama yapabiliriz.Buna binary classification deriz(good-bad, 0-1)
# .. Model 2 boyutlu ise datayı tek boyutlu bir doğruyla ayırabiliyorduk. 3 boyutlu ise düzlemle vs ayırabilirz
# Bir de multi class olan var. Bu şekilde de sınıflandırma yapılabilir(iyi,kötü,çok iyi, çok kötü --> Mesela 4 sınıf)
# Nerelerde kullanılır classification
    # Klinik uygulamalarında(Hasta mı değil mi, Diabet mi değil mi gibi...)
    # Sahtekarlık tespiti
    # Devam eden müşteri, ayrılan müşteri

#Çalışma saatlerine göre aldığı skor. Bu normalde regression problemi
#Biz bu datadan çıkarım yapmak istiyoruz. Bu çok temel bir problem. 
#Feature lar normalde yeterli değil

# Sınıflama probleminde ise çalışma saatine göre geçti kaldı var burada(1:geçti, 0:kaldı)
# Logistic regressionda değerleri 0 ve 1 olarak sınıflandırmış olduk
# Bunların olasılıkları var aslında. Olasılık 0.5 üstünde ise 1 e atıyor. Altında ise 0 a atıyor

# Bunu nasıl yapıyoruz
# Bunu yapmak için sigmoid fonksiyonu kullanıyoruz
# Sigmoid fonksiyon 0 ile 1 arasında değişiyor
# Sigmoid ile datalarımı 0 ve 1 e çekmiş oluyorum. Logistic regression da eksi sonsuz artı sonsuz aralığı vardı
# Eşik değer 0.5 in üstünde ise 1 , altındaysa 0 diyoruz

# Bazen stersek biz olasılıklarıda kullanabiliyoruz(Proba)
# Bazen de 0 ve 1 şeklinde görmek istiyoruz

# Normalde biz featureları kullanarak regressionda yapabilirim ama biz bunu sigmoid e uydurmak istiyoruz
# Sigmoid de fit ettiğimizde bir p olasılığı buluyoruz
# IN(odds) formülünde kullanıyoruz bu p yi
# Sonra sonuç olarak sigmoid e fit edince ve "odd" ları kullanarak değerlerimi 0 ve 1 e çekiyoruz
# Not: In = log e

# Örneğin 0.5 değeri için In(0.5/1-0.5) = 0 geliyor
# p--> 1  e giderken fonksiyon sonsuza gidiyor
# p--> 0  a giderken fonksiyon eksi sonsuza gidiyor
# Not: In = log e

# Bir çok farklı sigmoid olabilir(Alttaki grafikteki turuncu, sarı, kırmızı çizgiler)
# Hangi sigmoid i buna fit edeceğini max likelihood ile belirliyoruz
# Maximum likelihood estimation(MLE): Olasılıkların çarpımı(Yani mavi noktaların çapımı ve bunların target olma olasılığı, kırmızıların çarpımı ile target olma olasılığı)
# Böylelikle likelihood u maximize ediyor ve en uygun sigmoid i elde etmeye çalışıyoruz
# Cost(Hata/Maliyet) function ımızda alttaki J(x) fonksiyonu
# John Hoca: Mülakatlarda bu bilgiler işe yarayabiliyor

# class chat soru: hocam olasılıkları çarmanın mantığını bir daha anlatabilirmisiniz?
# John HocaTarget genelde 1 kabul edilir. Burada 1 olma olasılıklarını çarpıyoruz(mavi noktalar)
# .. kırmızı noktaları da 1-p olarak bulabiliriz. Bunu maximize etmeye çalışıyorum
# .. böylelikle en uygun sigmoid bulunmuş oluyor ve modelim öğrenmiş oluyor

# Class chatten: sigmoid in değerleri [0,1] aralığına dönüştürmesini aşağıdaki kod üzerinden görebiliriz
"""
import numpy
x= list(map(int,input().split(",")))
x = numpy.asarray(x)
x
y = 1 / (1 + np.exp(-x))
y
"""
# Regression da r2,mae,mse,rmse metriklerimiz vardı
# Classificationda başka metiklerimiz var modeli değerlendirmek(evaluate) için
# Normalde 50-60 metrik var burada 3-5 tane kullanacağız temel olarak

# Error metriklerini tanımlamak için 4 tane terimimiz var. Bunları çokça kullanacağız
# Bu metriklere göre modelimizi değerlendireceğiz

# TP: True Pozitif
# TN: True Negatif
# FP: False Pozitif
# FN: False Negatif
# GT: Grand Truth : Gerçek değerlerim/Doğrulama verisi

# Target ımız 1 dir. Non-target ımız 0 dır deriz genelde. Bunlar şu şekilde adlandırılabilir
    # T: Target , NT: Non-Target
    # P: Pozitif , N: Negatif
    # Target her zaman predict etmeye çalıştığımız şey
    
# Model 2 şey söyler bana test aşamasında
    # Pozitif --> 1 
    # Negatif --> 0

# Grand truth da bana 2 şey söyleyebilir
    # True  --> 1
    # False --> 0
    
# Sonuç olarak, modelin söylediği sonuçları değerlendirerek tanımlamalar yapıyoruz

# TP: True Pozitif   : 1-1 (Gerçek değeri 1 iken 1 olarak tahmin etmek)
# TN: True Negatif   : 0-0 (Gerçek değeri 0 iken 0 olarak tahmin etmek)
# FP: False Pozitif  : 0-1 (Gerçek değeri 0 iken 1 olarak tahmin etmek)
# FN: False Negatif  : 1-0 (Gerçek değeri 1 iken 0 olarak tahmin etmek)
# GT: Grand Truth : Gerçek değerlerim/Doğrulama verisi

# class chat soru: pozitif dedigimiz sey predict etmeye calistigimiz deger oluyor degil mi
# Orion Hoca: genelde yakalamaya çalıştığımız değer

# Bütün classification larda bunu kullanacağız
# Confusion matrix ile bu sonuçları değerlendireceğiz

# Örnek:
# Mesela target ım koltuklar
# Model buna(siyah şekle) koltuk demiş ise --> ...
# Sağdaki Koltuğa koltuk demişte --> ...
# Koltuğa- çiçek demişse -->  ...

# Python ın ürettiği çıktı alttaki gibidir
# Google da bunun transpoz halini görebilirsiniz. Bir fark yok
# Confusion matrixe gerçek değerler ve tahmin edilen değerleri verince bize
# .. bir matris oluşturacak sağ alttak gibi

# Classification sonucunda bir confusion matrix oluşturuyoruz
# Sonra biz bu 4 tane metriği değerlendireceğiz

# Accuracy: Doğruluk : Accuracy = All true / All prediction
# Doğru olarak predict edilen değerler(1 i 1 diye, 0 ı 0 diye tahmin edilenler) / Toplam prediction sayısı

# Bazı metriklerde accuracy, bazen recall bazen precision önemlidir
# Benim için önemli olan gerçekte 1 olan değeri doğru tahmin etmektir(Kanser olana kanser değil demişim) (Recall)
# Ya da kanser olmayan birine kanserdir demişim bir sürü tetkik sonucunda masraf yapmışım vs vs (Precision)

# Recall: Hassasiyet : 
# Kaç tane kanserli hastayı(gerçekte 1 olan) doğru bilmişim(1 olarak tahmin etmişim) / Bütün kanserliler(gerçekte 1 olanlar)
# Hastalık meselesinde çok daha hayati olduğu için recall burada precision a göre daha önemlidir
# .. Yani precision için masraf yaptık vs dedik ama bundan ziyade insan hayatını etkileyecek
# .. bir şey benim için daha önemli(recall daha önemli burada)
# Problemlerde recall u maximize et yada accuracy yi maximize et  vs diye seçeceğiz

# Precision: Kesinlik
# Precision ı yüksek yaparsam TP sayım artar. Ancak False larında sayısı artar
# Precision, modelimin "tahmin gücünü" gösterir. # Recall target yakalama oranı
# Tahminlerin hepsine 1 diyelim madem target ı yüksek oranda 1 diye tahmin edeyim ama
# .. o da sıkıntı olacaktır burada. O yüzden FP sayısı artacaktı bu da istediğimiz bir şey değil

# Michael Gd-Tech.Coord
    # Recall=Target yakalama oranı
    # Precision=Target tahmin isabet oranı

# Precision ve recall un harmonik ortalaması
# Precision ve recall un yüksek olmasını istersek F1 i kullanabiliriz
# Problem için 2 si de eşit ağırlıkta önemli ise F1 i kullanabiliriz
# Michael Gd-Tech.Coord; "Unbalanced datalarda genelde yüksek sayıdaki label a sahip sınıf tercih edilecek 
# .. şekilde model tahmin etmeye meyillidir. Bu gibi durumlarda F1 score tercih edilir."

# Specifity: Çok kullanmayacağız. John Hoca: Precision ın tam tersi

#ROC curve eğrisi: Receiver Operator Characteristics
    # Eğriye verdiğimiz isim
    # True positive Rate(Sensivity)
    # False Positive Rate(1-specifity)
    # En ideali kırmızı olarak çizilen(y ekseni ve x ekseni üzerinden geçen) yer ve AUC =1 çıkar
    # NOT: Sadece binary classification da kullanılır
# AUC             : Area under curve : Eğrinin altında kalan alan
    # AUC elde etmek için ROC kullanıyoruz
    
# Class chat ten : best treshold'u bulmak için ROC, best model'i bulmak için AUC kullanılır

# Modelin her bir threshold a göre sınıflara atama değişiyor
# Her bir threshold için değerler bulunur. Farklı threshold için farklı değerler bulunur
# John Hoca: Kısaca bu kadar bilmemiz yeterli. Uygulamada bahsedeceğiz

# Precision - Recall eğrisi unbalanced datalarda kullanılır
# Unbalanced data: Dataların oranları 1 e 10, 1 100 olması gibi
# Mesela gerçek değer 97 tane 1 tane olsun 3 tane 0 değer olsun
# Tahmin yaparken ;
# Model hepsine 1 verse başarı %97 olur. Ama başarılı diyemeyiz(Önce data balance hali getirilebilir vs)



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
#%matplotlib notebook
plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_csv('hearing_test.csv') #data&resources/

df.head()

#     age  physical_score      test_result
#0  33.00           40.70            1
#1  50.00           37.20            1
#2  52.00           24.70            0
#3  56.00           31.00            0
#4  35.00           42.90            1

#Exploratory Data Analysis and Visualization

df.info()

df.describe().T 
#                 count  mean   std   min   25%   50%   75%   max
#age            5000.00 51.61 11.29 18.00 43.00 51.00 60.00 90.00
#physical_score 5000.00 32.76  8.17 -0.00 26.70 35.30 38.90 50.00
#test_result    5000.00  0.60  0.49  0.00  0.00  1.00  1.00  1.00

df['test_result'].value_counts()
#1    3000
#0    2000
#Name: test_result, dtype: int64
# 1 ve 0'lar arasında balance durumuna baktık, burada çok belirgin bir fark yok.
ax = sns.countplot(df['test_result'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + 0.3, p.get_height() * 1.03));

# Data balanced mı değil mi diye bakıyoruz. Burada büyük bir problem yok gibi

sns.boxplot(x='test_result',y='age',data=df);
# Age'e göre boxplot'a baktığımızda; gençlerde testi geçme oranının yüksek olduğunu, yaş ilerledikçe bu sayının düştüğünü görüyoruz.

sns.boxplot(x='test_result',y='physical_score',data=df)
# Fiziksel skoru yüksek olanların çoğu testi geçmiş. Fiziksel skoru 30'un altında olanlar genel olarak geçememiş.

sns.scatterplot(x='age', y='physical_score', data=df, hue='test_result', alpha=0.5);
# Kırmızı ile mavi kesişen yerlerden bir yerlerden ayıracağız
# Turuncu renkli olanlar(gençler) genel olarak testi geçerken, mavi olanlar geçememiş. 
# Görsele bakarak, fiziksel skorun age'e göre çok daha belirleyici bir sütun olduğunu anlıyoruz.

sns.pairplot(df,hue='test_result');

sns.heatmap(df.corr(), annot=True);
# Yaş azaldıkça testi geçme oranı artmış. 
# Fiziksel skor arttıkça testi geçme oranı artmış.
sns.scatterplot(x='physical_score',y='test_result',data=df);
# Testi geçenler 20-50 arasında;
# Testi geçemeyenler 0-50 arasında.



#Train | Test Split and Scaling
#test_result target label. Buna göre split ve scale işlemlerini uyguladık.


X = df.drop('test_result',axis=1)
y = df['test_result']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
#Datanın %90' ını train için, %10'unu test için ayırdık
X_train.shape
#(4500, 2)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train) # train e fit_transform
X_test_scaled = scaler.transform(X_test)       # test e sadece transform
#X_test_scaler'a fit islemi uygulanmaz. Cunku modelimiz X_train'deki bilgilerden kopye cekebilir.(Data leakage)

#Modelling
#Logistic Regression da bir linear modeldir. Sayısal verileri olasılık üzerinden 
#bir classification'a çevirir. Arka planda Linear Regression çalışır ama Linear
#Regression da alamadığımız verileri bir trick ile olasılıklara dönüştürür.
#Yani -∞ ile +∞ arasındaki sayıları 0 ile 1 arasına sıkıştırır.
#Default değeri 0.5'tir. 0.5 üzerini 1, 0.5 altını 0 olarak kabul eder
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
#Linear Regression haricinde bütün modelllerde arka planda Ridge ve Lasso 
#default olarak regularization işlemini yaparlar. Yukarıda LogisticRegression()
#içinde penalty= 'l2' default olarak tanımlı. Bu da Ridge yöntemi demek oluyor. 
#Biz onun yerine 'l1' de yazıp Lasso da kullanabiliriz. Fakat gerekli olmadıkça 
#default değerlere dokunmamak gerekir. Kaynaklarda da cezalandırma
#parametrelerine dokunulması tavsiye edilmez.
#Linear modelde regularization yapan alpha değeri yerine burada C değeri var.
#Default değeri 'C=1.0' olarak tanımlanmış. C değeri, Ridge ve Lassodaki alpha
#değeri ile ters mantıkta çalışır. Linear modelde alpha büyüdükçe uyguladığı 
#regularization katsayısı büyüyor. Burda C değeri küçüldükçe uyguladığı regularization katsayısı büyür.

#Linear Regression'da olduğu gibi multicollinearity veya feature selection sorunları
#Logistic Regression'da yok. Çünkü modelin arkasında bunları 
#default olarak yapan parametreler var.

log_model.fit(X_train_scaled, y_train)  # Eğitim yapıyoruz

df.head()
#    age  physical_score  test_result
#0 33.00           40.70            1
#1 50.00           37.20            1
#2 52.00           24.70            0
#3 56.00           31.00            0
#4 35.00           42.90            1

log_model.coef_   
# array([[-0.94953524,  3.45991194]])
#coef_'deki ilk katsayı age sütununun, ikincisi ise physical_score'un katsayısıdır.
#Linear regression'da şöyle bir yorum yapabiliyorduk : ''age' deki bir birim artış, 
#target sütunda -0.94' l,ük bir artışa sebep olmuş.''
#Logistic Regression'da böyle bir yorum yapamıyoruz. 
#Sadece şunu söyleyebiliyoruz : ''physical_score katsayısı
#age'in katsayısından daha büyük.'' (Mutlak değerlere gore degerlendiriyoruz.)
#Yani; dinleme testinden geçip geçmemede, physical_score age'den daha etkili.

#İlk değerin negatif olması, arada ters bir ilişki olduğunu, pozitif olması 
#doğru orantılı bir ilişki olduğunu gösterir. Yaş arttıkça testten geçme 
#oranı düşer; physical_score arttıkça testten geçme oranı artar.

y_pred = log_model.predict(X_test_scaled) # Modelimiz öğrendi ve bir prediction yapıyoruz burada
y_pred

y_pred_proba = log_model.predict_proba(X_test_scaled)
y_pred_proba
# [9.89194168e-01, 1.08058325e-02],
# [1.90768955e-03, 9.98092310e-01],
# [9.75012619e-01, 2.49873806e-02],
# [9.89652504e-01, 1.03474957e-02],
# [7.40226674e-02, 9.25977333e-01],
# [1.70943342e-02, 9.82905666e-01],
#predict_proba'da çıkan değerlerin ilki ---> 0 sınıfına ait olma olasılığı,
#ikincisi ---> 1 sınıfına ait olma olasılığı
#Logistic Regression bu predict_proba değerlerini kullanarak 
#prediction'ları yapar. 0.5'in üzerinde olanları 1 sınıfına atar, altında olanları 0 sınıfına atar.
#0 noktasında >0.5 ise 0'a atar. 1 noktasında >0.5 ise 1 noktasına atar.
#Çünkü bunlar doğru tahmin edildiği anlamına gelir.


#X_test + y_yest + y_pred + y_pred_proba
test_data = pd.concat([X_test, y_test], axis=1)
test_data.head() # Bunlar gerçek sonuçlarımız

#       age  physical_score  test_result
#1718 39.00           37.80            1
#2511 45.00           38.70            1
#345  56.00           21.80            0
#2521 40.00           44.00            1
#54   64.00           25.40            0

test_data["pred"] = y_pred
test_data.head()
#       age  physical_score  test_result  pred
#1718 39.00           37.80            1     1
#2511 45.00           38.70            1     1
#345  56.00           21.80            0     0
#2521 40.00           44.00            1     1
#54   64.00           25.40            0     0
test_data["pred_proba"] = y_pred_proba[:,1] #1 e ait olma olasiliklarini cektik ve bu olasiklara gore tahminleri gorduk(pred)
test_data.head()
#       age  physical_score  test_result  pred_proba  pred
#1718 39.00           37.80            1        0.98     1
#2511 45.00           38.70            1        0.97     1
#345  56.00           21.80            0        0.01     0
#2521 40.00           44.00            1        1.00     1
#54   64.00           25.40            0        0.02     0
#Karşımıza çıkan dataların çoğu binary ağırlıkta olacak (Hasta-hasta değil gibi). 
#Bu yüzden hedef class her zaman 1 olmalı. Cross validation, GridSearch gibi çalışacağımız 
#metrikler 1 class'ını tespit etmek üzerine çalışır. Çünkü datalarda hedef genelde 1'dir. 
#Hasta=1, hasta değil=0 gibi. Modele en başta hedef class neyse verilmelidir
#ki her seferinde uğraşmak zorunda kalmayalım
#test_data.sample(10)
# Tutarlı görünüyor ama tüm data için bakmalıyız buna.
# Metriklere ve Confusion matrix e bakacağız

#       age  physical_score  test_result  pred
#257  46.00           38.30            1     1
#4139 56.00           19.00            0     0
#4073 64.00           25.30            0     0
#2276 40.00           39.70            1     1
#244  57.00           33.10            1     1
#2588 41.00           41.90            1     1
#1924 47.00           41.00            1     1
#4675 34.00           41.90            1     1
#3321 75.00           16.70            0     0
#2941 28.00           40.40            1     1
#Yukarıda aynı satırda test_result ile pred değerlerinde farklı değerler görebiliyoruz.
#Bunlar, modelin yanlış tahmin yaptığı kısımlardır.


#Model Performance on Classification Tasks
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
#hedef class in 1 secilmesine dikkat edilmelidir 
accuracy_score(y_test,y_pred)#(Doğru bilinen tahminler)/(tüm tahminler)
#0.93
log_model.score(X_test_scaled, y_test)
#0.93
# Yukarıdaki kod ile aynı sonucu döndürür. Ama bu kod exra bir işlem daha yapar.
# Arka planda x_test_scaled'i predict eder ve elde ettiği y_pred ile y_test'i karşılaştırır.
precision_score(y_test, y_pred)
#0.9331210191082803
#PRECISION (Kaç pozitif tahmin yaptım, tahminlerimin kaçını bildim?) :
#precision_score içinde default olarak pos_label=1 var. Yani aldığımız skor 1 class'ına ait skordur.
recall_score(y_test, y_pred)
#0.9543973941368078
#RECALL (Doğru olanların kaçını bildim?) :
f1_score(y_test,y_pred)
#0.9436392914653785

#Bu skorların hepsini ayrı ayrı hesaplamaya gerek yok. Hepsini birlikte kullanabileceğimiz fonksiyonlar var:
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix

confusion_matrix(y_test,y_pred)
#array([[172,  21],
#      [ 14, 293]], dtype=int64)

# 172 : 0-0 : True negatif
# 21  : 0-1 : False pozitif
# 14  : 1-0 : False negatif
# 293 : 1-1 : True pozitif
# Genelde karşımıza çıkan böyledir ancak anşılması için grafik çizdirebiliriz

#(P/N) OLARAK (T/F) TAHMIN ETTKILERIM 
#ork:hasta olarak dogru tahmin ettiklerim(TP)
#ork:hasta degil olarak yanlis tahmin ettiklerim(FN)
plot_confusion_matrix(log_model, X_test_scaled, y_test);

plot_confusion_matrix(log_model, X_test_scaled, y_test, normalize='pred');
# Bunların oransal olarak gösterimi aşağıdaki gibi
plot_confusion_matrix(log_model, X_test_scaled, y_test, normalize='true');


print(classification_report(y_test,y_pred))
# classification_report, gerçek değerler ile tahmin edilen değerleri verince bize bir rapor çıkartıyor
# Target önemli benim için precision: 0.93, recall:0.95, f1:0.94. Değerler iyi gibi(Bu degerlerin birbirlerine yakin olmasi beklenir )
#Eger precision cok dusuk recall cok yuksekse model cok tahmin yapiyor(salliyor) demektir 
#Eger precision cok yuksek recall dusukse bu seferde az tahmin yapiyor demektir 
#support(gozlem sayisi) a yakin tahmin yapmasi beklenir 
# Overfitting olabilir. Bir kontral yapacağız train datası ile altta
# Recall    : 0.95 : Kanserli hastaların çoğunu bulmuş
# Precision : 0.93 :
# Macro avg : değerlerin ortalaması örneğin recall : 0.92  = (0.89 + 0.95) / 2 (unbalance datalarda dikkate alınır,data dengesizse)
# Weigted avg: ağırlıklı ortalama örneğin precision :0.93  = (193*0.92 + 307*0.93) / 500 
#(unbalance olmayan datalarda dikkate alınır-data dengeli fakat çoğunluk olan sınıfı dikkate alınması istendiği zaman)
# accuracy:Data dengeliyse dikkate alınabilir
#              precision    recall  f1-score   support
#
#           0       0.92      0.89      0.91       193
#           1       0.93      0.95      0.94       307
#
#    accuracy                           0.93       500
#   macro avg       0.93      0.92      0.93       500
#weighted avg       0.93      0.93      0.93       500



#ork>recall:1 precision:0.1 yani hasta olan yuz kisiyi bilmis fakat bunun icin 1000 tahmin yapmis,hastane hasta olmayan 900 kisiyede tedavi uygulayacak(zarar)
#precision:1 recall:0.1 yani 10 tahmin yapiti 10 u nuda bildi fakat 90 hastayi kacirci.bu iki durumda istenilmez
#recall ve precision degerlerinin yakin olmasi beklenir

#'macro avg' ve 'weight' skorlari cok sinifli modellerde dikkate alinir.
#macro : Datada unbalance durum oldugunda bakilir. 0 ve 1 degerlerini toplar ikiye boler. 
#Mesela precision icin : (0.92 + 0.93 / 2 = 0.93)

#weight : unbalance olmayan datalarda dikkate alinir. Buyuk sinifa agirlik verir. 
#Precision icin : [(0'in gozlem sayisi 0'in olasiligi) + (1'in gozlem sayisi 1'in olasiligi)] / Toplam gozlem sayisi

#Datada butun skorlar birbirine yakinsa yani dengeliyse accuracy (micro score) kullanilir.
#Cogunluk skoru dikkate almak icin weight, data dengesizse macro score kullanilabilir.
#Her zaman modelllerdeki 1 skorlarını artıramaya çalışacağız. 
#Çünkü amacımız mesela hasta tespit edebilmek. Datamızda balance bir durum olduğu 
#için 0 ve 1 değerlerinin tahminleri iyi çıktı ama unbalance 
#durumlarda hasta sayısı az olursa 1 label'ındaki skorlar kötü çıkacak.

y_train_pred = log_model.predict(X_train_scaled)
#print(classification_report(y_train,y_train_pred))
#              precision    recall  f1-score   support
#
#           0       0.91      0.87      0.89      1807
#           1       0.91      0.95      0.93      2693
#
#    accuracy                           0.91      4500
#   macro avg       0.91      0.91      0.91      4500
#weighted avg       0.91      0.91      0.91      4500



#Amacımız; precision ve recall arasındaki dengeyi sağlayabilmek.
#Bu ikisinin dengesini sağlayan da F1-Score'dur. F1-Score ikisinin harmonik ortalamasını alır.
#Precision : Modelin tahmin gücü recall : Gerçekte o sınıfa ait olanların ne kadarını bilebildim
#Precision çok düşük recall çok yüksekse model sallıyor demek, precision çok 
#yüksek recall çok düşükse model az tahmin yapıyor demektir. Bu yüzden ikisinin dengede olmasını istiyoruz.
# Değerler test ile tutarlı. 
# Traindeki değerler yüksek testteki değerler düşük olsaydı overfitting olduğunu söyleyecektik



plot_confusion_matrix(log_model, X_train_scaled, y_train);

#Cross Validate
# Modelin gerçek performansını görmek için CV yapalım
from sklearn.model_selection import cross_validate

import sklearn
sklearn.metrics.SCORERS.keys()
#dict_keys(['explained_variance', 'r2', 'max_error', 'neg_median_absolute_error',
#           'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 
#           'neg_mean_squared_error', 'neg_mean_squared_log_error',
#           'neg_root_mean_squared_error', 'neg_mean_poisson_deviance', 
#           'neg_mean_gamma_deviance', 'accuracy', 'top_k_accuracy', 'roc_auc',
#           'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 
#           'roc_auc_ovo_weighted', 'balanced_accuracy', 'average_precision', 
#           'neg_log_loss', 'neg_brier_score', 'adjusted_rand_score', 'rand_score',
#           'homogeneity_score', 'completeness_score', 'v_measure_score',
#           'mutual_info_score', 'adjusted_mutual_info_score', 
#           'normalized_mutual_info_score', 'fowlkes_mallows_score', 
#           'precision', 'precision_macro', 'precision_micro', 'precision_samples',
#           'precision_weighted', 'recall', 'recall_macro', 'recall_micro', 
#           'recall_samples', 'recall_weighted', 'f1', 'f1_macro', 'f1_micro', 
#           'f1_samples', 'f1_weighted', 'jaccard', 'jaccard_macro',
#           'jaccard_micro', 'jaccard_samples', 'jaccard_weighted'])
model = LogisticRegression()

scores = cross_validate(model, X_train_scaled, y_train, scoring = ['accuracy', 'precision','recall','f1'], cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores
#    fit_time  score_time  test_accuracy  test_precision  test_recall  test_f1
#1       0.04        0.01           0.90            0.89         0.95     0.92
#2       0.01        0.01           0.92            0.92         0.96     0.94
#3       0.01        0.01           0.94            0.92         0.98     0.95
#4       0.01        0.00           0.93            0.94         0.95     0.94
#5       0.01        0.00           0.94            0.94         0.96     0.95
#6       0.01        0.00           0.90            0.93         0.90     0.92
#7       0.01        0.01           0.90            0.89         0.95     0.92
#8       0.01        0.01           0.91            0.90         0.96     0.93
#9       0.01        0.01           0.89            0.91         0.91     0.91
#10      0.01        0.01           0.92            0.91         0.95     0.93
# 10 kere yapmış ve hepsini hesaplamış

df_scores.mean()[2:] # Üstteki değerlerin ortalamasını hesapladık
# Diğer sonuçlara yakın. Ancak bu CV olduğu için en ideal sonuçlarımız bunlar
#test_accuracy    0.91
#test_precision   0.91
#test_recall      0.95
#test_f1          0.93

# Değerleri birlikte görelim
print("Test Set")
print(classification_report(y_test,y_pred))
print("Train Set\n")
y_train_pred = log_model.predict(X_train_scaled)
print(classification_report(y_train,y_train_pred))
#Test Set
#              precision    recall  f1-score   support
#
#           0       0.92      0.89      0.91       193
#           1       0.93      0.95      0.94       307
#
#    accuracy                           0.93       500
#   macro avg       0.93      0.92      0.93       500
#weighted avg       0.93      0.93      0.93       500

#Train Set

#              precision    recall  f1-score   support

#           0       0.91      0.87      0.89      1807
#           1       0.91      0.95      0.93      2693

#    accuracy                           0.91      4500
#   macro avg       0.91      0.91      0.91      4500
#weighted avg       0.91      0.91      0.91      4500


#ROC (Receiver Operating Curve) and AUC (Area Under Curve)

from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve, roc_auc_score, auc, roc_curve
#1 sınıfını düşman, 0 sınıfını dost gibi düşünüyoruz ve amacımız düşmanı tespit etmek.
#y ekseni, düşman olarak doğru tahmin ettiklerimiz. (True Positive Rate)
#x ekseni, düşman olarak yanlış tahmin ettiklerimiz. (False Positive Rate)
#ROC/AUC; birçok treshold değeri belirler ve buna göre eksende noktalar bulur. 
#(Treshold = 0.5'e göre düşman olduğunu bildim veya bilemedim gibi.) 
#Bu noktaların altında kalan alan ne kadar büyükse, model dost ile düşmanı ayırmakta o kadar başarılı demektir.
#Amacımız, True Positive Rate' i artırıp False Positive Rate'i düşürmek. Modelimizin başarısı 0.96 (Grafiğin sağ altında).

plot_roc_curve(log_model, X_test_scaled, y_test);
# Yorum: En ideali alttaki resimdeki(Çıktının altındaki resim) koyu mavili yerdi
# O koyu eğriye ne kadar yakınsa o kadar iyi
# Eğer bunu bir skora dönüştürmemiz gerekirse AUC u kullanıyoruz
# John Hoca: Binary classification da çok başarılı bir metriktir. Çokça kullandım
# Çizgi üzerindeki noktalar --> Orion hoca: Thresholdlar
# John Hoca: Kısaca bu treshold lara göre çizdiriliyor
#Amaç TP yi max FP yi min yapmak,yani düşmanları olabildiğince fazla tespit edicem(TP-radar örneği),
#dosta da düşman dememen lazım(FP yi min e çekmeye çalışmak)
roc_auc_score(y_test, y_pred_proba[:,1])
#olasılıklar üzerine curve i çizdiği için 1 sınıfına ait olasılıklar verilir
# Orion Hoca:
#!!!!!! Dengeli datasetlerinde ROC / AUC, dengesiz datasetlerinde Precision Recall Curve kullanılır. !!!!!!!!

   
"""
Auc 50 altı yazı tura at daha iyi
50-60 kötü
60-70 idare eder
70-80 iyi
80-90 pekiyi
90 üstü ballı  
"""

#unbalance durumu:Target countları arasında ciddi fark olması,
#ancak bu tek başına yeterli değildir.skorlara da bakılması gerekir,skorlar 
#düşük ve countlar arasında ciddi fark var ise bu durumdan bahsedilebilir
 
#Final Model and Model Deployment
scaler = StandardScaler().fit(X)     
#pickle.dump'ın içine modelimizi hangi isimle kaydedeceğimizi yazıyoruz 
#ve wb olarak kaydet diyoruz. Scale edilmiş datayı kaydetmeden önce sadece 
#fit işlemi uyguluyoruz, transformu kayıt işleminden sonra uyguluyoruz. 
#Yoksa sonradan prediction yaptıracağımız datalarla çalışma yaparken hata alırız.

import pickle
pickle.dump(scaler, open("scaler_hearing", 'wb'))

X_scaled = scaler.transform(X)
# Datanın hepsine transform işlemini uyguladık.
final_model = LogisticRegression().fit(X_scaled, y)

#Modeli lokalimize kaydetmek için tekrar open deyip hangi isimle ve formatla 
#kaydetmek istediğimizi yazıyoruz. Böylece scale - fit edilmiş datayı ve modelimizi lokalimize kaydetmiş olduk.
pickle.dump(final_model, open("final_model_hearing", 'wb'))

#Modelimizie bir tahmin yaptıralım. Bunun için aşağıda bir dict içinde tahmin etmek 
#istediğimiz değerleri veriyoruz. Ardından DataFrame'e dönüştürüyoruz.
my_dict = {"age": [20, 30, 40, 50, 60],
           "physical_score": [50, 30, 20, 10, 5]}

sample = pd.DataFrame(my_dict)
sample
#	age	physical_score
#0	20	 50
#1	30	 30
#2	40	 20
#3	50	 10
#4	60	 5

#Yukarıda kaydettiğimiz scaler'ı tahmin yapması için çağırıyoruz. 
#Bu sefer 'wb' yerine 'rb' yazıyoruz çünkü okutma işlemi yapıyoruz.

scaler_hearing = pickle.load(open("scaler_hearing", "rb"))
#!!!!!!! Scaler'ı transformsuz kaydetmiştik, transform işlemini burda yapıyoruz !!!!!!!!!!

sample_scaled = scaler_hearing.transform(sample)
sample_scaled
#array([[-2.80075819,  2.11038956],
#       [-1.91469467, -0.33789511],
#       [-1.02863115, -1.56203745],
#       [-0.14256762, -2.78617979],
#       [ 0.7434959 , -3.39825096]])

#Lokale kaydettiğimiz modelimizi çağıralım :

final_model = pickle.load(open("final_model_hearing", "rb"))
predictions = final_model.predict(sample_scaled)
predictions_proba = final_model.predict_proba(sample_scaled)   

#Karşılaştırmayı daha iyi yapabilmek için hem 'predict' hem de 'predict_proba' yı kullandık.
#Az önce oluşturduğumuz DataFrame olan sample'a prediction' larımızı sütun olarak ekledik ki sonuçları karşılaştıralım.

sample["pred"] = predictions
sample["pred_proba"] = predictions_proba[:,1]
sample
	age	physical_score	pred	pred_proba
0	20	50	            1	     1.000
1	30	30		        1        0.730
2	40	20		        0        0.016
3	50	10		        0        0.000
4	60	5		        0        0.000


#%%
####################LESSON 9############################################

#NOTEBOOK 9

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%matplotlib inline
#%matplotlib notebook
plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
#pd.options.display.float_format = '{:.3f}'.format

#Bu data setinde en iyi treshold'u seçmeyi öğreneceğiz.
#Elimizde dengesiz bir data seti var, skorları nasıl iyileştirebiliriz bunu göreceğiz.

df=pd.read_csv("diabetes.csv")
df.head()

#   Pregnancies  Glucose  BloodPressure  ...  DiabetesPedigreeFunction  Age  Outcome
#0            6      148             72  ...                     0.627   50        1
#1            1       85             66  ...                     0.351   31        0
#2            8      183             64  ...                     0.672   32        1
#3            1       89             66  ...                     0.167   21        0
#4            0      137             40  ...                     2.288   33        1

#Pregnancies : Kaç defa hamile kalındığı
#Glucose : Vücuttaki şeker oranı
#BloodPressure : Tansiyon
#SkinThickness : Deri kalınlığı
#Insulin : Vücudun ürettiği insulin oranı
#BMI : Vücut indexi
#DiabetesPedigreeFunction : Ailede şeker hastalığı olup olmaması durumuna göre skorlar
#Outcome : 1--> Şeker hastası, 0---> Şeker hastası değil

df.shape
#(768, 9)

#Exploratory Data Analysis and Visualization

df.info()
df.describe().T

df.Outcome.value_counts()  
#0    500
#1    268
#Name: Outcome, dtype: int64
# 1 sayısı az görünüyor ama skorlara bakmadan 'dengesizlik var' gibi bir yorum yapmıyoruz.

sns.countplot(df.Outcome);
sns.boxplot(df.Pregnancies);    
#df=df[df.Pregnancies<13]   Outlier değerler gerçekte de olabilir. 17 kere hamile kalan kadınlar var. 
#Gerçek dünya verileri olduğu için tutuyoruz ama sayıları az olduğu için atıladabilir.

sns.boxplot(df.SkinThickness);
df=df[df.SkinThickness<70]   
# Gerçekte 100 diye bir deri kalınlığı olmadığı için onu attık.
sns.boxplot(df.SkinThickness);

sns.boxplot(df.Insulin);

sns.boxplot(df.Glucose);
df=df[df.Glucose>0]    # Glukoz 0 olamaz o yüzden attık.
sns.boxplot(df.Glucose);

sns.boxplot(df.BloodPressure);
df=df[df.BloodPressure>35]    # Kan basıncı 30'un altında olamaz, o yüzden 35 altını attık.
sns.boxplot(df.BloodPressure);

sns.boxplot(df.BMI);
df=df[df.BMI>0]    # Vücut kitle indexi 0 olamaz, o yüzden attık.
sns.boxplot(df.BMI);

df.shape
#(720, 9)

df.Outcome.value_counts()     
#0    473
#1    247
#Name: Outcome, dtype: int64
# Bazı verileri attıktan sonra veri sayımız biraz düştü.

index = 0
plt.figure(figsize=(20,20))
for feature in df.columns:
    if feature != "Outcome":
        index += 1
        plt.subplot(3,3,index)
        sns.boxplot(x='Outcome',y=feature,data=df)

plt.figure(figsize=(10,8))# Multicollineraity olsa bile Ridge ve Lasso arka planda bu sorunu giderecek (Default--> Ridge)
sns.heatmap(df.corr(), annot=True);    

# df.corr()                                                         
# 1 sınıfıyla olan corr'ların görseli.
# df.corr()["Outcome"].sort_values().plot.barh()                    
# En yüksek corr ilişkisi glukoz ile.
df.corr()["Outcome"].drop("Outcome").sort_values().plot.barh();     
# Glukoz, kilo(BMI), Age yüksekse şeker hastası olma ihtimali yüksek.

sns.pairplot(df, hue = "Outcome");


#Train | Test Split and Scaling
X=df.drop(["Outcome"], axis=1)
y=df["Outcome"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

#Datada dengesizlik olduğu düşünülüyorsa hem test
#hem de train seti split işleminde eşit oranlarla 
#dağılsın diye 'statify = y' denir.
#Bu şekilde 0 sınıfının da %20'sini 1 sınıfının da %20'sini 
#test için ayırır.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Modelling
from sklearn.linear_model import LogisticRegression
log_model=LogisticRegression()
log_model.fit(X_train_scaled, y_train)
y_pred=log_model.predict(X_test_scaled)
y_pred_proba = log_model.predict_proba(X_test_scaled)
#pred'leri karşılaştırmak için X_test ve y_test'leri birleştirip
#pred ve pred_probayı feature olarak ekledik.
#pred_proba'da 0.5'in üstündekileri 1'e altındakileri 0'a atadığını görüyoruz.

test_data = pd.concat([X_test, y_test], axis=1)
test_data["pred"] = y_pred
test_data["pred_proba"] = y_pred_proba[:,1]
test_data.sample(3)
#     Pregnancies  Glucose  BloodPressure  ...  Outcome  pred  pred_proba
#425            4      184             78  ...        1     1       0.784
#630            7      114             64  ...        1     0       0.322
#147            2      106             64  ...        0     0       0.275
#Model Performance on Classification Tasks

from sklearn.metrics import confusion_matrix, classification_report
#Hem train hem de test setini aynı anda görmek için
#aşağıdaki fonksiyonu yazdık. Amacımız, train ile
#test datalarındaki skorları kıyaslayarak bir
#overfitting veya underfitting durumu var mı 
#bunu tespit etmek.


def eval_metric(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    
    print("Test_Set")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print("Train_Set")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))

#df' de 0 ve 1 sınıfları arasında unbalance bir
#durum olabileceğini gözlemlemiştik. Aşağıdaki 
#skorlara baktığımızda bunu teyit edebiliyoruz, skorlar kötü.
#Train_Set ve Test_Set skorları birbirine yakın olduğu
#için overfitting bir durumdan da bahsedemeyiz.

eval_metric(log_model, X_train_scaled, y_train, X_test_scaled, y_test)
#Test_Set
#[[85 10]
# [20 29]]
#              precision    recall  f1-score   support
#
#           0       0.81      0.89      0.85        95
#           1       0.74      0.59      0.66        49
#
#    accuracy                           0.79       144
#   macro avg       0.78      0.74      0.75       144
#weighted avg       0.79      0.79      0.79       144


#Train_Set
#[[337  41]
# [ 89 109]]
#              precision    recall  f1-score   support

#           0       0.79      0.89      0.84       378
#           1       0.73      0.55      0.63       198

#    accuracy                           0.77       576
#   macro avg       0.76      0.72      0.73       576
#weighted avg       0.77      0.77      0.77       576

#0 sınıfına ait skorların daha iyi, 1 sınıfına ait skorların
#daha kötü olduğunu gözlemliyoruz. Peki bunun sebebi ne?
#Çünkü 0 sınıfına ait gözlem sayısı daha fazla. 
#Bu yüzden eğitimini daha iyi yapmış.
#Cross Validate ile yukarıda aldığımız skorları teyit edeceğiz :

#    

from sklearn.model_selection import cross_validate
model = LogisticRegression()
scores = cross_validate(model, X_train_scaled, y_train, scoring = ['precision','recall','f1','accuracy'], cv = 10)

df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores                                                  
# Binary modellerdeki skorlar her zaman 1 class'ına ait skorlardır.
#    fit_time  score_time  test_precision  test_recall  test_f1  test_accuracy
#1      0.009       0.005           0.600        0.450    0.514          0.707
#2      0.011       0.008           0.643        0.450    0.529          0.724
#3      0.007       0.005           0.923        0.600    0.727          0.845
#4      0.006       0.004           0.857        0.600    0.706          0.828
#5      0.005       0.004           0.706        0.600    0.649          0.776
#6      0.005       0.003           0.647        0.550    0.595          0.741
#7      0.006       0.004           0.714        0.526    0.606          0.772
#8      0.006       0.004           0.647        0.579    0.611          0.754
#9      0.007       0.007           0.733        0.550    0.629          0.772
#10     0.005       0.005           0.625        0.500    0.556          0.719

df_scores.mean()[2:]     
# Sadece skorları görebilmek için 2. indexten sonrasına bakıyoruz.   (Scale edilmiş skorlar)
#test_precision   0.710
#test_recall      0.541
#test_f1          0.612
#test_accuracy    0.764
#dtype: float64

#Aşağıya eval_metric' i tekrar yazdıralım, scale edilmeden
#önceki skorlar(yukarıdaki) ile sonraki skorları (aşağıdaki) kıyaslayalım.
#Scale edildikten sonra skorların biraz düştüğünü gördük. 
#Ama çok bariz bir fark yok diyebiliriz. Cross Validate işlemi bu durumu tespit etmek adına önemli.
eval_metric(log_model, X_train_scaled, y_train, X_test_scaled, y_test)
# (Scale edilmemiş skorlar)
#Yukarıdaki skorlarımızda Recall çok düşük. Amacımız, 
#Recall'ı artırmak ama Precision ile dengeli bir şekilde. 
#Dolayısıyla f1-score' da artmış olacak. Modelimiz iyileşecek.
#!!!!!!! Çok dengesiz datasetlerinde Test_Set ile Train_Set 
#arasında çok fazla fark olduğunda overfitting durumu var diyemeyiz. 
#Bu durumda bakacağımız skorlar 'macro' ve 'weighted' olmalı. 
#Overfitting olup olmadığına bu iki skorla karar verebiliriz. !!!!!!!
#Modelimizde overfitting olsa bile logisticRegression() içine penalty = l1 gibi değerler yazarak overfitting ile mücadele edebiliriz. (Default değeri l2.)

#Cross Validate for 0 class
#Eğer default class olan 1 değil de 0 sınıfı için cross validate 
#yapmak istersek 'make_scorer' fonksiyonunu import ediyoruz

from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

import sklearn
sklearn.metrics.SCORERS.keys()
#Yukarıdaki skorları direk make_scorer içinde kullanamıyorum
#çünkü kullandığım zaman 1 sınıfının skorlarını hesaplıyor. 
#Biz ise 0 sınıfının skorlarını istiyoruz.

#make_scorer : 'pos_label' aslında make_scorer' ın 
#içinde geçmiyor ama make_scorer içinde kullanılan 
#fonksiyonun da içinde geçenleri kullanabilirsin diyor, 
#böyle bir esneklik sağlıyor. Mesela f1_score()'un içinde 
#geçen pos_label' ı burada kullanabileceğiz. Bu yüzden 
#make_scorer'ın içine pos_label' ı ekledik.

f1_0 = make_scorer(f1_score, pos_label =0)
precision_0 = make_scorer(precision_score, pos_label =0)
recall_0 = make_scorer(recall_score, pos_label =0)

model = LogisticRegression()
#cross_validate'de scoring'lere mse, rmse gibi 
#skorları yazıyorduk. Burda 0 sınıfına ait skorları istediğimiz için, yukarıda 
#tanımladığımız f1, precision ve recall değerlerini dict olarak 
#cross_validate'in içine veriyoruz. Böylece 0 sınıfına ait skorları alabileceğiz.

scores = cross_validate(model, X_train_scaled, y_train,scoring = {"precision_0":precision_0, "recall_0":recall_0, "f1_0":f1_0}, cv = 10)
#Bulduğumuz skorları DataFrame yapısına çevirdik :
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores
#    fit_time  score_time  test_precision_0  test_recall_0  test_f1_0
#1      0.012       0.003             0.744          0.842      0.790
#2      0.011       0.007             0.750          0.868      0.805
#3      0.012       0.006             0.822          0.974      0.892
#4      0.008       0.007             0.818          0.947      0.878
#5      0.007       0.004             0.805          0.868      0.835
#6      0.008       0.005             0.780          0.842      0.810
#7      0.008       0.004             0.791          0.895      0.840
#8      0.010       0.005             0.800          0.842      0.821
#9      0.011       0.007             0.786          0.892      0.835
#10     0.007       0.005             0.756          0.838      0.795

df_scores.mean()[2:]         # Yukarıda bulduğumuz skorların ortalamasını aldık.
#test_precision_0   0.785
#test_recall_0      0.881
#test_f1_0          0.830
#dtype: float64
#Aşağıda Cross_Validate yapılmamış değerleri tekrar yazdırdık ki bir kıyas yapalım, 
#yukarıda yeni bulduğum değerlerle cross işlemi öncesi skorlarım değişmiş mi?
#Kıyaslama için bu sefer 0 değerlerine odaklanacağız. Çünkü 0 değerlerinin 
#scorlarını bulduk. (cross_validate işleminden sonra skorlarımın biraz düştüğünü gözlemliyorum.)
eval_metric(log_model, X_train_scaled, y_train, X_test_scaled, y_test)
#Test_Set
#[[85 10]
# [20 29]]
#              precision    recall  f1-score   support

#           0       0.81      0.89      0.85        95
#           1       0.74      0.59      0.66        49

#    accuracy                           0.79       144
#   macro avg       0.78      0.74      0.75       144
#weighted avg       0.79      0.79      0.79       144
#Train_Set
#[[337  41]
# [ 89 109]]
#              precision    recall  f1-score   support

#           0       0.79      0.89      0.84       378
#           1       0.73      0.55      0.63       198

#    accuracy                           0.77       576
#   macro avg       0.76      0.72      0.73       576
#weighted avg       0.77      0.77      0.77       576

#GridSearchCV
#Skorlarımız çok iyi çıkmadı. Peki bunları nasıl iyileştireceğiz?
import sklearn
sklearn.metrics.SCORERS.keys()
from sklearn.model_selection import GridSearchCV

model = LogisticRegression()
#Logistic Regression İçindeki Parametreler :
#ogisticRegression overfitting ile mücadele etmek amacıyla içine penalty = l1, l2, elasticnet parametrelerini alıyordu.
#l2 ----> Ridge
#l1 ----> Lasso
#Linear Regression'daki alpha yerine burda C parametresi var. Bu parametre alpha 
#ile ters orantılı çalışır. Alpha büyüdükçe regularization artar; C küçüldükçe 
#regularization artar (bias ekler). Yani C değerinin küçülmesi iyi bir şey.
#class_weight : Class sayıları arasında dengesizlik varsa; sayısı az olan 
#sınıfı daha çok ağırlıklandırır. Yani zayıf olan sınıfa daha çok tahmin yaptırır.
#solver : Modeller metricleri minimize etmek için 'Gradient Descent tabanlı' 
#çalışırlar. Solver metrikleri de Gradient Descent methodlarıdır. 
#Çok bilinmiyorsa default değerlerinin değiştirilmesi önerilmez. Çoğunlukla default değeri 
#iyi sonuç verir. (solver : 'lbfgs')
#Eğer data küçükse ''solver : liblinear'', çok büyük datalarda ise ''solver : sag'' 
#veya ''solver : saga'' iyi bir seçim olabilir. Kafamızda soru işareti oluştuğu 
#zaman bunları deneyerek sonuçları karşılaştırabiliriz.
#multi_class : 0, 1, 2 diye üç sınıf olsun. ROC/AUC çizerken 2 sadece binary 
#olanları çizebiliyor. Burdaki 3 sınıfı çizmek için herhangi bir sınıfı alıp 
#geri kalanına tek bir sınıf gibi davranır. Böylece 2 sınıf varmış gibi olur. 
#Tüm ihtimaller için bunu yapar ve çizgilerini çizer. 
#multi_class = 'ovr' bunu sağlar. default = 'auto'
#Biz aşağıda bir fonksiyon tanımlayarak Ridge ve Lasso'dan hangisinin daha iyi sonuç verdiğine bakacağız :

penalty = ["l1", "l2"]                
# l1 ve l2 skorlarına bakacağız.
C = np.logspace(-1, 5, 20)            
# C parametresi logspace aralığında daha iyi sonuçlar verir. (Hangi sayının logunu aldığımda bu aralıktan bir sayı döndürür?)
class_weight= ["balanced", None]      
# Classlar arası dengeleme yapsın veya yapmasın.

# The "balanced" mode uses the values of y to automatically adjust weights inversely proportional to class frequencies 
# in the input data

solver = ["lbfgs", "liblinear", "sag", "saga"]   
# Gradient descent methodlarından hangisini kullanayım?
param_grid = {"penalty" : penalty,
              "C" : C,
              "class_weight":class_weight,
              "solver":solver}


grid_model = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          cv=10,
                          scoring = "recall",     
                          n_jobs = -1) 
# 1 sınıfına ait en iyi recall'ı hangi parametreler getirecek? Bunu hesaplar.             
# Recall dedik çünkü skorlarımızda bu değer kötü. f1 de diyebilirdik. Sırayla denenebilir.
grid_model.fit(X_train_scaled,y_train)     # Eğitimimizi yaptık.

grid_model.best_params_     
#{'C': 0.1, 'class_weight': 'balanced', 'penalty': 'l1', 'solver': 'liblinear'}   
# Yukarıda tanımlanan modele göre çıkan en iyi parametre değerleri.
eval_metric(grid_model, X_train_scaled, y_train, X_test_scaled, y_test)
#Test_Set
#[[76 19]
# [13 36]]
#              precision    recall  f1-score   support
#
#           0       0.85      0.80      0.83        95
#           1       0.65      0.73      0.69        49
#
#    accuracy                           0.78       144
#   macro avg       0.75      0.77      0.76       144
#weighted avg       0.79      0.78      0.78       144


#Train_Set
#[[288  90]
#[ 49 149]]
#              precision    recall  f1-score   support

#           0       0.85      0.76      0.81       378
#           1       0.62      0.75      0.68       198

#    accuracy                           0.76       576
#   macro avg       0.74      0.76      0.74       576
#weighted avg       0.78      0.76      0.76       576


#Yeni sonuç ile eski sonucu kıyasladığımızda; precision değerinin düştüğünü ama 
#recall değerinin de yükseldiğini görüyoruz. f1 score da 66'dan 70'e çıkarak 
#dengeyi korumuş. Amacımıza ulaştık; recall değerini dengeli bir şekilde artırdık.

#0 skorları ise düştü. Bizim amacımız hasta olanları yani 1'leri tespit etmek.
#Bu yüzden 1 olanları iyileştirmeye yönelik parametreler kullandık.

#Tek bir modelde hem 1 hem 0' lar için skorlara bakılmaz, bu hatalı olur. 
#0 skorlarına bakıyorsak ayrı model, 1 skorlarına bakıyorsak ayrı model kullanmalıyız.



#ROC (Receiver Operating Curve) and AUC (Area Under Curve)

#ROC/AUC; birçok treshold değeri belirler ve buna göre eksende noktalar bulur. 
#(Treshold = 0.5'e göre düşman olduğunu bildim veya bilemedim gibi.) 
#Bu noktaların altında kalan alan ne kadar büyükse, model dost ile düşmanı ayırmakta o kadar başarılı demektir.

#1 sınıfını düşman, 0 sınıfını dost gibi düşünüyoruz ve amacımız düşmanı tespit etmek.

#y ekseni, düşman olarak doğru tahmin ettiklerimiz. (True Positive Rate)

#x ekseni, düşman olarak yanlış tahmin ettiklerimiz. (False Positive Rate)


from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve, roc_auc_score, auc, roc_curve, average_precision_score, precision_recall_curve

#1 noktası True-Positive'in en yüksek olduğu nokta; 0 noktası ise False- Positive'in
#en düşük olduğu nokta. Amacımız ilkini yüksek, ikinciyi düşük yapmak ki alttaki alan büyüsün.

plot_roc_curve(grid_model, X_test_scaled, y_test);                # Modelin başarısı : 0.85




#Precision-Recall-Curve
plot_precision_recall_curve(grid_model, X_test_scaled, y_test);      #Modelin başarısı : 0.78

#Bizim için geçerli olan yöntem : precision_recall_curve. Çünkü datasetimiz dengesiz.
#(İlk yöntem daha iyi olmasına rağmen ikinci yöntemi seçtik.)


#*********Finding Best Threshold************

#Amacımız yeni treshold'lar belirleyerek en büyük alanı çizebilmek ve model başarısını artırabilmek.
#default olarak treshold değeri = 0.5
#Yeni tresholdlar belirleyerek mesela 0.3; 0.3 ün altındakilere 0; üstündekilere 1 diyeceğiz.

#ROC ve AUC ile Best Treshold :
#Bunun için en iyi treshold değerini bulacağız yani alttaki alanın en büyük olduğu treshold değeri.
#!!!!!! Dengeli datasetlerinde ROC / AUC, dengesiz datasetlerinde Precision Recall Curve kullanılır. !!!!!!!!
#!!!!!!!!! Best treshold sadece train setinde bulunur. Eğer test setinde de denersek 'data leakage(kopye)' olur. !!!!!!!!

plot_roc_curve(log_model, X_train_scaled, y_train);   
y_pred_proba = log_model.predict_proba(X_train_scaled)  
# Train setindeki predict_proba' yı aldık ki yukardaki grafikteki skorla karşılaştırabilelim.
roc_auc_score(y_train, y_pred_proba[:,1])      
#0.8378493934049489     
# roc_ouc_score içine eğittiğimiz y yi ve y_train'den aldığımız proba'nın 1 sınıfı için olan değer

#Değerimiz yukarıdaki grafikte 0.84 çıkmıştı predict işleminde de 0.83 çıktı. Birbirine yakın değerler elde ettik.
#fp_rate : False - Positive Rate (Amaç minimum yapmak). (FPR)
#tp_rate : True - Positive Rate (Amaç maximum yapmak). (TPR)
#treshold : 0 - 1 arasında aldığı olasılıklar.

#fp_rate    # Her bir treshold'a göre aldığı olasılık değerleri.

#tp_rate     #Her bir treshold'a göre aldığı olasılık değerleri.

fp_rate, tp_rate, thresholds = roc_curve(y_train, y_pred_proba[:,1])


#(max TPR) - (min FPR) çıkarırsak; burası düşmanın en iyi tespit edildiği noktadır.
#(Düşmana düşman dediğim max değerden, dosta düşman dediğim min değeri çıkardım.)

optimal_idx = np.argmax(tp_rate - fp_rate)          
# İçerideki max değer neyse onun index nosunu döndürür.
optimal_threshold = thresholds[optimal_idx]         
# Bulunan indexi tresholdun içine verdik. En optimal treshold'u bize döndürür.
optimal_threshold
#0.33938184887578754

#!!!!! Best treshold için ROC ve AUC da kullanabiliriz,
#Precision-Recall-Curve da kullanabiliriz. Aynı sonuçlar çıkar, 
#sadece hesaplamaları farklı. (ROC AUC mantığı daha kolay) !!!!!

#Precision-Recall-Curve ile Best Treshold :

plot_precision_recall_curve(grid_model, X_train_scaled, y_train);   

y_pred_proba = log_model.predict_proba(X_train_scaled)
average_precision_score(y_train, y_pred_proba[:,1])
#0.7120696300524079

precisions, recalls, thresholds = precision_recall_curve(y_train, y_pred_proba[:,1])

optimal_idx = np.argmax((2 * precisions * recalls) / (precisions + recalls))
optimal_threshold = thresholds[optimal_idx]
optimal_threshold
#0.33938184887578754


grid_model.predict_proba(X_test_scaled)[:,1]    
# 0.5 treshold'a göre dönen değerler.

#Biz aşağıdaki fonksiyonda artık 0.5 değil de yeni bulduğumuz best treshold olan 0.33'e göre değerler döndüreceğiz.
#Önce seri içine yazdık yoksa apply fonk. uygulayamazdık

#Aldığımız değer yeni treshold'dan büyükse (0.33), 1 sonucunu döndür; değilse 0 döndür.

y_pred2 = pd.Series(grid_model.predict_proba(X_test_scaled)[:,1]).apply(lambda x : 1 if x >= optimal_threshold else 0)

print(confusion_matrix(y_test,y_pred2))
print(classification_report(y_test,y_pred2))
#[[55 40]
# [ 3 46]]
#              precision    recall  f1-score   support

#           0       0.95      0.58      0.72        95
#           1       0.53      0.94      0.68        49

#    accuracy                           0.70       144
#   macro avg       0.74      0.76      0.70       144
#weighted avg       0.81      0.70      0.71       144

#Yukarıdaki sonuçlara baktığımızda 1 class'ına ait precision değerleri hemen 
#hemen aynı ama recall değeri baya yükseldi.

#Aşağıda yeni treshold değeri ile train setine de baktık. 
#Orda da 1 sınıfına ait recall değerlerinin iyileştiğini görüyoruz.

y_train_pred2 = pd.Series(grid_model.predict_proba(X_train_scaled)[:,1]).apply(lambda x : 1 if x >= optimal_threshold else 0)
print(confusion_matrix(y_train, y_train_pred2))
print(classification_report(y_train, y_train_pred2))

#[[196 182]
# [ 16 182]]
#              precision    recall  f1-score   support

#           0       0.92      0.52      0.66       378
#           1       0.50      0.92      0.65       198

#    accuracy                           0.66       576
#   macro avg       0.71      0.72      0.66       576
#weighted avg       0.78      0.66      0.66       576

#Aşağıdaki fonksiyon, treshold'u ile oynanmış bir dataya Cross Validation'ın 
#arkada yaptığı işlemin manual olarak yapılması. LogisticRegression'da yapılan işlemleri içeriyor :


from sklearn.model_selection import StratifiedKFold    # Modeli kaç parçaya ayırmak istiyorsak ona göre index numaraları belirler.

def CV(n, est, X, y, optimal_threshold):
    skf = StratifiedKFold(n_splits = n, shuffle = True, random_state = 42)
    acc_scores = []
    pre_scores = []
    rec_scores = []
    f1_scores  = []
    
    X = X.reset_index(drop=True)       # Index no'ları her işlemden sonra sıfırlaması için.
    y = y.reset_index(drop=True)
    
    for train_index, test_index in skf.split(X, y):
        
        X_train = X.loc[train_index]
        y_train = y.loc[train_index]
        X_test = X.loc[test_index]
        y_test = y.loc[test_index]
        
        
        est = est
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        y_pred_proba = est.predict_proba(X_test)
             
        y_pred2 = pd.Series(y_pred_proba[:,1]).apply(lambda x : 1 if x >= optimal_threshold else 0)
        
        acc_scores.append(accuracy_score(y_test, y_pred2))
        pre_scores.append(precision_score(y_test, y_pred2, pos_label=1))
        rec_scores.append(recall_score(y_test, y_pred2, pos_label=1))
        f1_scores.append(f1_score(y_test, y_pred2, pos_label=1))
    
    print(f'Accuracy {np.mean(acc_scores)*100:>10,.2f}%  std {np.std(acc_scores)*100:.2f}%')
    print(f'Precision-1 {np.mean(pre_scores)*100:>7,.2f}%  std {np.std(pre_scores)*100:.2f}%')
    print(f'Recall-1 {np.mean(rec_scores)*100:>10,.2f}%  std {np.std(rec_scores)*100:.2f}%')
    print(f'F1_score-1 {np.mean(f1_scores)*100:>8,.2f}%  std {np.std(f1_scores)*100:.2f}%')


model = LogisticRegression(C= 0.1, class_weight= 'balanced',penalty= 'l1',solver= 'liblinear')  
CV(10, model, pd.DataFrame(X_train_scaled), y_train, optimal_threshold)   
#Accuracy      64.76%  std 5.47%
#Precision-1   49.86%  std 4.68%
#Recall-1      91.45%  std 5.95%
#F1_score-1    64.25%  std 3.31%
# Bulduğumuz C değeri ve kullandığımız parametreler neyse yazmalıyız.
# Scale edilmiş data array' e dönüştüğü için burda tekrar DataFrame'e dönüştürüyoruz.

#n_split : Data setini 10'a böl 9' unu train, 1' ini test seti yapar. 9 tane train'in indexlerini belirler.
#test_index : Test seti için ayırdıklarının index no'su.
#Bu index no'lara göre yukarıdaki fonksiyondaki for döngüsüne girer. Index no' lara göre 
#X_train, y_train, X_test, y_test değerlerini belirler. Her for döngüsünde train ve test setleri değişir.
#est = est kısmında modeli eğitir ve pred ve predict_proba' ları alır.
#y_pred2 = pd.Series(y_pred_proba[:,1]).apply(lambda x : 1 if x >= optimal_threshold else 0) 
#kısmında ise daha önce yukarda yaptığımız gibi optimal treshold değerlerini bulur.
#Bulduğu her değeri acc_scores = [], pre_scores = [], rec_scores = [], f1_scores = [] içine atar.
#print kısmında ise bulunan değerlerin ortalamasını alır.


#**********Finding Best Threshold for the most balanced score between recall and precision****

#eğer precision ile recal arasındaki en dengeli skor isteniyorsa 
plot_roc_curve(grid_model, X_train_scaled, y_train);   #Modelin başarısı : 0.78

y_pred_proba = grid_model.predict_proba(X_train_scaled)
average_precision_score(y_train, y_pred_proba[:,1])
#0.7039749907641875

fp_rate, tp_rate, thresholds = roc_curve(y_train, y_pred_proba[:,1])

optimal_idx = np.argmax(tp_rate-fp_rate)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold
#0.4864636302091975

y_pred2 = pd.Series(grid_model.predict_proba(X_test_scaled)[:,1]).apply(lambda x : 1 if x >= optimal_threshold else 0)

print(confusion_matrix(y_test,y_pred2))
print(classification_report(y_test,y_pred2))
#[[76 19]
# [12 37]]
#              precision    recall  f1-score   support
#
#           0       0.86      0.80      0.83        95
#           1       0.66      0.76      0.70        49
#
#    accuracy                           0.78       144
#   macro avg       0.76      0.78      0.77       144
#weighted avg       0.79      0.78      0.79       144

y_train_pred2 = pd.Series(grid_model.predict_proba(X_train_scaled)[:,1]).apply(lambda x : 1 if x >= optimal_threshold else 0)
print(confusion_matrix(y_train, y_train_pred2))
print(classification_report(y_train, y_train_pred2))

#[[284  94]
# [ 45 153]]
#             precision    recall  f1-score   support
#
#           0       0.86      0.75      0.80       378
#           1       0.62      0.77      0.69       198
#
#    accuracy                           0.76       576
#   macro avg       0.74      0.76      0.75       576
#weighted avg       0.78      0.76      0.76       576

model = LogisticRegression(C= 0.1, class_weight= 'balanced',penalty= 'l1',solver= 'liblinear')  
CV(10, model, pd.DataFrame(X_train_scaled), y_train, optimal_threshold)   

#Accuracy      73.61%  std 6.37%
#Precision-1   59.76%  std 8.11%
#Recall-1      75.79%  std 11.53%
#F1_score-1    66.36%  std 7.60%



#Final Model and Model Deployment
scaler = StandardScaler().fit(X)
    
import pickle
pickle.dump(scaler, open("scaler_diabates", 'wb'))   
    
X_scaled = scaler.transform(X)

final_model = LogisticRegression(C=0.1,class_weight = "balanced",penalty='l1',solver='liblinear').fit(X_scaled, y)

pickle.dump(final_model, open("final_model_diabates", 'wb'))

X.describe().T


my_dict = {"Pregnancies": [3, 6, 5],
           "Glucose": [117, 140, 120],
           "BloodPressure": [72, 80, 75],
           "SkinThickness": [23, 33, 25],
           "Insulin": [48, 132, 55],
           "BMI": [32, 36.5, 34],
           "DiabetesPedigreeFunction": [0.38, 0.63, 0.45],
           "Age": [29, 40, 33]
          }

sample = pd.DataFrame(my_dict)
sample

scaler_diabates = pickle.load(open("scaler_diabates", "rb"))

sample_scaled = scaler_diabates.transform(sample)
sample_scaled

final_model = pickle.load(open("final_model_diabates", "rb"))

predictions = final_model.predict(sample_scaled)
predictions_proba = final_model.predict_proba(sample_scaled)
predictions2 = [1 if i >= optimal_threshold else 0 for i in predictions_proba[:,1]]


sample["pred_proba"] = predictions_proba[:,1]
sample["pred_0.50"] = predictions
sample["pred_0.34"] = predictions2
sample

scaler_diabates = pickle.load(open("scaler_diabates", "rb"))

sample_scaled = scaler_diabates.transform(sample)
sample_scaled

final_model = pickle.load(open("final_model_diabates", "rb"))

predictions = final_model.predict(sample_scaled)
predictions_proba = final_model.predict_proba(sample_scaled)
predictions2 = [1 if i >= optimal_threshold else 0 for i in predictions_proba[:,1]]

sample["pred_proba"] = predictions_proba[:,1]
sample["pred_0.50"] = predictions
sample["pred_0.34"] = predictions2
sample



#%%

#Multi-Class Logistic Regression

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")
pd.set_option('display.float_format', lambda x: '%.3f' % x)


df = pd.read_csv('iris.csv')

df.head()

#   sepal_length  sepal_width  petal_length  petal_width species
#0         5.100        3.500         1.400        0.200  setosa
#1         4.900        3.000         1.400        0.200  setosa
#2         4.700        3.200         1.300        0.200  setosa
#3         4.600        3.100         1.500        0.200  setosa
#4         5.000        3.600         1.400        0.200  setosa

#Exploratory Data Analysis and Visualization

df.info()
df.describe().T
df['species'].value_counts()
#setosa        50
#versicolor    50
#virginica     50
#Name: species, dtype: int64

sns.pairplot(df,hue='species');
sns.heatmap(df.corr(),annot=True);

#Train | Test Split and Scaling
X = df.drop('species',axis=1)
y = df['species']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Modelling and Model Performance

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix


#With Default Parameters

log_model = LogisticRegression()
log_model.fit(X_train_scaled,y_train)
y_pred = log_model.predict(X_test_scaled)
y_pred

plot_confusion_matrix(log_model, X_test_scaled, y_test)

def eval_metric(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    
    print("Test_Set")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print("Train_Set")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))

eval_metric(log_model, X_train_scaled, y_train, X_test_scaled, y_test)

#Test_Set
#[[10  0  0]
# [ 0 12  0]
# [ 0  1  7]]
#              precision    recall  f1-score   support
#
#      setosa       1.00      1.00      1.00        10
#  versicolor       0.92      1.00      0.96        12
#   virginica       1.00      0.88      0.93         8
#
#    accuracy                           0.97        30
#   macro avg       0.97      0.96      0.96        30
#weighted avg       0.97      0.97      0.97        30


#Train_Set
#[[40  0  0]
# [ 0 35  3]
# [ 0  1 41]]
#              precision    recall  f1-score   support
#
#      setosa       1.00      1.00      1.00        40
#  versicolor       0.97      0.92      0.95        38
#   virginica       0.93      0.98      0.95        42
#
#    accuracy                           0.97       120
#   macro avg       0.97      0.97      0.97       120
#weighted avg       0.97      0.97      0.97       120

#Cross Validate

from sklearn.model_selection import cross_validate

model = LogisticRegression()

scores = cross_validate(model, X_train_scaled, y_train, scoring = ['accuracy', 'precision_weighted','recall_weighted',
                                                                   'f1_weighted'], cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
#test_accuracy             0.950
#test_precision_weighted   0.960
#test_recall_weighted      0.950
#test_f1_weighted          0.949
#dtype: float64

#Cross Validate for versicolar
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score


f1_versicolor = make_scorer(f1_score, average = None, labels =["versicolor"])
precision_versicolor = make_scorer(precision_score, average = None, labels =["versicolor"])
recall_versicolor = make_scorer(recall_score, average = None, labels =["versicolor"])


scoring = {"f1_versicolor":f1_versicolor, 
           "precision_versicolor":precision_versicolor,
           "recall_versicolor":recall_versicolor}

model = LogisticRegression()

scores = cross_validate(model, X_train_scaled, y_train, scoring = scoring, cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
#test_f1_versicolor          0.924
#test_precision_versicolor   0.940
#test_recall_versicolor      0.925
#dtype: float64

#Class prediction
y_pred=log_model.predict(X_test_scaled)
y_pred_proba = log_model.predict_proba(X_test_scaled)

test_data = pd.concat([X_test, y_test], axis=1)
test_data["pred"] = y_pred
test_data["pred_proba_setosa"] = y_pred_proba[:,0]
test_data["pred_proba_versicolar"] = y_pred_proba[:,1]
test_data["pred_proba_virginica"] = y_pred_proba[:,2]
test_data.sample(10)

#With Best Parameters (GridsearchCV)
log_model = LogisticRegression(max_iter=5000)
penalty = ["l1", "l2"]
C = [0.01, 0.1, 1, 5, 16, 19, 22, 25]

param_grid = {"penalty" : penalty,
             "C" : C}


grid_model = GridSearchCV(log_model, param_grid = param_grid, cv=5) 
#scoring = f1_versicolor = make_scorer(f1_score, average = None, labels =["versicolor"]) 

grid_model.fit(X_train_scaled,y_train)
#GridSearchCV(cv=5, estimator=LogisticRegression(max_iter=5000),
#             param_grid={'C': [0.01, 0.1, 1, 5, 16, 19, 22, 25],
#                         'penalty': ['l1', 'l2']})


grid_model.best_params_
#{'C': 19, 'penalty': 'l2'}

grid_model.best_score_
# 0.975

y_pred = grid_model.predict(X_test_scaled)
y_pred

plot_confusion_matrix(grid_model, X_test_scaled, y_test)


eval_metric(grid_model, X_train_scaled, y_train, X_test_scaled, y_test)

#Test_Set
#[[10  0  0]
# [ 0 12  0]
# [ 0  0  8]]
#              precision    recall  f1-score   support
#
#      setosa       1.00      1.00      1.00        10
#  versicolor       1.00      1.00      1.00        12
#   virginica       1.00      1.00      1.00         8

#    accuracy                           1.00        30
#   macro avg       1.00      1.00      1.00        30
#weighted avg       1.00      1.00      1.00        30


#Train_Set
#[[40  0  0]
# [ 0 37  1]
# [ 0  1 41]]
#              precision    recall  f1-score   support
#
#      setosa       1.00      1.00      1.00        40
#  versicolor       0.97      0.97      0.97        38
#   virginica       0.98      0.98      0.98        42
#
#    accuracy                           0.98       120
#   macro avg       0.98      0.98      0.98       120
#weighted avg       0.98      0.98      0.98       120


#ROC (Receiver Operating Curve) and AUC (Area Under Curve)
from sklearn.metrics import plot_roc_curve
plot_roc_curve(grid_model, X_test_scaled, y_test);

from yellowbrick.classifier import ROCAUC
model = LogisticRegression(C= 19, max_iter=5000)
visualizer = ROCAUC(model) # for binary data per_class=False, binary=True

visualizer.fit(X_train_scaled, y_train)        # Fit the training data to the visualizer
visualizer.score(X_test_scaled, y_test)        # Evaluate the model on the test data
visualizer.show();               

##############

from yellowbrick.classifier import PrecisionRecallCurve

model = LogisticRegression(C= 19, max_iter=5000)

viz = PrecisionRecallCurve(
    model,
    classes = ["setosa","versicolar","virginica"],
    per_class=True
)
viz.fit(X_train_scaled, y_train)
viz.score(X_test_scaled, y_test)
viz.show();


#Final Model and Model Deployment
scaler = StandardScaler().fit(X)

import pickle
pickle.dump(scaler, open("scaler_iris", 'wb'))

X_scaled = scaler.transform(X)

final_model = LogisticRegression().fit(X_scaled, y)
pickle.dump(final_model, open("final_model_iris", 'wb'))

X.describe().T
my_dict = {"sepal_length": [4.5, 5.8, 7.5],
           "sepal_width": [2.2, 3, 4.2],
           "petal_length": [1.3, 4.3, 6.5],
           "petal_width": [0.2, 1.3, 2.3]
          }

sample = pd.DataFrame(my_dict)
sample

scaler_iris = pickle.load(open("scaler_iris", "rb"))

sample_scaled = scaler_iris.transform(sample)
sample_scaled

final_model = pickle.load(open("final_model_iris", "rb"))

predictions = final_model.predict(sample_scaled)
predictions_proba = final_model.predict_proba(sample_scaled)

sample["pred"] = predictions
sample["pred_proba_setosa"] = predictions_proba[:,0]
sample["pred_proba_versicolor"] = predictions_proba[:,1]
sample["pred_proba_virginica"] = predictions_proba[:,2]
sample



#%%%
#LESSON 10

#K Nearest Neighbors Theory (KNN)

#*'A lazy lenear' bir algoritmadir 
#*Training data yoktur 
#*Butun datayi hafizasina alir,yeni gelen data nereye gelirse en yakinindaki 
#komsularina gore o datayi siniflandirir 
#Linear regression ve logistic regression parametric bir algoritmalardi.Fakat,KNN 
#non-parametric bir algoritmadir.Yani b1 ve bo gibi katsayilar yoktur.
#Hicbir hesaplama yapmaz

#KNN genelde az verili data setlerinde kullanilir.Buyuk data setleri icin uygun 
#degildir.
#Low dimensional datasets,Fault detection,Recommender systems gibi alanlarda
#kullanilir


#Hyper Parameters (k=5,weights='uniform',metric='minkowski',p)

#1- "k"=>Sample in khangi class a atilacagina k komsuluk sayisina gore karar 
#verir.Bu yuzden k secimi cok onemlidir.Ornegin,k=5 verilirse en yakin 5 
#komsuya bakar,encok hangi class tan eleman var ise sample i o class a atar 
#*eger k degeri cok buyuk secilirse "large bias" durumu ortaya cikar.Underfit 
#durumu ortaya cikar 

#Eger k degeri cok kucuk secilirse overfit durumu ortaya cikar.Train set te cok 
#iyi bir basari elde eder ama test datasinda cok kotu tahminler yapar 
 
#Bu yuzden optimal k degeri bulmak onemlidir.Optimal k yi bulabilmek icin,
#train-test datalari bolunur.K icin bir aralik verilir ve bu her k degeri icin 
#Cross Validate islemi yapilarak en az hata veren k degeri secilir 


#**ELBOW METHOD**
#K icin verilen araliktaki her k degeri icin error metrikleri hesaplanir.Bu 
#error metriklerine gore bir grafik elde edilir 
#KNN metodunda Grid Search islemi yerine elbow metodu ile cizilen grafige 
#gore karar vermek daha mantiklidir 

#Gridhsearch islemi k yi 20-25 degerleri arasina goturur 

#2 Weights =>{'uniform','distance'}
#uniform(default deger)=>her bir komsunun 1 oyu var 

#distance=>Yakin olan komsunun biraz daha uzak olana ustunlugu olur.Yakin olanin 
#oyu artar.Bu sekilde tek bir komsu,birden fazla komsudan daha agir oldugu icin 
#ustunluk kazanabilir 

#3 Metric=>Mesafe hesabi yapan yaparetme.(hem kategoric hem cotinius olan)
#datalarda kullanilir

#Euclidean Distance
#Manhattan Distance   =>bu ikisinin karisimi olan ("minkowski") default parametre

#a
#|            eucllidean distance=|ac| uzunlugu
#|            manhattan distance=|ab|+|bc| uzunlugu
#|________
#b        c

#4-  p=>Minkowski parametresi icin kullanilan bir parametre

#p=1 =>Manhattan distance gibi davranir 
#hatalari cezalandirdigi icin outlier lara karsi direncli 
#daha cok multimensional data setlerinde kullanilir
#(feature sayisi>5 olan data setlerinde)

#p=2 =>Euclidean distance gibi davranir 
#outlier lar ile mucadelelerde iyi degil.Daha cok kucuk data setlerinde kullanilir
#(feature sayisi<5 olan data setlerinde)

#!!!!!!KNN de scale islemi zorunlu.Cunku mesafe tabanli bir algoritma

#Olumlu yonleri>
#*Lazy learner oldugu icin assumptionlar ile ilgilenmez.Bu yuzden her datada 
#kullanilabilir
#Anlamasi ve uygulamasi kolaydir
#Training yoktur
#Distance metricleri degistirilirse farkli sonuclar elde edilebilir
#Hem classification hem de regression analizlerinde kullanilabilir

#Olumsuz yonleri:
#Buyuk datalarda kullanissiz(Tum noktalar tek tek hesaplandigi icin uzun surer
#Outlierlardan cok etkilenir(Dengesiz data setleri icin uygun degil)
#Overfit tehlikesinden dolayi k secimi cok onemli
#Scalling islemi zorunludur

#regression analizlerinde de yine komsularina bakar.Bunlarin ortalamasina bakarak
#sample i bir class a atar 
#classification=>Komsularin moduna bakar 
#regression=>komsularin ortalamasina bakar


#%% 
###########################NOTEBOOK 10#######################################
# Önceki dersin özeti
# Accuracy: Balanced datasetlerinde kullanıyor. Unbalanced da hangi sınıf çoğunluktayda onu daha çok yakalıyor
# Recall, Precision, F1: Unbalanced verisetlerinde bu metriklere bakılır.
# ROC/AUC: Her bir threshold dikkate alınarak çiziyor. Modelimiz sınıflandırmayı 
#ne derece düzgün yapıyor. Bunun hakkında bilgi veriyordu

# KNN Theory
# Veri etrafındaki noktalara bakarak hangi sınıfa ait olduğunu bulma mantığıyla çalışır.
# Distance-based modeldir. Bundan dolayı scaling yapmak gereklidir.

# KNN genel özellikleri
# Classification algoritması. Regression da yapıyor ancak biz classification modelini göreceğiz.
# Regression modeli tercih edilmiyor
# Eğitime ihtiyaç duymayan lazy learner bir modeldir. Her bir data noktasını hafızasına alır buna göre map leme yapar
# .. Yeni bir data noktası geldiğinde o data noktasının diğerlerine olan uzaklığını hesaplıyor
# Non-lineer ve non-parametrictir. katsayısı yoktur.Bir varsayımı yoktur

# 1-0 şeklinde sınıflandırmayı nasıl yapıyor?
# Örneğin yıldız gözlemini düşünelim, mavilere mi atacak, 
# turunculara mı atayacak bu noktayı
# k : komşu sayısı
# k seçimi önemli. En yakın kaç komşuya bakacağımızı belirliyoruz k ile.
# k nın seçiminin dışında burada mesafe de önemli.

# Alttaki problemi nasıl sınıflandıracağız bakalım bir sonraki slight a

# Slightlarda 3 farklı nokta için komşuluklarına bakmışız.
# Yıldız için
    # k=1 ise en yakınındaki değer turuncu renk olduğu için turuncu sınıfına atayacak 
    # k=3 olduğunda en yakındaki 3 komşuda(noktada) ağırlıklı olan sınıf mavi olduğu için maviye atadı
    # k=5 olduğunda en yakındaki 5 komşuda(noktada) ağırlıklı olan sınıf mavi olduğu için maviye atadı
# Çarpı için
    # k=1 ise en yakınındaki değer turuncu renk olduğu için turuncu sınıfına atayacak 
    # k=3 olduğunda en yakındaki 3 komşuda(noktada) ağırlıklı olan sınıf mavi olduğu için maviye atadı
    # k=5 olduğunda en yakındaki 5 komşuda(noktada) ağırlıklı olan sınıf mavi olduğu için maviye atadı
# Yeşil top için
    # Benzer mantıkta
    
# Özetle: k seçimine göre sınıflandırma sonuçlarımız değişebiliyor. k seçimi önemli

# k seçimi önemli dedik
# k yı büyük değer seçersek underfit (high bias). k arttıkça tranin ve validation hataları artıyor grafikte gördüğümüz gibi
# k yı küçük değer seçersek overfit oluyor (low bias)
# Ares Hoca: Alttaki grafiği aklımızda tutalım bu işin mantığını anlatıyor
# knn küçük datasetlerinde kullanılır. Çünkü büyük k değeri seçtiğimizde büyük hesaplama maliyetlerine katlanmak zorundayız

# K seçimi için 2 metod var
    # 1.elbow metodu
    # 2.grid search
# Error rate e bakacağız. Bu skor accuracy, precision, recall olabilir
# 1-accuracy değerini hesaplayacağız. Bu bizim error rate i olacak. Yani accuracy:0.92 ise hata rate imiz: 0.08 olacak
# Miminum hata oranını bulmamız gerekiyor
# Mesela burada k =35,37,38 de vs hata düşük görünüyor ama k yı büyük seçeriz bu da underfit e gitmeye sebep olur
# Ancak örneğin k=18 e baktığımızda hatadaki değişim çok az oluyor. O yüzden k=18 de hata biraz daha yüksek olmasına rağmen
# .. k=18 i k=36 ya tercih etmek daha mantıklıdır. Çünkü k=36 hesaplama maliyetimiz var
# Buradaki Bias-variance dengesini u sağlamalıyız
# Not olarak k=12 vs de seçilebilirdi. Bu değer de alınabilir. Farklı değerlerde alınabilir. Bunları deneyip karar vereceğiz

# Elbow da   : Kırılım(Dirsek) yerine bakarak yorumlayacağız. Uygulamada göreceğiz
# Grid search: Genelde hatanın minimum olduğu noktayı bulur. Seçimimiz bizim k=18 olacakken, grid search k=34 seçer
# .. kaynaklarda grid search ün bulduğu k değerinin kullanılmaması gerektiğini görürüz. Grafikte detaylı göreceğiz
# Genelde elbow metodu ile karar vereceğiz

# k haricinde bir diğer(ikinci) hiperparametremiz weight
# weight: modelin gözlemleri nasıl ağırlıklandıracağı
    # uniform: Bütün gözlemleri eşit sekilde ağırlıklandırır
        # k =5 olduğunda tüm değerler eşit ağırlıklı olursa hangi sınıf fazla olursa o sınıfa atama yapacak.
    # distance : Mesafeye göre ağırlıklandırma. Yakın gözleme daha büyük bir ağırlık veriyor
        # k=3 olduğunda bir değer yakın olsun noktamıza(0 a ait), iki değer uzak olsun(1 e ait)
        # .. 0 a ait olanın ağırlığı 1.8, 1 lere ait olan 0.7 ve 0.6 olursa bu noktayı 0 sınıfına atayacaktır
    # NOT: Modelden hangisinin daha iyi olduğu değişecektir

# 3. hyperparametriğimiz de Euclidian ya da manhattan
# Mesafeleri nasıl ölçeceğiz
# Default olarak model Minkowski yi kullanır
# p = 2 seçersek mesafe metriğimiz euclidian .  Dik üçgendeki(3-4-5 olsun) hipotenüsü hesaplayacak gibi düşünebiliriz (Yani 5)
# p = 1 seçersek mesafe metriğimiz Manhattan olacak . Dik üçgendeki kenarları topluyor gibi düşünebiliriz(Yani 3+4=7)

# Manhattan outlier lara karşı iyi mücadele eder mesafeyi daha fazla hesapladığı için
# Euclidian da outlier lara karşı o kadar hassas değil kuş bakışı mesafeye baktığı için

# 1. ders sonları 2. ders başları
# Distance-based model olduğu için scale yapmamız gerekiyor demiştik
# Bunu yapmazsak yanlış tahminlerle karşılaşırız
# Alttaki grafiklere bakalım. X1 ve X2 feature ları var
# Soldaki grafikte X1 feature ımız  -2 ye +2 aralığında gibi görünüyor. Range=4 birim. X1 in değişimi 4 birimlik alanda değişiyor
# .. X2 nin de -15 ile 50 arasında 65 birimlik bir değişim var. Örneğin siyah noktamızı düşünelim
# .. Acaba bu noktayı hangi noktaya atar? k=10 olsun. X1 in range i küçük olduğu için değerleri X1 e göre değerlendirecek X2 den ziyade
# .. ve buna göre bir sınıflandırma yapacak modelimiz. Bu da yanlış sınıflandırma yapmamıza sebep olur
# Sağdaki grafikte scale ettikten sonraki halini görüyoruz. Bu şekilde scale edilmiş halde sınıflandırma yapmamız daha doğru

# Avantajları
    # Bir varsayımı yoktu
    # Anlaması ve yorumlaması kolay bir model
    # Eğitime ihtiyaç duyulmuyor. "Map" leme yapıyor
    # Mesafeyi nasıl ağırlıklandırdığımız önemli
    # Regresyon ve Classification problemlerinde kullanılır

# Dezavantajları
    # Çok boyutlu veri setlerinde iyi çalışmıyor. Feature sayısının az olduğu datasetlerinde çalışıyor
    # Outlier lara ve dengesiz verisetlerine(çoğunluk olan sınıfa atama ihtimali yüksek) karşı hassas çünkü distance-based model olduğu için
    # k seçiminin dengesi önemli
    # Scale edilmesi gerekli bir model
        
# 1. Memory-based yaklaşım ile sınıflandırma yapar
# 2. Veri büyükçe hesaplama maliyetimiz artar
    # Yeni datanın BÜTÜN datalara olan uzaklığını hesaplayıp sonra k sayısına göre seçim yapıyor.
    # .. Bu yüzden hesaplama maliyeti büyük
# Sonuç olarak, 2 side doğru        
        
######## K-Nearest Neighbors(KNN)   
#Ares Hoca: KNN dediğimiz zaman aklımıza gelmesi gerekenler;
    # Ideal for small datasets
    # Scaling data is important
    # Distance based model        
        
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

df = pd.read_csv('gene_expression.csv')
df.head()
#   Gene One  Gene Two  Cancer Present
#0     4.300     3.900               1
#1     2.500     6.300               0
#2     5.700     3.900               1
#3     6.100     6.200               0
#4     7.400     3.400               1
# 1 ler kanser 0 lar kanser değil
# Kanserli olup olmadığını tahmin edeceğiz.

#Exploratory Data Analysis and Visualization

df.info() # Missing value yok, dtype lar nümerik

df.describe() # std>mean gibi bir durum yok. Outlier ımız yok diyebiliriz

df["Cancer Present"].value_counts()  # Balanced bir data görüyoruz
#1    1500
#0    1500
#Name: Cancer Present, dtype: int64
ax= sns.countplot(df["Cancer Present"]);
ax.bar_label(ax.containers[0])

sns.scatterplot(x='Gene One',y='Gene Two',hue='Cancer Present',data=df,alpha=0.7)
# Grafikte datanın bazı noktalarda iç içe girdiğini görüyoruz. Orada modelimiz yanlış tahminler yapacaktır
# Bunu bir alt grafikte daha yakın gözlemleyelim
#Yukaridaki grafikteki 2-6 ve 4-8 araliklarinda grift bir durum var. 
#Bu noktalara zoom yaparak baktik. Modelimizin bu noktalarda hata 
#Yapma olasiligi yuksek. KNN model, komsuluga bakarak class atamasi 
#yaptigi icin boyle grift datalarda cok iyi sonuc vermeyebilir 
# Üstteki grafikle belli bir noktaya yakınlaştırmış halini inceliyoruz
sns.scatterplot(x='Gene One',y='Gene Two',hue='Cancer Present',data=df, alpha=0.7, style= "Cancer Present")
plt.xlim(2,6)
plt.ylim(4,8)

sns.pairplot(data=df, hue="Cancer Present", height=4, aspect =1)
# Sınıfların iç içe girdiğini görüyoruz(Sol üst ve sağ alttaki grafiklerde kde grafiklerinde)

sns.boxplot(x= 'Cancer Present', y = 'Gene One', data=df)
# 0 ve 1 class ında outlier görünmüyor
# Ares hoca: Yorum olarak Gene one büyükse kanser olma durumu artıyor diyebiliriz

sns.boxplot(x= 'Cancer Present', y = 'Gene Two', data=df)
# Gene two data noktası daha büyükse kanser olmama durumu daha yükse diyebiliriz

sns.heatmap(df.corr(), annot=True);
# Multicolliniearity görünmüyor
# Gene one ın sayısal değeri arttıkça kanser olma ihtimali artıyor(0.55)
# Gene two için tersi denebilir(-0.69)
# Ufak insightlar çıkardık

from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Gene One'], df['Gene Two'], df['Cancer Present'],c=df['Cancer Present']);
# 3 feature varsa bu şekilde grafikler çizdirebiliriz
# 3 boyutlu olunca data noktalarının birbirinden nasıl ayrıldığını net olarak görüyoruz

# EDA aşamasından sonra modellemeye geçiyoruz

######### Train|Test Split and Scaling Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
X = df.drop('Cancer Present',axis=1)
y = df['Cancer Present']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

######### Modelling
#PARAMETERS
#N_NEIGHBORS ---> En yakindaki kac komsuya gore atama islemi yapilsin? 
#(Default=5) Binary modellerde k sayisini tek sayi secmek makuldur.
#
#WEIGHTS ---> Komsularin uzak ve yakin olmasina gore agirliklandirma islemi yapar. 
#2 cesidi vardir :
#
#'uniform' --> Butun class' larin oyu esittir. (Default deger)
#
#'distance' --> Yakin olan komsunun uzak olana ustunlugu vardir, daha fazla 
#agirliklandirilir. Bu yuzden bir komsu, diger komsudan agir oldugu icin ustunluk
# kazanabilir.

#METRIC ---> Mesafe hesabi yapan parametredir. Euclidean Distance(kus ucusu mesafe) 
#ve Manhattan Distance' in karisimi olan minkowski parametresi default degeridir.

#P ---> Minkowski parametresi icin kullanilan bir parametredir. p=1 secilirse
# Manhattan Distance gibi davranir; p=2 secilirse Euclidean Distance gibi davranir. (Default=2)

#Manhattan Distance, hatalari cezalandirdigi icin outlier' lar ile mucadele eder.
# Feature sayisi 5' ten fazla olan datasetleri icin uygundur.

#Euclidean Distance, outlier' lar ile mucadelede iyi degildir. Feature 
#sayisi 5' ten kucuk datasetleri icin uygundur.

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5) # k=5 seçtik. Normalde deafult
#değeri 5 zaten.
knn_model.fit(X_train_scaled,y_train)   
# Modelimizde burada train datamızı fit ediyoruz. Eğitim derken ütün data noktalarını map liyor aslında
#KNN egitim isleminde hicbir hesaplama yapmadan komusuluk sayisina gore sayisi
#en fazla olan class' a gozlemleri yerlestirir (Lazy learning). Cok buyuk 
#datalarda maliyetli bir modeldir. Cunku butun data icin gozlemler arasi 
#mesafeleri teker teker olcer. Bu yuzden daha cok kucuk datalarda tercih 
#edilen bir modeldir
y_pred = knn_model.predict(X_test_scaled)
y_pred
#predict_proba isleminde n_neighbors=5' e gore 5 sinifi sayar. Mesela 895. 
#sample' a bakarsak; bu sample' in cevresindeki 5 komsudan 2 tanesi 0 sinifina 
#ait, 3 tanesi ise 1 sinifina aitmis. Bu yuzden bu sample' i 1 sinifina atmis 
#(Bu atama islemi default deger olan 'uniform' a gore yapilmis. 
#Eger distance' a gore olsaydi komsulari agirlik degeri degisecegi icin 
#sonuclar da farklilik gosterecekti) :

y_pred_proba = knn_model.predict_proba(X_test_scaled)
# Acaba ne kadar bir olasılıkla yapmış tahminlerimizi bakıyoruz

pd.DataFrame(y_pred_proba)  # Örneğin 0. indexte 0 class ına 0.0 olasılıkla atamış, 1 class ına 1.0 olasılıkla atamış

my_dict = {"Actual": y_test, "Pred":y_pred, "Proba_1":y_pred_proba[:,1], "Proba_0":y_pred_proba[:,0]}
# 1 sınıfına ait olma ve 0 sınıfına ait olma olasılıkları alıyoruz bu adımda
# İlk satıra bakarsak Gerçek değeri 0 pred i 0 olarak tahmin etmiş 0.8 olasılıkla vs... 

# Class chat soru : Hocam probayı weighted distance  a göre mi belirliyor? --> Ares Hoca: Evet

pd.DataFrame.from_dict(my_dict).sample(10)
#      Actual  Pred  Proba_1  Proba_0
#1569       1     1    1.000    0.000
#2631       0     0    0.000    1.000
#2596       0     0    0.000    1.000
#1288       0     0    0.000    1.000
#63         0     0    0.200    0.800
#2706       1     1    1.000    0.000
#940        1     1    1.000    0.000
#2929       1     1    0.600    0.400
#2519       0     0    0.000    1.000
#1920       1     0    0.400    0.600
############ Model Performance on Classification Tasks
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
confusion_matrix(y_test, y_pred)
plot_confusion_matrix(knn_model, X_test_scaled, y_test);
# 439 ve 396 ımız bizim TRUE değerlerimizdi. 31 ve 34 False değerlerdi
# Modelimiz 31+34 = 65 tane gözlemi yanlış tahmin etmiş

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#[[439  31]
# [ 34 396]]
#              precision    recall  f1-score   support
#
#           0       0.93      0.93      0.93       470
#           1       0.93      0.92      0.92       430
#
#    accuracy                           0.93       900
#   macro avg       0.93      0.93      0.93       900
#weighted avg       0.93      0.93      0.93       900
# Balanced data seti olduğu için accuracy ye bakabiliriz.0.93
# Recall ve precision a bakmamıza gerek yok burada çünkü balanced bir data seti

y_train_pred = knn_model.predict(X_train_scaled)
print(confusion_matrix(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred))
#y_train_pred))
#[[ 972   58]
# [  61 1009]]
#              precision    recall  f1-score   support
#
#           0       0.94      0.94      0.94      1030
#           1       0.95      0.94      0.94      1070
#
#    accuracy                           0.94      2100
#   macro avg       0.94      0.94      0.94      2100
#weighted avg       0.94      0.94      0.94      2100
# Accuracy burada yüzde 94. Skorlar birbirine yakın. Overfit durumu görünmüyor

############ Elbow Method for Choosing Reasonable K Values
#Optimal k degerini bulmak icin Elbow metodunu veya GridSearch' u kullanmamiz gerekiyor :
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
test_error_rates = []
for k in range(1,30):
    knn_model = KNeighborsClassifier(n_neighbors=k) # k yı 1 den 30 a kadar değişecek
    knn_model.fit(X_train_scaled,y_train) 
    y_pred_test = knn_model.predict(X_test_scaled)   # farklı k lara göre tahminler alacak
    test_error = 1 - accuracy_score(y_test,y_pred_test) # balanced data olduğundan error olarak accuracy üzerinden hesaplama yapıyoruz
    test_error_rates.append(test_error)  # Bu hataları yukardaki listemize ekliyoruz   
# Class chat soru: Burada 30 değerine nasıl karar veriyoruz? --> Orion hoca: Deneme
# k yı arttırdığımızda model underfit e doğru gidecektir(Örneğin 30 yerine 300 yazarsak deneyebiliriz(Altta denendiğinde çıktıyı görebiliriz))

plt.figure(figsize=(15,8))
plt.plot(range(1,30), test_error_rates, color='blue', linestyle='--', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K_values')
plt.ylabel('Error Rate')
plt.hlines(y=0.050, xmin = 0, xmax = 30, colors= 'r', linestyles="--")
plt.hlines(y=0.057, xmin = 0, xmax = 30, colors= 'r', linestyles="--")
# Optimal k yı 9 seçebiliriz. Daha büyük değerlerde daha düşük değerler var (22 de mesela)
# .. Ancak hatalar arasındaki çok küçük bir değişim için 22 yi seçmek mantıklı olmayacaktır

# Class chat soru: hocam bu train errorları mı sadece? --> Orion Hoca: Test
    # Thread devamı: test ile traini karşılaştırarak yapacak herhalde devamında? --> Orion Hoca: Evet
#X ekseni, 1-30 arasi k degerleri, y ekseni her k' ye denk gelen error degerleri. 
#k=9' a gelen kisim ile k=22' ye gelen kisim arasinda cok az bir error farki var. 
#Bu kadar az bir error farki icin modelin comlexity' sini 9' dan 22' ye cikarmaya 
#deger mi bunu dusunmek gerekir. Bu islem ile extra hesaplama masrafi cikar :
############# Overfiting and underfiting control for k values
# 3. ders başı
# Train hatalarına da bakalım . Kodlar benzer
test_error_rates = []
train_error_rates = []
for k in range(1,30):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train_scaled,y_train) 
    y_pred_test = knn_model.predict(X_test_scaled)
    y_pred_train = knn_model.predict(X_train_scaled)
    test_error = 1 - accuracy_score(y_test,y_pred_test)
    train_error = 1 - accuracy_score(y_train,y_pred_train)
    test_error_rates.append(test_error)
    train_error_rates.append(train_error)

plt.figure(figsize=(15,8))
plt.plot(range(1,30), test_error_rates, color='blue', linestyle='--', marker='o',
         markerfacecolor='red', markersize=10)
plt.plot(range(1,30), train_error_rates, color='green', linestyle='--', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K_values')
plt.ylabel('Error Rate')
plt.hlines(y=0.050, xmin = 0, xmax = 30, colors= 'r', linestyles="--")
plt.hlines(y=0.057, xmin = 0, xmax = 30, colors= 'r', linestyles="--")
# Mavi noktalar train hataları
# Kırmızı noktalar test hataları
# Bunları birbirine yaklaştığı nokta 9 noktası gibi görünüyor. Burada da farklı bir bakış açısıyla insight elde ettik

########### Scores by Various K Values
def eval_metric(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    print("Test_Set\n")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print("Train_Set\n")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))

knn = KNeighborsClassifier(n_neighbors=1)  # Şimdi 1 komşuluğunda deneyelim train ve test skorlarıma bakalım
knn.fit(X_train_scaled,y_train)
print('WITH K=1\n')
eval_metric(knn, X_train_scaled, y_train, X_test_scaled, y_test)
# Train de accuracy 0.98  gelmiş , test te 0.89 olmuş accuracy . yani overfit olmuş k küçük iken

knn = KNeighborsClassifier(n_neighbors=22)  # k =22 de hatam düşüktü ama bakalım test ve train e
knn.fit(X_train_scaled,y_train)
print('WITH K=22\n')
eval_metric(knn, X_train_scaled, y_train, X_test_scaled, y_test)
# train de accuracy:0.93, test te 0.95. Değerler yakın. Tercih edilebilir ama bir de 9 değerine bakalım

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train_scaled,y_train)
print('WITH K=9\n')
eval_metric(knn, X_train_scaled, y_train, X_test_scaled, y_test)
# Train 0.94 e test 0.94 çıktı. Değerler birbirine çok yakın çıktı ve bunu k=9 yani düşük bir k değeriyle elde ettik
# 25+29= 54 tane hatalı tahmin yapmışız

knn = KNeighborsClassifier(n_neighbors=15)  # Bunu da deneyelim
knn.fit(X_train_scaled,y_train)
print('WITH K=15\n')
eval_metric(knn, X_train_scaled, y_train, X_test_scaled, y_test)
# Bu da tercih edilebilir. Train test skorları yakın
# Sonuç olarak k: 1,9,15,22 değerlerini denedik. 
# Elbow a göre en mantıklısı 9 olarak görünüyor
# Hesaplama maliyeti için 9 dan 22 ye çıkmaya gerek yok. Ancak 22 de makinanız güçlüyse tercih edilebilir

######### Cross Validate For Optimal K Value
from sklearn.model_selection import cross_val_score, cross_validate
model = KNeighborsClassifier(n_neighbors=9)
scores = cross_validate(model, X_train_scaled, y_train, scoring = ['accuracy', 'precision','recall',
                                                                   'f1'], cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores

df_scores.mean()[2:]
#test_accuracy    0.924
#test_precision   0.923
#test_recall      0.929
#test_f1          0.926
#dtype: float64
# Burada skorumuz 0.923810 çıkmış. k=9 iken 0.94 dü accuracy değeri. 
# Skorlar tutarlı görünüyor. Overfitting durumu da yok

######## Predict New Observation
# Model scaling yapıldığında ve yapılmadığındaki sonuçların farklılığını görelim
new_observation = [[3.5, 4.3]]
knn_model.predict(new_observation) # Scale yapılmadan aldığım tahmin de kanser(1 sınıfına ait) dedi modelimiz
#array([1], dtype=int64)
#Tahminimiz 1 class' ina atandi :
#Asagida predict proba sonuclarina gore %60 oraninda 1 class' inin ciktigini goruyoruz. Bu yuzden model ornegimiz icin 1 class' ini secti 
#knn_model.predict_proba(new_observation)
# Bu tahmini 0.65 olasılıkla yaptı
knn_model.predict_proba(new_observation)
#array([[0.34482759, 0.65517241]])
#Fakat yukaridaki islemimiz hatali. Cunku predict yapmak istedigimiz 
#sample' a scale islemi uygulamadik. Asagida ayni sample' a scale islemi
#uygulayarak modele verdigimizde 0 class' ina atama yapildigini goruyoruz.
#Data eger scale edildiyse predict edilecek data da scale edilmis olmali :
new_observation_scaled = scaler.transform(new_observation) # Scale edelim
new_observation_scaled
#array([[-1.1393583 , -0.62176572]])
knn_model.predict(new_observation_scaled)  # Scale yapıldıktan sonra aldığım tahmin de kanser değil(0 sınıfına ait) dedi modelimiz

knn_model.predict_proba(new_observation_scaled) # Bu tahmini 0.62 olasılıkla yaptı

######### Gridsearch Method for Choosing Reasonable K Values
# Elbow a baktık. Şimdi diğer method olan gridsearch e bakalım
from sklearn.model_selection import GridSearchCV
knn_grid = KNeighborsClassifier()
k_values= range(1,30)
param_grid = {"n_neighbors":k_values, "p": [1,2], "weights": ['uniform', "distance"]}
# En iyi parametreleri bulmak için paramatrelerimizde deneyeceği değerleri tanımlayalım
knn_grid_model = GridSearchCV(knn_grid, param_grid, cv=10, scoring= 'accuracy')
knn_grid_model.fit(X_train_scaled, y_train)
knn_grid_model.best_params_  
#{'n_neighbors': 21, 'p': 1, 'weights': 'uniform'}
# k=21 , p:2(eucledian) , weight: uniform(noktalara eşit ağırlıklar versin)
# Burada grid search ün verdiği değer genelde tercih edilmez
#GridSearch islemi en iyi degeri k=21 olarak secti. GridSearch en dusuk 
#error' u veren k degerini secer. Elbow metodda ise error' lara gore kendimiz 
#bir k degeri secebiliyorduk. Buyuk k degeri maliyet olarak geri donecegi icin
#Elbow metodu ile mutlaka error' lara bakip sonuca ona gore karar vermek gerekir.
#k=9 ile k=21 arasindaki skorlarda cok da bir fark yok. Bu yuzden k=9 degeri 
#ile modelimizi kurmaya karar verdik.
print('WITH K=21\n')
eval_metric(knn_grid_model, X_train_scaled, y_train, X_test_scaled, y_test)
# n_neighbors=9,      test_accuracy: 94  with 54 error
# n_neighbors=21,    test_accuracy: 94   with 50 error
# Tercih noktasında sizden ne istenildiğine göre karar verilebilir
# Ancak hesaplama maliyetine göre 9 u tercih etmek daha mantıklı görünüyor

######### Evaluating ROC Curves and AUC
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve, plot_roc_curve, roc_auc_score, roc_curve
knn_model = KNeighborsClassifier(n_neighbors=9).fit(X_train_scaled, y_train) # k=9 a göre ...
plot_roc_curve(knn_model, X_test_scaled, y_test)

y_pred_proba = knn_model.predict_proba(X_test_scaled)
roc_auc_score(y_test, y_pred_proba[:,1])
# False positive rate lerin minimum olmasını istiyoruz
# True positive rate i de maximize etmek istiyoruz
# AUC: Yaklaşık 0.98 .. Kanser olma ve olmama durumunu yüzde 98 olarak başarılı yapıyor diyebiliriz
#Elbow metodu baz alarak k=9 degerini sectik ve modelimizi kurduk. 
#Datamiz dengeli bir dataseti oldugu icin ROC/AUC grafigini cizdirdik. 
#Modelimizin genel performansi %98 :
########## Final Model and Model Deployment
import pickle
scaler = StandardScaler()
scaler.fit(X)  # Tüm datayı fit ediyoruz. Dikkat Fit_transform yapmıyoruz.
X_scaled = scaler.transform(X)
final_knn_model = KNeighborsClassifier(n_neighbors=9).fit(X_scaled,y)
pickle.dump(final_knn_model, open('knn_final.pkl', 'wb')) # wb: white binary(Daha az yer kaplaması için) 
# Modelimizi localimize kaydettik
pickle.dump(scaler, open('scaler_knn.pkl', 'wb')) # Yukarıda tanımladığımız scaler ı da kaydediyoruz

########## Predict New Observations
# Şimdi kullanmak istediğimizi düşenerek tekrar çağıralım kaydettiklerimizi
loaded_scaler = pickle.load(open('scaler_knn.pkl', 'rb')) 
loaded_model = pickle.load(open('knn_final.pkl', 'rb'))
X.columns
X.describe()

# Yapay gözlem oluşturalım
new_obs = {"Gene One": [1, 3, 4.3, 5.6, 7, 9.5, 2, 6], 
           "Gene Two": [1, 4, 4, 5.5, 6.7, 10, 8, 1]
          }

samples = pd.DataFrame(new_obs)
samples

samples_scaled = loaded_scaler.transform(samples) # Buradaki datamı dönüştürüyoruz. Localimize kaydettiğimiz scaler ile yapıyoruz bunu
samples_scaled

predictions = loaded_model.predict(samples_scaled) # k = 9 iken scale edilmiş tahminlerimizi alıyoruz
predictions_proba = loaded_model.predict_proba(samples_scaled) # Burada da olasılıklarımızı

samples["pred"] = predictions
samples["pred_proba_1"] = predictions_proba[:,1]  # 1 sınıfına ait olasılıklar
samples["pred_proba_0"] = predictions_proba[:,0]  # 0 sınıfına ait olasılıklar
samples
#   Gene One  Gene Two  pred  pred_proba_1  pred_proba_0
#0     1.000     1.000     0         0.000         1.000
#1     3.000     4.000     0         0.111         0.889
#2     4.300     4.000     1         1.000         0.000
#3     5.600     5.500     0         0.222         0.778
#4     7.000     6.700     0         0.000         1.000
#5     9.500    10.000     1         0.667         0.333
#6     2.000     8.000     0         0.000         1.000
#7     6.000     1.000     1         1.000         0.000
########## Pipeline
#What happens can be described as follows:

#Step 1: The data are split into TRAINING data and TEST data according to ratio 
#of train_test_split

#Step 2: the scaler is fitted on the TRAINING data

#Step 3: the scaler transforms TRAINING data

#Step 4: the models are fitted/trained using the transformed TRAINING data

#Step 5: the scaler is used to transform the TEST data

#Step 6: the trained models predict using the transformed TEST data

#Pipeline -----> fit ve transform ile yapilan islemleri siralandirir, optimize eder.

#Yani; scale, egitim ve tahmin islemlerinin hepsini tek bir kodla yapar.

#Yaptigimiz en buyuk hatalardan biri; X_train'i scale ettikten sonra ilerleyen
#islemlerde bunu unutup scale edilmemis datayi kullanmak. Pipeline bu sorunu
#ortadan kaldiriyor. Scale edilmesi gereken kisimlari scale eder, modele sokulmasi
#gereken kisimlari modele sokar.
#!!!!!!!! Pipeline, butun ML modellerinde kullanilabilir. !!!!!!!!!

# pipe.fit(X_train, y_train)--> scaler.fit_transform(X_train) --> knn.fit(scaled_X_train, y_train)
# pipe.predict(X_test) --> scaler.transform(X_test) --> knn.predict(scaled_X_test)


from sklearn.pipeline import Pipeline
operations = [("scaler", StandardScaler()), ("knn", KNeighborsClassifier())] # Bunların sırası önemli. Önce scaling sonra modelleme yapacak
# Data leakage olmaması için traine fit_transform yapıp, test e sadece transform yapacak. O adımı halledecek
# Not olarak KNN burada default değerle yani 5 değeriyle devam ediyoruz burada pipeline ı göstermek adına
Pipeline(steps=operations)
#steps --------> Ben islemleri otomize edecegim ama hangi sirayla yapayim diye soruyor.
#Bu yuzden yukarida operation diye bir degisken tanimladik.
#operation icindekileri koseli parantezle belirtmek zorundayiz.
#'scaler', StandardScaler() ----------> scaler islemi yapacagim, StandardScaler() ile.
#'knn', KNeighborsClassifier() ------------> knn modelini kullanacagim.
#!!!!! operation icine yazdigimiz ilk islemler mutlaka fit ve transform yapan 
#islemler olmak zorunda. !!!!!!!
#!!!!! operation icine yazdigimiz ikinci islem mutlaka fit ve predict i
#slemi yapan algoritmalar, yani ML algoritmalari olmak zorunda. 
#(Sadece 1 tane model ismi yazilir. Birden fazlasi yazilamaz) !!!!!!
#Pipeline' ni bir degiskene atadigimizda bu pipeline hem bir scaler gibi hem de algoritma gibi davranmaya basliyor.
pipe_model = Pipeline(steps=operations)
#Bu degiskeni olusturduktan sonra fit islemini uyguladigimizda pipeline sunu yapar :
#pipe.fit(X_train, y_train)--> scaler.fit_transform(X_train) 
#--> knn.fit(scaled_X_train, y_train)
#Yani, once scale islemini sadece X_train'e uygular, sonra modeli kurar
#datayi X_train ve y_train'i egitir :
#Pipeline(steps=[('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])


pipe_model.fit(X_train, y_train) 
# Eğitimi yaptık 
#Egittigimiz pipeline'a predict islemini uyguladigimizda sunu yapar :
#pipe.predict(X_test) --> scaler.transform(X_test) --> knn.predict(scaled_X_test)
#Yani; X_test'i scale eder, sonra tahminlerini alir. 
#Yani artik X_testi scale ettim mi korkusu ortadan kalkar :
y_pred = pipe_model.predict(X_test)
y_pred

########## Model Performance
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
plot_confusion_matrix(pipe_model, X_test, y_test);
#Artik her yere model ismi olarak pipe_model' i verebiliriz.

print(classification_report(y_test, y_pred))

eval_metric(pipe_model, X_train, y_train, X_test, y_test)
# 65 errors


########## Changing the parameters of the pipe_model
pipe_model.get_params() 
# Burada nasıl yazıldı ise pipe_model.set_params kısmında o şekilde yazmalıyız
#pipe_model icindeki parametreleri degistirmek istersek ne yapacagiz?
#Yukarida pipeline parametrelerine bakarsak; pipe_model icine tanimlanan 
#scaler ve modele sunu yapmis :
#operation icine tanimladigimiz 'scaler' ismini yazip iki tane alt cizgi 
#koymus ve sonra parametreleri yazmis (scaler__copy': True gibi)

#operation icine tanimladigimiz model olan 'knn' yi yazip iki tane alt 
#cizgi koymus ve sonra parametreleri yazmis (knn__n_neighbors': 5 gibi)

#Biz bu default degerleri degistirmek istersek sunu yapacagiz :
pipe_model.set_params(knn__n_neighbors= 9) 
# Parametreyi değiştiriyoruz burada. 5 i 9 ile değiştirdik
pipe_model.get_params() 
# Parametrenin değiştiğini görüyoruz (knn__n_neighbors': 9 kısmında)
pipe_model['scaler']        
# Datanin scale edilmis hali. Fit edilmeye hazir.

pipe_model["knn"]           
# Datanin egitilmis hali. Predict yapmaya hazir

############## GridSearch on Pipeline
from sklearn.model_selection import GridSearchCV
#Pipeline ile olusturulmus data GridSearch islemine tabi tutulacaksa, 
#GridSearch icine verilecek olan araliklarin pipe_model 
#parametrelerine uygun sekilde verilmesi gerekir, yani knn__n_neighbors gibi.
param_grid = {'knn__n_neighbors': range(1,30)} 
pipe_model = Pipeline(steps=operations)     
# Her yeni islemde modeli mutlaka sifirliyoruz.
pipe_grid = GridSearchCV(pipe_model, param_grid, cv=10, scoring= 'f1')
pipe_grid.fit(X_train,y_train) # Eğitim
pipe_grid.best_params_
#{'knn__n_neighbors': 21}
############## CrossValidate on Pipeline
# CV öncesi modeli sıfırlamamız lazım scaler ımızı tanımladık, n_neigbors=9 u tanımladık tekrar 
#CrossValidate isleminde data her seferinde 10 parcaya bolunup 
#bir tanesi test datasi olarak ayrilir. Ama icerdeki bu test datasi da scale edilmis oldugu icin test datasi kopye ceker (Data Leakage). 
#PipeLine kullandigimizda bunu onmelis oluruz.
operations = [('scaler',StandardScaler()),('knn',KNeighborsClassifier(n_neighbors=9))]
model = Pipeline(operations)
scores = cross_validate(model, X_train, y_train, scoring = ['precision','recall','f1','accuracy'], cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
#test_precision   0.923
#test_recall      0.931
#test_f1          0.927
#test_accuracy    0.925
# k=9 iken sonuçları görmüştük yukarda. Bunu pipeline ile de görmüş olduk
# Class chat soru: pipelineda parametrelerle oynamadan yukarıda değşkene atadığımız en iyi modeli kullanamaz mıyız?
# Orion: Tekrar aynı parametreleri yazıp eğitmeniz lazım ([('scaler',StandardScaler()),('knn',KNeighborsClassifier(n_neighbors=9))])

########## Final pipe_model
operations = [('scaler',StandardScaler()),('knn',KNeighborsClassifier(n_neighbors=9))]
pipe_final = Pipeline(operations)
pipe_final.fit(X, y)

########## Predict New Observations with pipe_model
new_obs = {"Gene One": [1, 3, 4.3, 5.6, 7, 9.5, 2, 6],
           "Gene Two": [1, 4, 4, 5.5, 6.7, 10, 8, 1]
          }
samples = pd.DataFrame(new_obs)
samples
#Pipeline' dan once predict kisminda scale islemini yapmayi unutabiliyorduk. 
#Bu sorun da burda ortadan kalkiyor cunku pipeline bizim yerimize scale islemini otomatik yapiyor
predictions = pipe_final.predict(samples)
predictions

predictions_proba = pipe_final.predict_proba(samples)
predictions_proba

samples["pred"] = predictions
samples["pred_proba"] = predictions_proba[:,1]
samples

#################################################################################################################################
#%% LAB-3
# Data Set Information:
# Images of Kecimen and Besni raisin varieties grown in Turkey were obtained with CVS. A total of 900 raisin grains were used, including 450 pieces from both varieties. These images were subjected to various stages of pre-processing and 7 morphological features were extracted. These features have been classified using three different artificial intelligence techniques.
# Attribute Information:
# 1. Area: Gives the number of pixels within the boundaries of the raisin.
# 2. Perimeter: It measures the environment by calculating the distance between the boundaries of the raisin and the pixels around it.
# 3. MajorAxisLength: Gives the length of the main axis, which is the longest line that can be drawn on the raisin.
# 4. MinorAxisLength: Gives the length of the small axis, which is the shortest line that can be drawn on the raisin.
# 5. Eccentricity: It gives a measure of the eccentricity of the ellipse, which has the same moments as raisins.
# 6. ConvexArea: Gives the number of pixels of the smallest convex shell of the region formed by the raisin.
# 7. Extent: Gives the ratio of the region formed by the raisin to the total pixels in the bounding box.
# 8. Class: Kecimen and Besni raisin.
# 
# https://archive.ics.uci.edu/ml/datasets/Raisin+Dataset

# # Import libraries
# libraries for EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import cufflinks as cf
#Enabling the offline mode for interactive plotting locally
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
cf.go_offline()
import plotly.io as pio
pio.renderers.default = "colab"
#To display the plots
get_ipython().run_line_magic('matplotlib', 'inline')

# sklearn library for machine learning algorithms, data preprocessing, and evaluation
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss, recall_score, accuracy_score, precision_score, f1_score

# yellowbrick library for visualizing the model performance
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.cluster import KElbowVisualizer 

from sklearn.pipeline import Pipeline
# to get rid of the warnings
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

# !pip install cufflinks
# !pip install plotly

### Exploratory Data Analysis and Visualization
df = pd.read_excel("Raisin_Dataset.xlsx") # Reading the dataset
df.head() 
# Türkiye de yetiştirilen Keçimen ve Besni kuru üzümlerinin 7 tane özelliği verilmiş
# Bu üzümleri yapısal özelliklerine göre sınıflandıracağız
df.info() # Missing value yok. Dtype lar nümerik. Data cleaning yapmayacağız
df.shape
df.duplicated().sum()   # No duplicates
df.isnull().sum().any() # No missing values
df.describe()    # Std > mean olduğu durum yok gibi görünüyor şu an için

ax= sns.countplot(df["Class"]);
ax.bar_label(ax.containers[0]) # to show the proportion of each class
# Balanced bir data seti var elimizde.(Yani değerlendirmek için accuracy yi kullanabiliriz)
# We have prety same amout of classes in the data set. So I can use accuracy as a metric to evaluate the performance of the classifier.

df["Class"] = df["Class"].map({"Kecimen":0,"Besni":1}) # mapping the classes to 0 and 1
# Keçimen ve Besniyi sayısal değerlere çevirelim. Map ledik

df.iloc[:,:-1].iplot(kind="box") # Tek alanda tüm boxplotları gördük plotly ile

# Sınıf bazında bakalım
fig = px.box(df, color="Class", color_discrete_map={"Kecimen":'#FF0000',"Besni":'#00FF00'})
fig.show()

df.iplot(kind="bar")
# Data ilk 450 satır Keçimen e ait, sonraki 450 si Besni olarak sınıflandırılmış
# Barplot a baktığımız zaman 450 den sonrasında areası büyükse besni(1) sınıfına ait diyebiliriz
# MajoraxisLength için aynı yorumu yapabiliriz
# MinorAxisLength için tam olarak ayırt edici diyemeyebiliriz
# Eccentricity için de tam olarak ayırt edici diyemeyiz
# ..
# Extent için de tam olarak ayırt edici diyemeyiz
# Perimeter için büyük olursa besni sınıfına ait olduğunu söyleyebiliriz

fig = px.bar(df,x=df.index,y="Area",color="Class",color_discrete_map={"Kecimen":'#FF0000',"Besni":'#00FF00'})
fig.show()
# Area yı inceledik class ile durumunu

plt.figure(figsize=(10,8))
sns.heatmap(df.select_dtypes(include='number').corr(),vmin=-1,vmax=1, annot=True, cmap='coolwarm')
# Ciddi korelasyonlar görünüyor
# Örneğin: Area ile Perimeter , Perimeter ile MajorAxislength
# Multicollinearity olduğunu görüyoruz burada
# Multicollinearity da baskın olan feature diğer feature ı ezmiş oluyor. Baskılıyor bir nevi
# Bu sorunu bizim modelimiz çözecek

corr_matrix = df.corr()
fig = px.imshow(corr_matrix)  # px: plotly.express kütüphanesinden
fig.show()
# Bu çıktıdan da çeşitli insightlar elde edebiliriz
# Yukardaki heatmap e alternatif olarak kullanabiliriz

sns.pairplot(df, hue = "Class")
# Sınıfların genelde iç içe girdiğini görüyoruz
# Yani datamın birbirinden düzgün bir şekilde ayrılmadığını görüyoruz

fig = px.scatter_3d(df, x='Perimeter', y='Area', z='Extent',
              color='Class')
fig.show()
# Grafiği tutup çevirebiliriz
# İç içe giren noktalar olduğunu görüyoruz.
# Bu da faydalanabileceğimiz başka tür bir grafik.
# EDA aşamasından sonra modellemeye geçiş yapabiliriz

### Train | Test Split and Scaling
X=df.drop(["Class"], axis=1)
y=df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=10)

scaler =StandardScaler() # will be used in the pipelines
log_model = LogisticRegression() # will be used in the pipelines

log_pipe = Pipeline([("scaler",scaler),("log_model",log_model)]) # pipeline for logistic regression
# Operationlarımızı yazıyoruz SIRAYLA "scaler" ve "log_model"
log_pipe.fit(X_train, y_train) # Bunun içinde scaling ve eğitim yapılıyor

y_pred=log_pipe.predict(X_test)  # X_test scale i kendisi oluşturmuş olacak ve tahmin alacak
y_pred_proba = log_pipe.predict_proba(X_test) # Ne kadar olasılıkla tahmin ettik bunları aldık

### Model Performance
def eval_metric(model, X_train, y_train, X_test, y_test):
    """ to get the metrics for the model """
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    print("Test_Set")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print("Train_Set")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))

eval_metric(log_pipe, X_train, y_train, X_test, y_test) # to get the metrics for the model
# log_pipe: Pipeline ile oluşturduğumuz modelimiz
# Balanced data setinde accuracy ye bakıp devam edebiliriz
# 0.87 test ve 0.87 train. Overfit görünmüyor
# Bu skorları CV ile kontrol edelim

#### Cross Validate
model = Pipeline([("scaler",scaler),("log_model",log_model)]) # Modeli tekrardan tanımlıyorduk CV de
scores = cross_validate(model, X_train, y_train, scoring = ['precision','recall','f1','accuracy'], cv = 10,error_score="raise")
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores
df_scores.mean()[2:]      # Accuracy: 0.86 # Alttaki çıktı ile karşılaştırdığımızda değerler yakın görünüyoruz. Overfit durumu yok

eval_metric(log_pipe, X_train, y_train, X_test, y_test)

##### GridSearchCV
from sklearn.model_selection import GridSearchCV
# pipeline for logistic regression
model = Pipeline([("scaler",scaler),("log_model",log_model)]) # Grid search öncesi oluşturuyoruz
# l1: Lasso, l2: Ridge
penalty = ["l1", "l2"]                     # Multicollinearity yi çözmek için 
# to get 20 values of C between -1 and 5
C = np.logspace(-1, 5, 20)                     # Hyperparameter
# balanced: class weights are balanced, None: no class weights
class_weight= ["balanced", None]               # Hyperparameter
# to get 4 values of solver
solver = ["lbfgs", "liblinear", "sag", "saga"]    # Optimize etmek için
# to get all the combinations of penalty, C, class_weight and solver
param_grid = {"log_model__penalty" : penalty,
              "log_model__C" : [C,1],
              "log_model__class_weight":class_weight,
              "log_model__solver":solver} 

# to get the best model
grid_model = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          cv=10,
                          scoring = 'accuracy',       
                          n_jobs = -1) 

grid_model.fit(X_train,y_train) # Eğitim yaptı ve en iyi hyperparametreler üzerinde
grid_model.best_params_ # to get the best parameters according to the best score

eval_metric(grid_model, X_train, y_train, X_test, y_test)  
# test set accuracy increased 0.87 to 0.88
# En iyi hyperparametrelerimiz ile sonuçlarımız
# Test scorum 0.88.. 1 puan iyileşmiş oldu

#### ROC (Receiver Operating Curve) and AUC (Area Under Curve)
# Bunların genel performansını görmek için roc çizdirelim
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve, roc_auc_score, auc, roc_curve, average_precision_score, precision_recall_curve
plot_roc_curve(grid_model, X_test, y_test) # we use ROC curve to get the AUC score and evaluate the model if it is good or not on every threshold
# Yorum Modelim %93 oranında keçimen ve besni sınıflarını ayrıştırabiliyor

plot_roc_curve(log_pipe, X_test, y_test)  # Eski modelim(Parametreleriyle oynamadığımız)
# ROC larda bir değişiklik olmadı
# log modeldeki başarımız yüzde 88 bir de KNN e bakalım

########## KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()  # to get a object of KNeighborsClassifier for pipeline 
# default k =5 olarak tanımladık
knn_pipe = Pipeline([("scaler",scaler),("knn",knn)]) # pipeline for KNeighborsClassifier
knn_pipe.fit(X_train, y_train)  # Scale ve eğitim yaptı
knn_pred = knn_pipe.predict(X_test)
eval_metric(knn_pipe, X_train, y_train, X_test, y_test)
# Test accuracy 0.86, train accuracy.  0.88 k = 5 iken başarım %86. 10+16=26 tane yanlış tahmin

##### Elbow Method for Choosing Reasonable K Values
# Optimal k yı bulmak için kullanıyorduk
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
test_error_rates = []
for k in range(1,30):
    model = Pipeline([("scaler",scaler),("knn",KNeighborsClassifier(n_neighbors=k))]) # p=1,weights="uniform",metric="minkowski"
    scores = cross_validate(model, X_train, y_train, scoring = ['accuracy'], cv = 10,error_score="raise")
    accuracy_mean = scores["test_accuracy"].mean() 
    test_error = 1 - accuracy_mean 
    test_error_rates.append(test_error)
# 1 den 30 a kadar k değerlerini deneyeceğiz
# Skorların tutarlı olması açısında cross_validate kullanıyoruz
# Tek seferlik aldığımız skordan ziyade 10 katlı Cv ile test_error hesaplamış olduk

# Dün yaptığımız gibi tek seferlik skorlara bakacağız bir de(Yukardaki cv 10 katlı skor, burası tek seferlik skor)
test_error_rates1 = []
for k in range(1,30):
    knn_model = Pipeline([("scaler",scaler),("knn",KNeighborsClassifier(n_neighbors=k))])
    knn_model.fit(X_train,y_train) 
    y_pred_test = knn_model.predict(X_test)
    test_error = 1 - accuracy_score(y_test,y_pred_test)
    test_error_rates1.append(test_error)

# Üstteki oluşturduğumuz listelerin grafiklerini çizdiriyoruz
plt.figure(figsize=(15,8))
plt.plot(range(1,30), test_error_rates1, color='blue', linestyle='--', marker='o',
         markerfacecolor='red', markersize=10)
plt.plot(range(1,30), test_error_rates, color='black', linestyle='-', marker='X',
         markerfacecolor='green', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K_values')
plt.ylabel('Error Rate')

# Cv ile yapılan daha tutarlı görünüyor
# Minimum hata olan 26-27 yi seçebiliriz. Ama maliyet artabilir
# 5,12,13,23 vs seçilebilir
# Biz 5 i seçeceğiz. Hata daha yüksek ama maliyet az(Ancak biz yine denemeler yapacağız altta)

########### Scores by Various K Values
knn = Pipeline([("scaler",scaler),("knn",KNeighborsClassifier(n_neighbors=2))])
knn.fit(X_train,y_train)
print('WITH K=2\n')
eval_metric(knn, X_train, y_train, X_test, y_test)
# k=2 için deneme yaptık
# Test accuracy 0.84, train accuracy 0.92 .. Model overfit olmuş oldu

knn = Pipeline([("scaler",scaler),("knn",KNeighborsClassifier(n_neighbors=8))])
knn.fit(X_train,y_train)
print('WITH K=8\n')
eval_metric(knn, X_train, y_train, X_test, y_test)
# k=8 için deneme yaptık
# Test accuracy 0.83, train accuracy 0.88 .. 
# Ares Hoca: Skorlar birbirine yakın gibi. Overfit olma ihtimali var..

knn = Pipeline([("scaler",scaler),("knn",KNeighborsClassifier(n_neighbors=25))])
knn.fit(X_train,y_train)
print('WITH 25K=\n')
eval_metric(knn, X_train, y_train, X_test, y_test)
# k=25 için deneme yaptık
# Test accuracy 0.86, train accuracy 0.87 .. Skorlar yakın tutarlı model. Ancak iyileşme için bu kadar
# .. büyük k tercih edilir mi ? ...

knn = Pipeline([("scaler",scaler),("knn",KNeighborsClassifier(n_neighbors=5))])   
knn.fit(X_train,y_train)
print('WITH 5K=\n')
eval_metric(knn, X_train, y_train, X_test, y_test)
# k=5 için
# Test accuracy 0.86, train accuracy 0.88 . Model 10+16=26 tane hata yapıyor
# Elbow da k=5 seçmeye karar verdik.
# Peki grid search ne diyecek k için bakacağız
# Ondan önce Cross validation yapalım(k=5 iken)


######### Cross Validate
model =Pipeline([("scaler",scaler),("knn",KNeighborsClassifier(n_neighbors=5))])
scores = cross_validate(model, X_train, y_train, scoring = ['precision','recall','f1','accuracy'], cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores
df_scores.mean()[2:]         # test Accuracy 0.85 .. Normalde 0.86 idi. Skorlar tutarlı

knn = Pipeline([("scaler",scaler),("knn",KNeighborsClassifier(n_neighbors=5))])   # test_accuracy:    0.86 , train_accuracy :  0.88
                                                                                  # test_accuracy     0.85  (cross validation)
                                                                                  # (k=5 with elbow) with 26 wrong prediction
knn.fit(X_train,y_train)
print('WITH K=5\n')
eval_metric(knn, X_train, y_train, X_test, y_test)  

######### Gridsearch Method for Choosing Reasonable K Values
knn.get_params()
# pipeline for KNeighborsClassifier
knn_grid = Pipeline([("scaler",scaler),("knn",KNeighborsClassifier())]) 
# to get all the values of k between 1 and 30
k_values= range(1,30)                  
# to get the values of weight
weight = ['uniform', 'distance']       # hyperparameter
# to get the values of p
p = [1,2]                              # hyperparameter
# to get the values of metric
metric = ['minkowski']                 # minkowski ye göre p seçimi yapılacak  
# to get all the combinations of k, weight, p and metric
param_grid = {'knn__n_neighbors': k_values,
              'knn__weights': weight, 
              'knn__p': p, 
              'knn__metric': metric} 
# to get the best model according to the best score
knn_grid_model = GridSearchCV(estimator= knn_grid, 
                             param_grid=param_grid,
                             cv=10, 
                             scoring= 'accuracy',
                             n_jobs=-1) 

knn_grid_model.fit(X_train, y_train)
knn_grid_model.best_params_ # to get the best parameters according to the best score
# 'uniform' : Hepsini eşit ağırlıklandırdı
# k=14. Yukardaki grafikte 14 de hata en düşük değildi çünkü farklı parametreler ile yaptık
# Burada parametreler değişti

test_error_rates2 = []
for k in range(1,30):
    model = Pipeline([("scaler",scaler),("knn",KNeighborsClassifier(n_neighbors=k, p=1))]) # p=1,weights="uniform",metric="minkowski"
    scores = cross_validate(model, X_train, y_train, scoring = ['accuracy'], cv = 10,error_score="raise")
    accuracy_mean = scores["test_accuracy"].mean() 
    test_error = 1 - accuracy_mean 
    test_error_rates.append(test_error)

plt.figure(figsize=(15,8))
plt.plot(range(1,30), test_error_rates1, color='blue', linestyle='--', marker='o',
         markerfacecolor='red', markersize=10)
plt.plot(range(1,30), test_error_rates, color='black', linestyle='-', marker='X',
         markerfacecolor='green', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K_values')
plt.ylabel('Error Rate')
# 14 te minimum olduğunu görüyoruz

print('WITH K=14\n')      #  knn      test_accuracy :   0.85  (k=14 with gridsearch) with 27 wrong prediction

                          #  knn      test_accuracy :   0.86  (k=5 with elbow) with 26 wrong prediction
eval_metric(knn_grid_model, X_train, y_train, X_test, y_test)
# Skorlar birbirine yakın ama tercihen k=5 seçiyoruz(Computational cost açısından)

############# Evaluating ROC Curves and AUC
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve, plot_roc_curve, roc_auc_score, roc_curve
model = KNeighborsClassifier(n_neighbors=14, p=1, metric="minkowski", weights="uniform") # best gridsearch model 
knn_model = Pipeline([("scaler",scaler),("knn",model)])
knn_model.fit(X_train, y_train)
# Bu kodları roc curve çizdirmek için yaztık(grid search deki sonuca göre)

# 0.85  (k=14 with gridsearch) with 27 wrong prediction
plot_roc_curve(knn_model, X_test, y_test) 

y_pred_proba = knn_model.predict_proba(X_test)
roc_auc_score(y_test, y_pred_proba[:,1])

model = KNeighborsClassifier(n_neighbors=5) # best elbow model
knn_model = Pipeline([("scaler",scaler),("knn",model)])
knn_model.fit(X_train, y_train)
# Bu kodları roc curve çizdirmek için yaztık(elbow daki sonuca göre)
#   knn test_accuracy :  0.85         (k=14 with gridsearch)       with 27 wrong prediction
#   knn test_accuracy :   0.86        (k=5 with elbow)             with 26 wrong prediction
plot_roc_curve(knn_model, X_test, y_test)

# k=14 iken Roc daha iyi buradaki kısım tercih meselesi
# k=5 ve k=14 .. ikiside doğru model
# Ares Hoca: Ben 5 komşuluğu tercih ediyorum

# Log_modeli kurarken çok uğraşmadık
# KNN de uğraştık, k değerlerini denedik parametreleri denedik, elbow ve grid search e baktık vs...
# Bundan dolayı;
# Bu data için tercih olarak KNN ve log_model arasında log_modeli tercih ederiz
# KNN tercih edersem de k=5 derim
# logistic regression daha hızlı daha az maliyetli

# class chat soru: gridsearhcv de en iyi k değeri 14 çıktı yani accuracy nin daha yüksek olması beklenmez miydi? neden k=5 de accuracy daha yüksek çıktı
# grid search de bir çok parametre ile oynadık. Skorlar iyi çıkmadı
# Orion hoca:Trainden güzel bir skor alınabilir ama hold-out test setten iyi bir skor almamız beklenir ama çıkmayabilir.
# .. ki neticede burada sonuç öyle olmuş(iyi skor olmamış anlamında)

# # Conclusion 
# * log_model Accuracy Score: 0.88 
# * log_model AUC : 0.93           
# * knn Accuracy Score :   0.86  (k=5 with elbow)  - 0.85  (k=14 with gridsearch)
# * knn AUC : 0.88 (elbow) - 0.90 (gridsearch)
# * As a conclusion we aplied two models to predict raisins classes and we got prety decent scores both of them
# * We decided to use the Logistic Model because of its slightly better score than the knn models, 
#plus the interpretability of logistic regression and its lower computational cost.




#%%
##########LESSON 11

##########Support Vector Machines Model(SVM)
#text,image recognition,image_based gender detection gibi classification sorunlarında
#kullanlır
#Linear olarak ayrılabilecek datalarda mükemmel sonuçlar verir 

#       |
#feature|      x      o  o        *iki class arasında sonsuz çizgi çizilebilir
#    1  |    x xx   o o o         *ama ben öyle 2 support seçeyim ki,ortalarından geçen 
#       |   x x x    o          çizgi iki support a da en uzak olsun  
#       |    x x     o  o
#       |____________________
#          feature 2

#Margin=>Seçilen supportlar arası uzaklık
#3 boyutlu olduğunda ise,araya line yerine hyperline çizer.Classları birbirinden 
#ayırır ve 3 boyutlu bir görüntü sağlar 

#Hard Margin-Soft Margin

#       |
#feature|      x      o  o        Hard Margin 
#    1  |    x xx .  o o o         Classları tam ayırmak için böyle bir çizgi çizerse 
#       |   x x x  .  o          "overfitting" durumu olur.Hard margin olur.Test datasında 
#       |    x x   x. o  o       ayrımı tam yapamaz  
#                    . 

#       |
#feature|      x   |   o  o         Soft Margin 
#    2  |    x xx  |  o o o      sağdaki x i hata olarak kabul etmiş ve line ı diğer   
#       |   x x x  |  o  o       sport noktalarına göre çizmiş.Train datasında biraz 
#       |    x x   |  x o  o     hata yapmış olsada test datasında daha iyi tahminler   
#                                yapacak

#Overfitting olmasındansa bazı hatalara izin verir.Bu işlem "C" parametresi ile 
#regularization uygulayarak yapılır 
#Yani margin,C parametresi ile ayarlanır 
#Regularization=>Hatalarda mücadele etmek için hata eklenerek overfit sorunundan 
#kurtulma işlemi 
#C parametresi=>Hataya ne kadar tolerans göstereceğim?(Overfit-underfitten kurtarma)

#KERNEL TRICK =>2D den 3D ye 
#    * *  |    oo                        o
#  * * *  |   o o o                     o*o
#   * * * |    o o                       o  
#_________|___________   
#   c c   |  p p        
#  c c c  | p   p 
#  c  c   |p p p 
#   c c   | p   p
#         |

#Yukarıdaki datalarımızda lineer bir line çizemeyiz.
#iç içe geçmiş bir durum var.Böyle durumlarda kernel trick ile 
#ile boyut değiştirme işlemi

#Kernel den yaptığı iş:
#1-Boyutu bir üst boyuta taşır (2D den 3D ye)
#Bu işlemi computional power gerektirmeden halleder
#oooxxxooo böyle bir datayı 1D den 2D ye çevirir ve araya hyperline o çizer

###SWM ÖZETLE=>>>Birbirine en yakın supportlar arasındaki margini belirleyip 
#ortaya hyperline çizer.
#Eğer data linear olarak ayrılamıyorsa kernel ile boyutu değiştirir,sonra hyperline çizer

#Kernel Parameters:
#1-Linear Kernel 
#2-Polynomial Kernel
#3-rbf
#4-sigmoid(Deep learning de kullanılır)
#5-precomputed(kendine güveniyorsan kendi kernelini kendin hesapla,buraya at)

#Linear=>Feature sayısı fazla,satır sayısı az ise önerilir 
#rbf=>Complex ve nonlinear datalarda tercih edilir.(Lineer datalarda bile kullanılabilir)
#Feature sayısı az satır sayısı fazla ise önerilir 

#C parameter(default=1)
#Marginin genişliğini ayarlar 
#C artınca Margin hard laşır(overfitting)

#Gama Parameter(default=1)
#Modelin complexity sini ayarlar.
#Gamma sadece nonlinear kernel larda kullanılır.(rfb,poly,sigmoid)
#Gamma arttıkça complexity artar 
#Yani linear kernal -> C parametresi
#nonlinear kernal   -> C parametresi,gamma 

###SVM Olumlu Yönleri 
#*Computer Vision sorunlarını çözer.(2D den 3D ye taşıma gibi) 
#Generalizaqtion error u düşük
#C ve gamma parametreleri ile oynayaarak complexity çözmek kolay 

##Olumsuz yönleri:
#Kernel parametre seçimi çok hassas
#Outlier veriler bu modellerde çok sorun çıkarabilir.Çünkü onlar yüzünden 
#hyperline ı yanlış çizebilir 

####Genel Bakış:
#SVC PARAMETERS :

#C --------> LinearRegression' daki alpha ile ters orantili calisir. Hataya ne
#kadar tolerans gosterecegimizi bu parametre ile ayarlariz. Default = 1. 
#(C kuculdukce uyguladigi regularization artar. Ayni LogisticRegression'da oldugu gibi).
#!!!!! Kucuk C overfitting' den kurtarir. !!!!!!
#C buyukse Hard Margin (overfitting durumu); C kucukse Soft Margin (underfitting durumu)
#KERNEL -----------> kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default = 'rbf'
#Eger data linear degilse, classlar icinde ic ice gecmis bir durum varsa ona 
#gore uygun kernel secimi yapilir.
#Modelin SVM olarak secilmesi, ic ice gecmis datalarda gundeme geliyor. Kernel
#burada bir trick uygulayarak datayi bir ust boyuta tasiyor (1D' den 2D' ye veya 2D' den 3D' ye). 
#Bu sekilde class' lar birbirinden kolay bir sekilde ayriliyor.
#'rbf' en iyi sonuclari verdigi icin en cok tercih edilen kernel. 'sigmoid' 
#daha cok deep learning modellerinde kullanilir.

#Hangi kernel' i sececegimize karar vermek icin modelimizi GridSearch'e 
#sokacagiz ve e bizim icin en iyisi kernel' i bulacak.

#DEGREE -------> Butun kernel' lar icin default = 1 olarak gorev yapar. Sadece 
#kernel = 'poly' secildigi anda degree' nin ust degerleri devreye girer ve 
#bu parametre ile oynayabiliriz. Diger butun durumlarda degree = 1' dir.

#GAMMA -------> {'scale', 'auto'} or float, default='scale'

#Cizilen margin' in alanini belirler. Gamma arttikca complexity artar. Sadece
#nonlinear datalarda kullanilir.

#!!!!!! Linear datalarda sadece C parametresine bakilir !!!!!!

#!!!!!! Nonlinear datalarda hem C hem de gamma degerlerine bakilir. !!!!!!

#!!!!!! C ile hata oranina, gamma ile ise margin sinirlarina karar verilir. 
#Ikisinin de olabilecek en kucuk degerlerini istiyoruz !!!!!!!   

#%%
######################NOTEBOOK 11#########################################
# Support Vector Machines(SVM)
# Çok kullanılan bir yöntem değildir
# 2 sınıf var bunları ayırmak istiyoruz
# 2 sınıf arasındaki birbirine en yakın ya da birden fazla nokta buluyor.
# Bu noktalar support vektor noktalarımız. O noktalardan geçen line lar support vektorlerimiz
# Bu 2 sınıfı ayıran da hyperplane yani karar yüzeyimiz
# Önceden text classification, pattern recognition , image-based gender detection da kullanılmış
# .. ancak şu an deep learning yöntemleri daha gelişmiş yöntemler buna nazaran

# Farklı hyperplanelerden en uygunununu buluyor ve bir karar yüzeyi oluşuyor

# 3 değişkenli data sette düzlem ile ayırabiliyoruz

# A daki gibi ayırmak mı, B deki gibi ayırmak mı doğrudur. Şu an da train aşamasındayız
# B çok daha sağlıklı. A mantıklı gibi görünüyor ancak mesela bütün noktaları
# .. ayırmaya çalışınca overfitting olabilir. O yüzden genelleme açısından B deki gibi ayırmak daha iyi

# Yeni gözlemler geldiğinde veya test te hatamızın büyümemesi için B yi seçiyoruz
# Yani overfitting in önüne geçiyoruz

# Hard margin : Train de hepsini ayırmaya çalışıyor
# Soft margin  : Train de daha genelleştirme yaparak ayırmaya çalışıyor. Hata daha büyük ama daha iyi

# SVM de 2 tane önemli hyperparameter var. C ve gama
# C: margini belirleyen parametre. Support vektörlerin açısını belirliyor
# C küçüldükçe margin büyüyor(yani soft margin yapıyor ve overfittingden kurtarıyor), 
# .. C büyüyünce margin küçülüyor(yani hard margin yapıyor ve hata küçülüyor ama overfitting e gitme ihtimalimiz artıyor)
# Marginleri bulduktan(ayarlandıktan) sonra hyperplane bulunuyor.
# John Hoca: Mülakatta sorulursa: Yani arka planda SVM de l2 regularization normu çalıştığı için overfittingle mücadele ediyor
# Marginler support vektörlere göre değişiyor.
# Orion Hoca: çizginin slope değişiyor dikkatli bakarsanız. 
# Orion Hoca: support vektörlerin yerini ayarlayarak hyperplane içine ne kadar misclasification yani hata gireceğini ayarlayarak modeli regülarize ediyor
# Grid search ile biz en uygun C yi bulacağız
# Class chat ten: # soft margin hataları kabul ediyor. # hard margin hatalara izin vermiyor.

# Kernel Trick
# Datalarımız her zaman kolay ayrılır data olmuyor. Bunlara non-linear data diyoruz
# Soldaki gibi bir data olabilir. Ayırmakta sıkıntı yaşayabiliriz  (1)
# Sağdaki şekilde de aynı şekilde hyperplane i belirlemek kolay değil (2)
# Aynı şekilde alttaki şekilde de sıkıntı  (3)
# Kernel bunlara bir çözüm sunuyor. Nasıl sunuyor? Bir sonraki slight a bakalım

# Datanın boyutunu arttırırsak bunları ayırmak daha kolay hale gelyior
# Yani parabolik hale getiriyoruz
# 3 boyuta geçince araya bir düzlem atarak bu noktaları ayırabiliyoruz
# Kernel aslında bir fonksiyondur. Kernel fonksiyonları tanımlıyorum. Ona göre datamı bir üst boyuta çıkartıyor
# .. ve datamız daha kolay ayrışıyor. Buna kernel trick deniyor
# Kernel trick mülakatlarda gelebilir

# Karesini alırsak burada biraz daha parabolik hale gelir ve 4. aşamada tek bir doğru ile bunları ayırabiliriz
# Kernel bazı problem datalarda işe yarayabilir

# 2. ders
# Kernel fonksiyonları
# Alttaki fonksiyonlardan birini seçiyoruz ve güzel bir şekilde ayırıyor sınıflarımızı
# Polynomial da 3 parametre, RBF te tek parametre , Sigmoid de 2 parametre var
# Bu parametreleri de tune etmem gerekir
# RBF en güçlüleridir

# Kernel da bir hyper parametredir
# Linear: Doğrusal bir şekilde ayırıyor
# RBF biraz daha esnek ayırıyor. Çok daha robust ve kullanışlı.(Gauss a benzer)
# Polynomial da parabolik bir şekilde ayırıyor

# Featurelar az, örnekler fazla iken RBF daha iyi sonuçlar veriyor diyebiliriz

# C hyperparametremiz. Margin ile ilgili bir parametre
# C arttıkça hard marginler oluşuyor. Varyans artıyor. Overfitting e doğru yol alıyor
# C nin yüksek olmasını tercih etmiyoruz. Mümkün olduğu kadar küçük ve pozitif olmasını isteriz

# Gamma hyperparametremiz. Kernel la ilgili bir parametre(Mülakatta gelebilir)
# Gamma, C ye göre daha belirgindir(Artıkça değişim daha çok gerçekleşir C ye nazaran).
# .. Gamma yı büyüttükçe hepsini cover etmeye çalışıyor ve bu da overfitting e götürüyor

# SVM
# Avantajları
    # Computer visionda iyi çalışıyor(Günümüzde öyle değil. Deep learning daha iyi çalışıyor)
    # Teste yeni data geldiğinde ondaki performans daha iyi
    # Kernel lar ile bir çok problemi aşar
# Dezavantajları
    # Parametre tuning ve hassas kernel seçimi gereklidir
    # Distance tabanlı yöntemlerde outlier(noise) sıkıntıdır. Burada da öyle

########## Support Vector Machines - Explaining Hyper Parameters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%matplotlib inline
#%matplotlib notebook
plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("mouse_viral_study.csv")
#Datamiz fareler uzerinde yapilan bir deneyle ilgili.
#Once farelere med_1 ilaci enjekte ediliyor. 
#1 hafta sonra med_2 ilaci enjekte ediliyor. 
#2 hafta kadar beklendikten sonra farelerde hala virus olup olmadigi kontrol ediliyor.
#Datamizdan herhangi bir prediction almayacagiz. 
#Amacimiz; hyperparameter' lar datamizda ne gibi degisiklikler yapiyor bunlari gozlemleyecegiz.
df.head()
#   Med_1_mL  Med_2_mL  Virus Present
#0     6.508     8.583              0
#1     4.126     3.073              1
#2     6.428     6.370              0
#3     3.673     4.905              1
#4     1.580     2.441              1
df.info()  # Missing values yok, Veri tipleri nümerik

########## Separating Hyperplane Manually
#Asagidaki grafikte class'larin birbirinden cok iyi ayrildigini goruyoruz :
sns.scatterplot(x='Med_1_mL', y='Med_2_mL', hue='Virus Present', data=df, palette='seismic')
#Asagidaki fonksiyonun amaci : 'Egimin -1, interceptin 11 oldugu linear bir
#dogru ciz.' x degerlerini 1 ile 10 arasinda verdik cunku datada med_1 ve med_2 
#sutunlari bu degerler arasinda.
x = np.linspace(0,10,100)
m = -1
b = 11
y = m*x + b
plt.plot(x,y,'black')
# Gayet kolay ayrılabilir 2 tane sınıf olduğunu görüyoruz

########## SVM - Support Vector Machine
# Amacımız bu karar yüzeylerine nasıl elde ediliyor
# NOT: SVM. classification ve regression için kullanılabilir

#SVM de KNN gibi cok maliyetli bir modeldir. Bu yuzden icine farkli degerler
#verip gozlemlemek cok mumkun degildir.
#SVM' nin iki cesidi var :
#1- Support Vector Classification (SVC) : Class' lari ayirmak icin kullanilir.
#2- Support Vector Machine for Regression (SVR) : Continuous verilerde kullanilir. 
#SVM' nin bu metodu pek kullanilmaz cunku iyi sonuclar vermez. Regresyonda cok 
#daha iyi sonuclar veren modeller tercih edilir.
#Biz bu datada class' lari ayirmak istedigimiz icin SVC metodunu kullanacagiz.
from sklearn.svm import SVC
#Asagidaki kod ile SVC icindeki hyperparameter' lari gorebiliriz :
help(SVC)
# NOTE: For this example, we will explore the algorithm, so we'll skip any scaling or even a train\test split for now
X = df.drop('Virus Present',axis=1)
y = df['Virus Present']

# plot_svm_boundary: support vektörlerini çizdirmek için kullanılıyor
# https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html
from svm_margin_plot import plot_svm_boundary # This is imported from the supplemental .py file

model = SVC(kernel='linear', C=0.9)
model.fit(X, y)
# Scale vs yapmıyoruz şu an sadece inceleme yapıyoruz. Kernel=lineer, c=0.9 demişiz(Örnek olarak)

plot_svm_boundary(model,X,y)
#!!!!!! Scale islemi bu modelde de cok onemli cunku unbalance
#durumlarda bazi feature' lara fazla agirlik verebilir. 
#Bu da tahminlerimizi yaniltir. !!!!!!

########## Hyper Parameters
### C
#Once asagida SVC modelimizi tanimlayip default degerler ile skorlarimiza bakacagiz :
model = SVC(kernel='linear', C=0.01)
model.fit(X, y)
#Asagidaki grafikte C degerini kuculttugumuzde margin araliginin genisledigini 
#ve hatalara daha toleransli olundugunu goruyoruz (Soft Margin) :
# C küçülünce margin büyümüş. Üstteki şekilde C=0.9 du(margin büyük olunca) margin küçüktü
plot_svm_boundary(model,X,y)

### Kernel

# Kernel ın etkisi nasıl oluyor bunu inceleyelim
#Yukaridaki grafikte SVM metodu ile linear datalara nasil bakilacagini gormustuk. 
#Simdi asagida nonlinear datalarda bu model nasil kullanilir onu 
#gorelim (Su anda kullandigimiz data linear bir data, biz isleyisi gormek 
#adina nonlinear duruma bakiyoruz).
#Once default C ve gamma degerleriyle grafigimizi cizelim :
model = SVC(kernel='rbf', C=1, gamma='scale')
model.fit(X, y)
plot_svm_boundary(model,X,y)
#Gamma degeri yukarida kesikli cizgilerin sinirlarini belirler. Gamma' nin 
#cizdigi sinirlarin disinda kalan alanlarin hepsi 'margin' olur. Cunku artik 
#2. boyuttan 3. boyuta gectik. Gamma sinirlari disinda kalan her noktayi 
#modelimiz yanlis tahmin edecek demektir.
#Gamma degeri buyudukce overfitting durumu ortaya cikar (C degerindeki durum ile ayni)
#Gamma cok buyuk olmadigi surece kontrol C' dedir. Ama gamma buyudukce kontrol 
#gammaya gecer. Bu durumda C parametresi buyuse de kuculse de overfitting durumu devam eder.
# Bu yuzden C ve gammanin birbirini dengeledigi durumu bulmak gerekir ki 
#bunu da GridSearch bizim yerimize yapar.
#!!!!!! C ve gamma kuculdukce uygulanan regularization artar !!!!!! 
#(Cok buyumelerini istemiyoruz, buyurlerse model overfitting' e gider.)
#plot_svm_boundary(model,X,y)
# RBF tam olarak parabol değil ama boyutu arttırıyor diyebiliriz.
# Gamma yı arttıralım şimdi altta
model = SVC(kernel='rbf', C=0.01, gamma=2)
model.fit(X, y)
plot_svm_boundary(model,X,y)

model = SVC(kernel='rbf', C=10, gamma=2)
model.fit(X, y)
plot_svm_boundary(model,X,y)

model = SVC(kernel='rbf', C=10, gamma=0.5)
model.fit(X, y)
plot_svm_boundary(model,X,y)
### Gamma

# gamma : {'scale', 'auto'} or float, default='scale' 
# Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
# - if ``gamma='scale'`` (default) is passed then it uses
#   1 / (n_features * X.var()) as value of gamma,
# - if 'auto', uses 1 / n_features.

model = SVC(kernel='rbf', C=1, gamma=0.4)
model.fit(X, y)
plot_svm_boundary(model,X,y)
# Gamma, scale ve auto olarak değer alabiliyor. Float da alabiliyor

model = SVC(kernel='sigmoid')
model.fit(X, y)
plot_svm_boundary(model,X,y)
# Sigmoid linear olmayan bir karar yüzeyi oluşturuyor

### Degree (poly kernels only)
# Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.
model = SVC(kernel='poly', C=1, degree=10)
model.fit(X, y)
plot_svm_boundary(model,X,y)
# Sonuç olarak kernelların etkisi farklı farklı oluyor

########## Grid Search
# Keep in mind, for this simple example, we saw the classes were easily separated, 
#which means each variation of 
# .. model could easily get 100% accuracy, meaning a grid search is "useless"

from sklearn.model_selection import GridSearchCV
svm = SVC()
svm.get_params()
param_grid = {'C':[0.0001,0.01,0.1],
              'kernel':['linear','rbf','sigmoid','poly'],
              'gamma':["scale", "auto"],
              'degree':[1,2]}         # Artırılabilir(polynomial varsa degree yi kullanıyor. Poly yi çağırmazsak degree yi kullanmıyor)
grid = GridSearchCV(svm,param_grid)
grid.fit(X,y)

#GridSearchCV(estimator=SVC(),
#             param_grid={'C': [0.0001, 0.01, 0.1], 'degree': [1, 2],
#                         'gamma': ['scale', 'auto'],
#                         'kernel': ['linear', 'rbf', 'sigmoid', 'poly']})

grid.best_score_   # Başarımız 1.0 Tamamen ayırt edebilmişiz
grid.best_params_  # Seçilen en iyi hyperparametrelerimiz

# {'C': 0.0001, 'degree': 1, 'gamma': 'scale', 'kernel': 'linear'}
# Class chat soru : c için verdiğimiz uç değer çıktığı için tekrar c için 
                 
# iste ayarlamaya gerek var mı hocam param_grid'de ?
# Orion hoca      : Deneyebilirsiniz, burada sadece etkileri görmek amacı ile örnek verildi

####################### Support Vector Machines - Classification

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%matplotlib inline
#%matplotlib notebook

plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df = pd.read_csv("diabetes.csv")
df.head()
df.shape
df.info() # Missing value yok. Dtype lar nümerik
df.describe().T # outlier var mı yok mu .. std>mean?
df.Outcome.value_counts()
sns.countplot(df.Outcome)
sns.boxplot(df.Pregnancies)
sns.boxplot(df.SkinThickness)
df=df[df.SkinThickness<70]
sns.boxplot(df.SkinThickness)
sns.boxplot(df.Insulin)
sns.boxplot(df.Glucose)
df=df[df.Glucose>0]
sns.boxplot(df.Glucose)
sns.boxplot(df.BloodPressure)
df=df[df.BloodPressure>35]
sns.boxplot(df.BloodPressure)
sns.boxplot(df.BMI)
df=df[df.BMI>0]
sns.boxplot(df.BMI)
df.shape
df.Outcome.value_counts()
index = 0
plt.figure(figsize=(20,20))
for feature in df.columns:
    if feature != "Outcome":
        index += 1
        plt.subplot(3,3,index)
        sns.boxplot(x='Outcome',y=feature,data=df)

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)
# Multicollinearity görünmüyor. Olsa da problem değildi SVM de l2 regularization bunu hallederdi

# df.corr()
# df.corr()["Outcome"].sort_values().plot.barh()
df.corr()["Outcome"].drop("Outcome").sort_values().plot.barh()

sns.pairplot(df, hue = "Outcome")

########## Train | Test Split
from sklearn.model_selection import train_test_split
X=df.drop(["Outcome"], axis=1)
y=df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
                                                                          ##########
# stratify=y: Dengeli oranda y yi böl(y oranı 2 ye 1 ise mesela bölerken de 2 ye 1 bölerek al)

########## Modelling and Model Performance
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix

def eval_metric(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)  
    print("Test_Set")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print("Train_Set")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))

######### Without Scalling
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
eval_metric(svm_model, X_train, y_train, X_test, y_test)
# Distance-based yöntem olduğu için scale yapmadan sonuçların kötü olduğunu görüyoruz

from sklearn.model_selection import cross_validate
model = SVC(random_state=42)
scores = cross_validate(model, X_train, y_train, scoring = ['accuracy', 'precision','recall','f1'], cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
# CV ile 0.73. 
# Üstte skor 0.80 yani datanın iyi yerine gelmiş.
#Scale edilmemis datada, Cross Validate isleminden sonra 
#recall ve f1 skorlarimiz baya dusuk cikti.

######### With Scalling
#Bundan sonra scale islemi uygulanacak butun datalarda 
#'pipeline' metodunu kullanacagiz ki CrossValidate ve 
#GridSearch' de olusacak data leakage sorununu onleyelim:
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
operations = [("scaler", StandardScaler()), ("SVC", SVC(random_state=42))] # Önce scale yapıp sonra modele sokacak pipeline
pipe_model = Pipeline(steps=operations)
# Pipeline: Bir kaç işlemi pipeline kurarak sırayla yapmasını sağlıyorduk
# ... train i fit_transform yapıp , test datasını transform yapıyordu

pipe_model.fit(X_train, y_train) # Eğitim
eval_metric(pipe_model, X_train, y_train, X_test, y_test)
# Yukardaki scale edilmemiş haline göre skorlar bu şekilde ama CV ye bakalım altta daha net sonuçlar için
#Cross Validate yaparken mutlaka pipe_modeli sifirlamak gerekiyor
operations = [("scaler", StandardScaler()), ("SVC", SVC(random_state=42))]
pipe_model = Pipeline(steps=operations)
scores = cross_validate(pipe_model, X_train, y_train, scoring = ['accuracy', 'precision','recall','f1'], cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
# test_accuracy: 0.75. Üstteki cv de(scale yapılmamış olan da) 0.73 dü
# Precision da performans düşmüş ama doğru olan scaling olan sonucumuz

########## With Best Parameters (GridsearchCV)
from sklearn.model_selection import GridSearchCV
#svm_model_grid.get_params

param_grid = {'SVC__C': [0.001, 0.05, 0.01, 0.1],
              'SVC__gamma': ["scale", "auto", 0.2, 0.3],
              'SVC__kernel': ['rbf', 'linear']}
# Hyperparametrelerimiz
# Polynomial olmadığı için degree yi de yazmadık. Yazadabilirdik
# John Hoca: Professional bir iş yapıyorsanız hepsini koymakta fayda var

operations = [("scaler", StandardScaler()), ("SVC", SVC(probability=True, class_weight="balanced", random_state=42))]# probability True to obtain ROC etc.
pipe_model = Pipeline(steps=operations)
svm_model_grid = GridSearchCV(pipe_model, param_grid, scoring="recall", cv=10)
# probability=True        : Olasılıkları muhafaza et. ROC çizdirirken lazım olarak
# Orion Hoca: SVC probability= true ile olasılıkları nasıl hesaplıyor, bilene benden çay
# .. Cevap: cv=5 default ile tahmin yapmasını sağlıyor ve hesaplamayı da yavaşlatıyor
# Class chat soru: Hocam probablity açıklmasında şu ifade yer alıyor. "Whether to enable probability estimates. 
#This must be enabled prior to calling fit, will slow down that method as it internally 
#uses 5-fold cross-validation, and predict_proba may be inconsistent with predict."
# .. Cross validation sonucunda predict ile predict_proba uyumsuz olabilir diyor? Bunu nasıl anlamalıyız
# Orion hoca: Buradaki predict probalara güvenmeyin(Öncesinde açıklaması var Orion hocanın.)
#(Saat13:29- videodan bakmak isteyenler için)
# Johnson Hoca: predict proba sonuçlarına güvenme predict sonucuna güven demek
# .. thread devamı : Probablity'yi True yapmamızın nedeni sadece ROC'mu hocam
# Orion hoca: evet ama güvenilir değil
# class_weight="balanced" : Targetdaki sınıf sayıları dengeli olmadığı 
#için(1 sınıfına ait olanların sayısı daha az), az sınıf modelde söz
# .. sahibi olmayabiliyor. Model fazla olan sınıfın daha önemli olduğunu 
#düşünüyor yani pozitif ayrımcılık yapıyor bir nevi
# .. sonra şunu yapıyor 1 sınıfına ait olanların(az olanların) ağırlığını mesela 
#2 ile çarpıyor, 0 sınıfına ait olanların(çok olanların) ağırlığını 1 ile çarpıyor
# .. Buna göre eşitleme yapıyor. Bunları dengeliyoruz ki daha doğru bir sınıflama yapsın.
# .. Sonuç olarak her bir sınıfa vereceğimiz ağırlığı belirlemiş oluyoruz
# Class chat: The “balanced” mode uses the values of y to automatically adjust
# weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
# class chat soru: Recall'ı artırmak gibi bir etkisi de olur diyebilir miyiz? --> John Hoca: Evet
# class chat : O zaman recall artacak ama FP lerde artacak
# John Hoca: inbalanced datanın çözümü ile ilgili 7-8 tane çözüm var. 
#Sentetik data üreterek data sayısını arttırıyor
# .. ya da çok olan sınıfı azaltarak vs gibi çözümler var ama bence en iyi 
#yöntem "balanced". Manuel olarak da yapabilirsiniz
# .. ya da dengeli olmak zorunda da değil. Hangi sınıf önemli ise problemimde 
#ona daha çok ağırlık ver diyebilirim. Ben yeterki o 40 müşteriye ulaşayım
# .. ya da 40 kanserliyi de bulmam lazım çünkü onu kaybetmek istemiyorum.
# scoring="recall" : skor u vermeseydik accuracy ye göre yapacaktı

# svm_model_grid.get_params
svm_model_grid.fit(X_train, y_train)
svm_model_grid.best_params_  # gamma yı 0.5, 0.6 lar vs de denenebilir.
pd.DataFrame(svm_model_grid.cv_results_)
svm_model_grid.best_index_
svm_model_grid.best_score_  # En iyi recall değerimiz
eval_metric(svm_model_grid, X_train, y_train, X_test, y_test)

operations = [("scaler", StandardScaler()), ("SVC", SVC(C= 0.05, gamma= 0.3, kernel= 'rbf', probability=True, 
                                                        class_weight="balanced", random_state=42))]
model = Pipeline(steps=operations)
scores = cross_validate(model, X_train, y_train, scoring = ['accuracy', 'precision','recall','f1'], cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
# CV yapıyoruz. Aslında çok gerek yok çünkü grid search de CV vardı yukarda
# Yukarda CV sonucunda bulduğum hyperparametreleri kullanarak burada CV yaptığımda recall değerinin aynı olduğunu görüyoruz

#Looking up parameters that can be passed to the pipeline
model.get_params().keys()
svm_model_grid.predict(X_test)

########### Overall performance of the model
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve, roc_auc_score, auc, roc_curve, average_precision_score, precision_recall_curve
plot_precision_recall_curve(svm_model_grid, X_test, y_test);
# Burada ROC da çizdirebilirdik.
# Precision-recall u daha çok imbalanced datalarda tercih ediyorduk

########### Finding Best Threshold for max f1 score
#Yukarida SVC' yi tanimlarken predict probalari alabilmek icin 
#'probability=True' demistik. Bunu yapmamizin sebebi bu probalarla ROC_AUC 
#veya precision_recall_curve' u cizebilmekti.
#'probability=True' aciklamasinda; predict ile predict_proba' lar arasinda 
#tutarsizliklar olabilir diyor. Uzakliklari olasiliklara donustururken birbirine 
#cok yakin olan uzakliklari donusturmede dogru olarak bir donusum yapabilmesi 
#icin kalibrasyon gerekir fakat kalibrasyon da maliyetlidir. <odelin kendisi de
#zaten maliyetli. Bu yuzden treshold' da ayar yapilsa bile skorlar pek degismeyecek. 
#Bu yuzden bu yontemde best treshold aranmaz.

#!!!!! SVC modelinde calisirken GridSearch sonucu cikan skorlari aliyoruz. Best Treshold' u aramiyoruz. !!!!!

operations = [("scaler", StandardScaler()), ("SVC", SVC(probability=True, random_state=42))]
svc_basic_model = Pipeline(steps=operations)
#probability=True -----> Modelimizin precision_recall_curve' u cizebilmesi icin
#bu parametreyi mutlaka 'True' yapmaliyiz. Asagidaki decision_function' a gore 
#SVM modelde olusturulan margine gore negatif olanlari 0 sinifina, pozitif 
#olanlari 1 sinifina atar. Sayisal degerler ise ortadaki best 
#hyperline' a olan uzakligi ifade eder. 'probability=True' parametresi bu 
#uzakliklari olasiliklara cevirir ve datanin hangi sinifa gidecegini belirler.

#ROC-AUC ve precision_recall_curve olasiliklara dayanarak cizim yaptiklari icin 
#de 'probability=True' mutlaka belirtilip degerler olasiliga cevrilmelidir.
svc_basic_model.fit(X_train, y_train)
plot_precision_recall_curve(svc_basic_model, X_train, y_train);

y_pred_proba = svc_basic_model.predict_proba(X_train)
average_precision_score(y_train, y_pred_proba[:,1])

#precision_recall_curve' de best_treshold f1 score' a gore belirlenir. 
#Recall' un max oldugu ama Precision' un da olabildigince en dengeli oldugu 
#noktayi secmemiz gerekiyor. Boylece en iyi f1 skoru, dolayisiyla da best_treshold' u bulmus oluruz.

precisions, recalls, thresholds = precision_recall_curve(y_train, y_pred_proba[:,1])

optimal_idx = np.argmax((2 * precisions * recalls) / (precisions + recalls))
optimal_threshold = thresholds[optimal_idx]
optimal_threshold
# F1 = (2 * precisions * recalls) / (precisions + recalls) = 2 * [ 1 / ((1/Recall) + (1/Precision)) ]
# optimal indexi veren threshold u bul --> 0.30
# np.argmax(recalls) yazarsak alttaki sonuçlarda recall 0.98 oluyor ama matriximiz;
"""
30  65
1   48
.. şekline dönüyor. diabetlerin hepsini hemen hemen yakalamış oluyoruz
"""

y_pred2 = pd.Series(svm_model_grid.predict_proba(X_test)[:,1]).apply(lambda x : 1 if x >= optimal_threshold else 0)
# 0.30 dan büyükleri 1, diğerlerini 0 sınıfına ata

print(confusion_matrix(y_test,y_pred2))
print(classification_report(y_test,y_pred2))
# Alttaki default sonuçlara göre f1 de yükseldi, recall da yükseldi

# Default değerlerimiz alttaki gibiydi
"""
test_accuracy    0.752
test_precision   0.673
test_recall      0.536
test_f1          0.594
"""

# Bunu gridsearch sonucunda elde ettiğimiz(svm_model_grid) e uygulayalım bu eşik değeri
y_train_pred2 = pd.Series(svm_model_grid.predict_proba(X_train)[:,1]).apply(lambda x : 1 if x >= optimal_threshold else 0)
print(confusion_matrix(y_train, y_train_pred2))
print(classification_report(y_train, y_train_pred2))
# F1 imiz arttırmış olduk. Recall düşmüş. Bir yer artarken diğer yerden kaybetmişiz

# Önceki sonuçlar(svm_model_grid)
"""
Test_Set
[[45 50]
 [ 3 46]]
              precision    recall  f1-score   support

           0       0.94      0.47      0.63        95
           1       0.48      0.94      0.63        49

    accuracy                           0.63       144
   macro avg       0.71      0.71      0.63       144
weighted avg       0.78      0.63      0.63       144
"""

#%%%
###############LESSON 12######################################

########Decision Tree Theory(DTT)
#Hem regression,hem classification da kullanılır 

#Classification problemlerinden:Medical Diagnosis,Text Classification,Credit 
#risk Analysis lerinde kullanılır 

#Decision Tree Diagram(Ters Ağaç)
                      
#                        Desicion Node
#       Desicion Node                   Desicion Node
#Leaf Node   Leaf Node            Leaf Node      Decision Node 
#                                              leaf node   leaf node 


#En üstteki=>Root Node 
#En alttaki=>Leaf Node 
#Aradakilerin hepsi=>Decision node

#class ları ayırabildiği en iyi yerden bölmeye çalışır
#Amacımız;highest homogeneity.Yani böldüğümüz bölgelerde tek bir sınıfın kalması 

#Decision root u hangi attribute ile seçeceğim?
#Seçtiğim featureı neresinde ayıracağım?
#Ağacın büyümesine ne kadar izin vereceğim?(Gini index-Information Gain/Entropy)

##########1-Gini Index:

    #O bölgedeki yanlış hesaplamanın değerini minimize etmeye çalışır 
#Gini=1-ΣP^2(Xi)
#1-(i classında olma olasılığı)

#Amacımız Gini yi küçültmek 
#Eger ayrılan sınıf "pure" ise 1 olma ihtimali 1-1=0 olur,yani başarı %100 olur

#Gini index Nasıl Çalışır?
#Diyelim ki 3 feature ve 1 targetımız var.3 feature için Gini formulü uygulanır 
#ve gini değerleri bulunur 
#column_1=0.364,column_2=0.360,column_3=0.381 =>Gini index,"root node" olarak
#column_2 yi seçer.Çünkü en küçük gini i arıyoruz.En iyi ayrımı gini index yapmış 

#########2-Entropy Formula 
#Entopy=Kaos demek
#Entropinin en az olmasını istiyoruz 

#H=-ΣP(xi)log2p(xi)

#Seçtiği bölgede(+) ve (-)ler birbirine eşit olursa "max entropy" olur 
#(+) olma olasılığı=>½50
#entropy=>1
#Pure değil

#Seçtiği bölgeler "pure" olursa entropy 0 olur 
#*(+) olma olasılığı =>%100
#*enropy=>0
#*Pure

#Information Gain (IG)
#En fazla bilgiyi hangi node üzerinden hesaplıyoruz(Root node nedir)
#Bize entropy i IG verecek 

#!!!Gini Impurity->düşürmeye çalışıyoruz
#!!!Information Gain->Yükseltmeye çalışıyoruz 
#Biri azalırsa diğeri yükselir 

#Information Gain Nasıl Çalışır?
#Mango,apple,banana classlarına ait ağırlık ve uzunluk feature ları olsun 


#Height(cm)|Width(cm)|Class(target) 
#                    |Mango
#                    |Banana
#                    |Apple

#Mesela Height>10 üzerinden entropy i hesaplar ve,IG(height)=0.696
#Widht>5 üzerinden entropy i hesaplar ve;IG(widht)=0.97
#Bu ikisinden büyük olanı seçer(Gininin aksine)

#DESICION TREE ILE REGRESSION-VARIENCE 

#CART=>Classification and Regression Tree ilk bu amaçla kullanılmış.
#Gini Impurity ve Entropy burda yok.Sadece classification da var.

#Bu yöntem,variance üzerinden çalışır. 
#Variance a göre sınıf ayrımı 
#Mean e göre prediction 

#İlk nerden bölersem iki taraftaki variance lar min olur? diyerek ayrım yapıyor
#Yani bölgeler arası variance ı en aza indirmeye çalışıyoruz.(Yani sayılar 
#arasındaki fark az ise aynı sınıfta)

#Tahminleri ise mean üzerinden yapar.

#Hangi bölgedeyse o bölgedeki noktaların mean i prediction olarak geri döner 
#!!!!Variance küçük olsun istiyoruz ama hataya hiç izin vermezsek datayı ezberler
#overfitting olur 

#%%
###################NOTEBOOK 12####################
# Ares Hoca: Advanced ML modellerine doğru ilerliyoruz
# Ağaç modellerinin temelinde olan bir model ile başlayacağız Decision Tree
# Decision Tree ile Regression ve Classification yapabiliyoruz
# Dataya sorular sorarak sınıflandırma ve regression yapan model
# Non-parametrik model. Yani assumption ı yok
# Outlierlardan etkilenmiyor
# Multicollinearity den etkilenmiyoruz

# Decision Tree Theory
# Kullanım alanları aşağıdaki gibi
# Overfitting e açık bir algoritma o yüzden hyperparametrelerle oynanması gerekli olan bir algoritma

# Ağacı ters çevrilmiş düşünelim
# Kök node u sonra sub-tree,sub-node-internal node vs de deniliyor
# IF ELSE soruları sorarak ayrıştırılmış oluyor
# "Leaf node" lada class larımız oluyor yani sınıflandırmayı yapmış bitmiş
# "Decision node" da daha sınıflandırmayı bitirmemiş devam etmesi gerekiyor vs
# Sağdaki şekle göre dışarı çıkacağız mesela havanın durumuna göre dataya sorular sorarak ilerleyen model
# Hangi feature dan başlayacak. Hangi thresholddan sonrasını alacak vs bunlara detaylı bir şekilde bakacağız birazdan
# Hangi feature dam bölmeye karar vermesi için gini index(Yanlış sınıflandırma olasılığı üzerinden hesaplanıyor) ve 
# .. information gain(Entropi üzerinden hesaplanıyor) e bakıyor.

# DT algoritması ilk başta income veya debt açısından çizgiler çiziyor. Ona göre karar veriyor
# 1 de hem x ekseninde hem y ekseninde çizgiler çizerek ayıracağına karar veriyor
# t1 den ayırırsam income için en iyi yapmış olurum diyor sonra sağ tarafta kalan kırmızıları bir nevi(şekil 2) ayırmış oluyor
# t2 den çizgi çiziyor sonra sol üstte mavileri arttırmış oluyor(şekil 3). Hala sol altta karışık sınıflar var
# 4 adımda(şekilde) t3 den ayırıyor yani 3. şekildeki sol altta kalan karışık kısmıda t3 ile 4. şekilde ayırmış oluyor
# En sonra Final da bütün datayı sınıflandırmış olacak

# Nümerik featureları belli threshollara göre bölüyor
# Root node da en iyi ayrımı yapacak feature u seçiyor. Burada X1 miş o feature
# x1>1 yapınca 6 tane yeşil 0 tane kırmızı geliyor(Şekilde solda kalan kısım),
# X1>1 -- > x2>1 --> x1>1.8 için 4,0 ve 0,6 şeklinde ayrım yapmış. (Şeklin sağ üstte ve orta üst kısmı)
# Pure leaves --> 0,6 mesela yani 0 yeşil sınıf 6 tane kırmızı sınıf 
# Bu şekilde ayırımlar yaparak çalışıyor decision tree

# Bölünme yaparken nasıl karar verecek decision tree
# Leaf node larda ya da sub-decision node larda Pure luğu sağlayacak şekilde yapmaya çalışıyor
# %100 pure luk da istemiyoruz o zamanda overfit e gidiyor
# 1.Root node umuzda hangi feature dan başlayacağımıza nasıl karar vereceğiz?
# 2.Feature u hangi thresholldan ayıracağız?(Age i 18 den mi 20 den mi böleceğiz gibi...)
# 3.Her node da farklı feature seçilecek buna nasıl karar vereceğiz?
# 4.Dallanmayı ne zaman durduracağız?
# Gini index ve informartion gain hesaplamalarına göre üstteki 4 sayıyı cevaplamış olacağız

# Pure luk durumları resmedilmiş
# Siz bırakırsanız model minimum impurity yapar ve overfit e gider
# Biz hyperparametreler ile oynayarak less impure yapmaya çalışacağız

# hangi feature dan başlayacağımıza alttakilere bakarak karar vereceğiz
# Gini heterojenliğe bakar ve bunun o yüzden minimum olmasını isteriz
# Information gain in maximum olmasını isteriz
# Not: Information gain entropi üzerinden hesaplanıyor

# n= cluster ımızın sayısı
# Formülün sağındaki Bir sınıfa ait olma olasılığı
# Biz 1 den çıkartınca yanlış sınıflandırma oranını hesaplamış oluyoruz o yüzden gini minimize ediliyor
# Yani heterojenliği minimize ederek sınıflandırmayı güzel bir şekilde yapıyoruz 

# Gini indexinin hesabı
# Arka planda algoritma bunu yapacak ama nasıl hesapladığına şöyle göz atalım
# 3 feature açısından gini indexi hesaplanmış. Bunlardan hangisi en küçükse root node u belirlemiş oluyor
# .. sağ alttaki şekle göre good blood en küçük(0.360)
# Bu feature dan model bölünmeye başlayacak

# Boyut artarsa sınıflandırmayı nasıl yapacağını sol alttaki şekle bakarak görebiliyoruz

# Bir diğer karar ölüçütümüz information gain di
# Bunun maximum olmasını istiyoruz
# Hangi feature umuz max ise onu root node olarak belirleyecek model 
# Bir sınıfa ait olma olasılığı ve bunun logaritması olarak adlandırılıyor

# Olasılık 1 olursa entropi 0 oluyor. (Bir küme içinde sadece tek bir sınıfın olması)(Sağ üstte hocanın çizdiği)
# Olasılık 0.5 olursa entropi max olmuş oluyor yani 1 oluyor(Bir küme içinde 1 tane 1 sınıfı, 1 tane 0 sınıfı olması)((Sağ altta hocanın çizdiği))

# Information gain i feature sayıma göre tüm feature lar için hesaplayıp. Hangisinin information gain i
# .. büyükse o feature a göre dallanma yapıyor.
# Bunu her adımda yapıyor

# Information gain hesabı var yine buradada
# Sol üstte: 3 tane meyveyi yüksekliği ve genişliğine göre nasıl sınıflandırırız
# Önce yüksekliğe göre mi genişliğe göre mi hesaplayayım sorusuna cevap bulmak için information gain e bakıyor
# Sağ alttaki şekilde information gain : 0.971, 0.696 şeklinde çıkmış sonuçlar. Root node olarak Width i seçiyor
# Class chat: Biz bunları belirledikten sonra mı modeli oluşturacağız?
# Ares Hoca: Modelin default parametresi gini indexi kullanarak yapıyor bu hesaplamaları.
# .. Biz bunu grid search te fine tunin yaparak başka parametreleri de deneyeceğiz
# Orion hoca:model kendi karar verecek. siz hangi yaklaşıma göre bölmesini  ve nerede durmasını belirleyeceksiniz
# Sonuç olarak gini indexin minimum information gain in max olmasını istiyoruz

# Buraya kadar olan classification kısmıydı
# Regression özelinde data noktaları nümerik olacağı için varyansı minimum yapmaya çalışıyor model
# Yeni gözlem geldiğinde if else sorularıyla bir karar veriyor. Hangi bölgede varyans küçükse yen, gözlemi o bölgeye atıyor
# Varyansın minimumluğuna bakıyor ve o bölgedeki tüm datamın ortalamasını(leaf node daki değer) alarak gözlemi atıyor bölgeye
# Class chat Soru: yeni gözleme de sorular sormuyor mu hocam?
# Orion Hoca: sorular belli trainde öğrenildi.sadece değerlere göre yerine gidiyor

# 2 ders başı
# Criterion: hangi featuredan başlayacağına karar vermesi için .. defaultgini
# max_depth: Ağaç ne kadar aşağıya dallanacak. Default None(Ağaç aşağı dallanmaya devam eder None da ve overfit e gider)
# NOT: En önemli parametre max_depth. Tree modellerinde ilk müdahale max_depth e yapılır

# splitter: Nereden böleceğine karar veriyor
# Max_features: Datadaki tüm featurelarımı kullanmasını mı istiyoruz(None). Yoksa 3 tanesini kullanarak mı bölme yapacağız

# min sample_split: Dallanmayı yapması için min gözlem sayısı
# min_samples_leaf: Yaprakların yaprak olması için minimum kaç gözlem olması gerekiyor

############# DECISION TREE CLASSIFICATION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

df = pd.read_csv("penguins_size.csv")
df.head()
# Penguenlerin vücut ölçüleri, bulundukları adalar, cinsiyetleri vs gibi bilgileri olan bir datamız var
# Penguenlerin türlerini belirlemeye çalışacağız. species bizim label ımız

df.info() # Missing value larımız var. Bunlarla uğraşmayacağız burada drop edip modellemeye geçeceğiz
df.describe() # std> mean durumu yok. Yani outlier yok
df.isnull().sum()
10/344  # 10 tane değer datanın yüzde 2 sini oluşturuyor. Çok büyük bir oran değil o yüzden drop edeceğiz
df.dropna(inplace=True)
df.info() # Null değerlerimiz gitmiş oldu
df.head()
df["sex"].unique()  # Cinsiyette yanlış girilmiş değer var. Uğraşmadan drop edeceğiz
df[df["sex"]== "."]
df.drop(index=336, inplace=True)
df2 = df.copy()   # df2 lazım olabilir diye kaydediyoruz
df["species"].value_counts()
ax = sns.countplot(x="species", data = df)
ax.bar_label(ax.containers[0]);
# Balanced bir durum yok. Unbalanced bir durum var mı yok mu karar vermek için skorlarımıza bakacağız
# Mesela 68 değeri güzel bir 68 değerlerdir belki. Skorlarımız güzelse balanced diyeceğiz

ax = sns.countplot(x="species", hue="sex", data = df)
for p in ax.containers:
    ax.bar_label(p)
# Cinsiyete göre species dağılımları

plt.figure(figsize=(12,6))
sns.pairplot(df,hue='species',palette='Dark2')
# Sınıflarda ciddi ayrışmalar var. Datamız güzel ayrışacak gibi duruyor. 
# İç içe geçen grafikleri 3. boyutlu inceleyeceğiz. Belki onlarda güzel ayrışıyordur 3 boyutta

df.species.unique()
# !pip install plotly
import plotly.express as px
df.select_dtypes("number")

plt.figure(figsize=(8,6))
sns.heatmap(df.select_dtypes("number").corr(),annot=True, cmap='viridis')
plt.title("Correlation Matrix")
plt.show()
# 0.87 var. Onun haricinde yüksek korelasyon görünmüyor
# Bu 0.87 için body_mass ile flipper length arasında multicollinear durum olsa bile modelim non-parametrik 
# .. olduğu için önemli değil
# Kısa bir Edadan sonra modellemeye geçiyoruz

######### Train | Test Split
X = df2.drop(columns="species")
y = df2['species']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

######## OrdinalEncoder and Categoric Variable
# Meselenin püf noktasına geldik. Kategorik feature larımız olduğu zaman one-hot encoder yapıyorduk
# one-hot-encoder her unique değere eşit muamalede bulunuyordu
# Burada Ordinal encoder kullanacağız ve nominal bir şekilde sıralamış olacağız yani mesela;
# Low: 0 , medium: 1, large :2 diyeceğiz mesela. Yani bir hierarşi olacak
# Bunu neden yapıyoruz
    # 1.Ordinal sıralama yaptığımız zaman tree base modeller daha hızlı çalışır
    # 2.Skorlarımız yüzde 1-2 arasında daha iyi olabiliyor
    # 3.Feature importance ı sağlıklı bir şekilde yapabilmemiz için kullanıyoruz
# Bu tree base modellerde geçerli. Kategorik değişkenimiz varsa ordinal encoder yapıyoruz

# Class chat soru: ordinal encoder'da feature sayısını artırmıyoruz. 
#Bu nedenle de daha hızlı çalışıyor diyebilir miyiz? --> Ares hoca: Evet
# ordinalcoder yapildiktan sonra scaling yapiliyormu? --> Ares hoca: Evet

cat = X_train.select_dtypes("object").columns
cat

X_train[cat]

from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
enc.fit_transform(X_train[cat])   # Categorik featureları fit_transform yapınca 0,1,2,3.. lere dönüşüyor

X_train[cat] = enc.fit_transform(X_train[cat])
X_train[cat]

X_train.head()  # island ve sex artık nümerik bir yapıda

X_test[cat] = enc.transform(X_test[cat])  # data leakage olmaması için X_test e sadece transform yaptık
X_test[cat]

X_test.head() # X_Test te de kategorik feature lar nümerik e döndü

"""
If we didn't use pipeline, we would set up the model as follows.

from sklearn.tree import DecisionTreeClassifier

DT_model = DecisionTreeClassifier(random_state=101)
DT_model.fit(X_train,y_train)
"""
############ Modeling with Pipeline
# Üstteki kısmı ord_enc ı anlatmak için yaptık. Aşağıda pipeline da yapacağız yine ordinalencoder ı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
X_train.head(2)
X_test.head(2)
cat

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
column_trans = make_column_transformer((ord_enc, cat), remainder='passthrough')
# handle_unknown='use_encoded_value' : Bir categorik column da bilinmeyen bir değer geldiği zaman encodlanmış değeri ver Yani --> -1 i ver
# unknown_value=-1 : Verilecek değer
# ord_enc, cat i yukarda tanımlamıştık. Bunlara dönüşüm yapacak
# remainder='passthrough' : Kategorik değişkenlere ord_enc yap. Diğerlerini pass geç

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
operations = [("OrdinalEncoder", column_trans), ("DT_model", DecisionTreeClassifier(random_state=101))]
# ("OrdinalEncoder", column_trans): Kategorik değişkenlere ordinal encoder yapacak ve bilinmeyen bir değer geldiği zaman -1 verecek
# ("DT_model", DecisionTreeClassifier(random_state=101) : Modelimizi uygulayacak
pipe_model = Pipeline(steps=operations)
pipe_model.fit(X_train, y_train)      # Bu satırda eğitimi tamamlamış olduk

############ Model Performance on Classification Tasks
from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix

def eval_metric(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    
    print("Test_Set")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print("Train_Set")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))

plot_confusion_matrix(pipe_model,X_test,y_test)
# Yorum : 
# Adelle   : Modelim 30 tanesini doğru bilmiş 2 hata yapmış
# Chinsrap : Modelim 15 tanesini doğru bilmiş 1 hata yapmış
# Gento    : Modelim 19 tanesini doğru bilmiş 0 hata yapmış

from yellowbrick.classifier import ClassPredictionError
visualizer = ClassPredictionError(pipe_model)
# Fit the training data to the visualizer
visualizer.fit(X_train, y_train)
# Evaluate the model on the test data
visualizer.score(X_test, y_test)
# Draw visualization
visualizer.poof();
# Yorum:
# Adelle: Soldaki sütunda maviler adelle mavi ve yeşiller hatalar
# Chinstrap: Ortadaki sütunda yeşiller chinstrap 1 tane adelle
# Ghento  : Tamamen kırmızı. Hiç hata yapmamış
# Böyle görselleştirme ile de bie insight çıkarmış olduk

eval_metric(pipe_model, X_train, y_train, X_test, y_test)
# Skorlar yüksek
# Normalde class lar unbalanced dı ama skorlar iyi gelmiş. Bundan dolayı datamızı "balanced" olarak değerlendireceğiz
# Bu yüzden(data balanced olduğu için) accuracy üzerinden gideceğiz.
# Müdahale etmediğimiz için(hyperparametreler ile oynamadığımız için) gördüğümüz gibi train de 1.00 gelmiş
# .. Test te 0.96 gelmiş. Burada skorlar yakın olduğu için sorun yok diyebiliriz. Overfitting durumu görünmüyor
# Eğer içimizin rahat olmasını istiyorsak bir hata payı ekleyerek train de hatayı düşürebiliriz. Bazı data scientistlerin yaklaşımı böyle
# Unbalanced durumu olsaydı macro avg ve weighted avg skorlarına bakacaktık
# macro avg : örneğin precision hesabı    : üstteki değerlerin ortalamasıdır. 0.94+0.94+1.00 / 3 = 0.96
# weighted avg : örneğin precision hesabı : Support ve precision değerleri kullanılarak hesaplanır 31*0.94+16*0.94+20*1.00 / (31+16+20) = 0.96
# NOT: micro avg skorlar accuracy ile aynıdır(Alttak CV kısmında göreceğiz)
# Class chat soru: data eğitim datası olduğu için böyle çıkması normal herhalde. autoscout datasıyla test etsek bu rakamları görmezdik büyük ihtimalle..
# Orion hoca: görme ihtimalinizde var

############# Cross Validate
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import make_scorer

# Modelimizi 0 dan kurup Cv yapalım. Gerçek skorumuza bakalım
operations = [("OrdinalEncoder", column_trans), ("DT_model", DecisionTreeClassifier(random_state=101))]
model = Pipeline(steps=operations)
scores = cross_validate(model, X_train, y_train, scoring = ["accuracy", "precision_micro", "recall_micro", "f1_micro"], cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores
df_scores.mean()[2:]  # Asıl skorumuz 0.97 imiş. Overfitting durumu yok. Yukarda aldığımız skorlar ile yakın

########## Evaluating ROC Curves and AUC
from yellowbrick.classifier import ROCAUC
model = pipe_model
visualizer = ROCAUC(model)
visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show();                       # Finalize and render the figure
# ROC değerleri gayet yüksek
# Ağırlıklı skorlarımızında yüksek olduğunu görüyoruz
# Class chat soru: roc sadece biniary modeller icin cizdirmiyormuyduk. 
# yoksa o sadece logistic regression icinmi gecerli?
# Orion hoca: görmek isterseniz one versus rest üzerinden multidede çizdirebilirsiniz.
#metricler modelden bağımsızdır.roc bütün classification algorithmalarında roc
# .. RMSE bütün regression algorithmalarında RMSE gibi
# Thread devamı : Anladim hocam. John hocam sadece binanry icin cizdiriyoruz demisti. 
# Hatta bir notebookta cizdirmeye calismisit hata vermisti. Tsk ederim cvp icin.
# .. Orion Hoca: sklearnde bu yetenek yok oyüzden söyledi.zaten ROC binary classification için çıkmış bir metric

############ Feature İmportances with Pipeline
pipe_model["DT_model"].feature_importances_ # DT_model.feature_importances_
# Buradaki feature importance sıralaması make_column_transformdan gelen sıralamaya göre yani
# .. verideki asıl olan sıralamamıza göre değil(df.head() e göre DEĞİL). Yani;(Bir alt koda bakalım)
# Orion hoca: bu model de coef var mı? --> Ares hoca: Non-parametrik model. Buradaki değerlerimiz feature değerlendirmesi
# .. bunların toplamı da 1 yapar

X_train.head(1) # Üstteki feature importance değerleri buradaki sütun sırasına karşılık gelen importance lar değil
# Normalde;
# island           : 0.03327601
# sex              : 0.02363079  .. Yani mesela culmen_length_mm  --> 0.02363079  DEĞİL
# culmen_length_mm : 0.35128085
# culmen_depth_mm  : 0.04724943
# flipper_length_mm: 0.54456291
# body_mass_g      : 0.0

pd.DataFrame(pipe_model["OrdinalEncoder"].fit_transform(X_train))
# İlk 2 feature ım kategorik featurelar make_column_transform dan dolayı başa alınmış oldu
# Sütunlarda 0 yazan yer "island" . 1 yazan yer "sex"

X_train.columns  # Normal sıra böyle bu sırayı ayarlamalıyız feature importance ile eşleşmesi için

list(X_train.select_dtypes("object").columns)  # Önce catleri alıyoruz

list(X_train.select_dtypes("number").columns)  # Sonra nümerikleri alıyoruz

features = list(X_train.select_dtypes("object").columns) + list(X_train.select_dtypes("number").columns)
features
# Bunları burada birleştiriyoruz ve sıra düzelmiş oluyor

df_f_i = pd.DataFrame(data = pipe_model["DT_model"].feature_importances_, index=features, #index=X.columns
                      columns = ["Feature Importance"]).sort_values("Feature Importance", ascending=False)
df_f_i
#  index=features: island ve sex başta olacak şekilde indexlerimizi ayarlamış olduk

ax = sns.barplot(x = df_f_i.index, y = 'Feature Importance', data = df_f_i)
ax.bar_label(ax.containers[0],fmt="%.3f");
plt.xticks(rotation = 90)
plt.tight_layout()
# Overfitting model complexity nin artmasıydı. Train ve test skorları arasında uçurum olmasıydı
# Bazen en önemli feature overfitting e neden olabiliyor.
# Flipper length her adımda en önemli feature olarak belirlenip diğer feature lara soru sormayabilir model bu da overfit e neden olur
# Flipper length i atıp denenebilir tree modellerde.
# Her zaman olan bir şey değil ama denememiz lazım

############## Drop most important feature
# Burada bu denemeyi yapalım
"""
The feature that weighs too much on the estimate can sometimes cause overfitting. For this reason, 
the most important feature can be dropped and the scores can be checked again
"""
X.head(2)

X2 = X.drop(columns = ["flipper_length_mm"])  # flipper_length_mm dışındaki tüm feature larımızı aldık
# Altta aynı şeyleri yapıyoruz tekrar

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y, test_size=0.2, random_state=101)

operations = [("OrdinalEncoder", column_trans), ("DT_model", DecisionTreeClassifier(random_state=101))]
pipe_model2 = Pipeline(steps=operations)
pipe_model2.fit(X_train2, y_train2)

eval_metric(pipe_model2, X_train2, y_train2, X_test2, y_test2)
# Skorlarım test te 0.99, train de 1.00
# Yani skorlar daha da yakınlaşmış oldu
# Yukardaki eski skorlarda test te 0.96 , train de 1.00 di
# Yani flipper_length_mm overfitting e neden oluyormuş.
# Skorum gerçekten 0.99 mu? CV yapıp bakalım

operations = [("OrdinalEncoder", column_trans), ("DT_model", DecisionTreeClassifier(random_state=101))]
model = Pipeline(steps=operations)
scores = cross_validate(model, X_train2, y_train2, scoring = ["accuracy", "precision_micro", "recall_micro", "f1_micro"], cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores

df_scores.mean()[2:] # Gerçek skorum 0.98 miş (Flipper length yokken)
# Tekrardan feature importance çizdirebilirdik ama notebook uzamasın diye yapmadık

"""
# Flipper length varken skorlarmız bunlardı
scores with most importance feature  (

test_accuracy           0.973789
test_precision_micro    0.973789
test_recall_micro       0.973789
test_f1_micro           0.973789
"""
########### Adelie-Chinstrap weighted metric Scores
"""
# Bu kodu tek sınıfa ait skorlar almak istersek alttaki kodu kullanabiliriz

# We can look at the scores of the whole target column individually like this

scoring = {'precision-Adelie': make_scorer(precision_score,  average=None, labels=["Adelie"]),
           'recall-Adelie': make_scorer(recall_score, average=None, labels =["Adelie"]),
           'f1-Adelie': make_scorer(f1_score, average=None, labels = ["Adelie"]),
          
          'precision-Chinstrap': make_scorer(precision_score,  average=None, labels=["Chinstrap"]),
          'recall-Chinstrap': make_scorer(recall_score, average=None, labels=["Chinstrap"]),
          'f1-Chinstrap': make_scorer(f1_score, average=None, labels=["Chinstrap"]),
          
          
          'precision-Gentoo': make_scorer(precision_score,  average=None, labels=["Gentoo"]),
          'recall-Gentoo': make_scorer(recall_score, average=None, labels = ["Gentoo"]),
          'f1-Gentoo': make_scorer(f1_score, average=None, labels = ["Gentoo"]),
          
          }

"""

# 1 den fazla skor almak istediğimizde average="weighted" yazmalıyız
scoring = {'precision-Adelie-Chinstrap': make_scorer(precision_score,  average="weighted", labels=["Adelie", "Chinstrap"]),
           'recall-Adelie-Chinstrap': make_scorer(recall_score, average="weighted", labels =["Adelie", "Chinstrap"]),
           'f1-Adelie-Chinstrap': make_scorer(f1_score, average="weighted", labels = ["Adelie", "Chinstrap"])
          }

operations = [("OrdinalEncoder", column_trans), ("DT_model", DecisionTreeClassifier(random_state=101))]
model = Pipeline(steps=operations)
scores = cross_validate(model, X_train2, y_train2, scoring = scoring, cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
# Sonuç olarak 2 sınıf sizin için önemli ise bu şekilde sonuçlar alabiliyoruz

########## Visualize the Tree
from sklearn.tree import plot_tree
plt.figure(figsize=(12,8), dpi=75)
plot_tree(pipe_model["DT_model"], fontsize=10);
# Bu belirsiz görseli aşağıda daha güzel hale getirelim

X.columns

features

plt.figure(figsize=(12,10),dpi=100)
plot_tree(pipe_model["DT_model"], filled=True, feature_names=features, class_names= y.unique(), fontsize=7);
# Dallandırmaları gini index üzerinden yapmış
# Burada ilk node da flipper_length i görüyoruz. gini=0.636 .. Demek ki flipper length in gini si en düşükmüş
# En sol altta values 7,0,0 ve 0,1,0 gelmiş. Yani pure luk sağlanmış.
# Altta kalan diğer node lara bakrığımızda pure sonuçlar görebiliriz.
# Sonuç olarak;  bu sağlanana kadar devam ediyor model

############ Understanding Hyperparameters
###### Max depth, min samples split, min samples leaf
def report_model(model):
    model_pred = model.predict(X_test)
    model_train_pred = model.predict(X_train)
    print('\n')
    print("Test Set")
    print(confusion_matrix(y_test, model_pred))
    print('\n')
    print(classification_report(y_test,model_pred))
    print('\n')
    print("Train Set")
    print(confusion_matrix(y_train, model_train_pred))
    print('\n')
    print(classification_report(y_train,model_train_pred))
    plt.figure(figsize=(12,8),dpi=100)
    plot_tree(model["DT_model"], filled=True, feature_names=features, class_names = y.unique(), fontsize=10);
    #feature_names=X.columns
    
# Test ve Train in confusion matrix ve classification reportlarını topladık bu fonksiyonda

DT_model = DecisionTreeClassifier(max_depth=2, random_state=101)
operations = [("OrdinalEncoder", column_trans), ("DT_model", DT_model)]
pruned_tree = Pipeline(steps=operations) #pruned_tree = DecisionTreeClassifier(max_depth=2, random_state=101)
pruned_tree.fit(X_train,y_train)
# max_depth= None normalde .. Biz max_depth=2 yapıp deniyoruz şu an

report_model(pruned_tree)
# Yukardaki modellerde kıyaslandığı zaman hatalarım düştü. Bir nevi hata ekleyerek skorları birbirine yakınlaştırdık diyebiliriz

########## Max Leaf Nodes
# Diğer parametreleri deneyelim
DT_model = DecisionTreeClassifier(max_leaf_nodes=7, random_state=101)
operations = [("OneHotEncoder", column_trans), ("DT_model", DT_model)]
pruned_tree_2 = Pipeline(steps=operations)
pruned_tree_2.fit(X_train,y_train)
# Burada max_leaf_nodes=7 yapıp deneyelim. Default: max_leaf_nodes=None
# Class chat: 7 pure leaf olunca bitiriyor .. Orion hoca: pure olmasa da 7

report_model(pruned_tree_2)
# Skorlar yakınlaşmış oldu yine

######### Criterion
DT_model = DecisionTreeClassifier(criterion='entropy', random_state=101)
operations = [("OneHotEncoder", column_trans), ("DT_model", DT_model)]
entropy_tree = Pipeline(steps=operations)
entropy_tree.fit(X_train,y_train)
# criterion='entropy': entropinin düşük olmasını istiyoruz: default: gini(Genelde gini olmasını tercih ediyoruz)

report_model(entropy_tree)
# Yukardaki skorlarımla aynı skorlar geldi 0.96 ya 1.00
# Entropi ile gini arasında bir fark olmadı burada şu an

########### Max_features, Splitter
DT_model = DecisionTreeClassifier(splitter = "random", max_features=3)
operations = [("OneHotEncoder", column_trans), ("DT_model", DT_model)]
tree = Pipeline(steps=operations)
tree.fit(X_train,y_train)
# splitter = "random" : herhangi bir feature üzerinden böler
# max_features=3     : 6 feature dan(6 feature mız vardı) 3 ünü alacak

report_model(tree)
# Normalde flipper length ti en önemli ama rasgele seçerken island dan başlamış(Alttaki şekil)

########### Find Best Parameters
from sklearn.model_selection import GridSearchCV
operations = [("OrdinalEncoder", column_trans), ("DT_model", DecisionTreeClassifier(random_state=101))]
model = Pipeline(steps=operations)
model.get_params()
param_grid = {"DT_model__splitter":["best", "random"],
              "DT_model__max_features":[None, "auto", "log2", 2, 3,4, 5,6, 7],
              "DT_model__max_depth": [None, 2, 3, 4, 5],
              "DT_model__min_samples_leaf": [1, 2, 3, 4, 5, 6,7],
              "DT_model__min_samples_split": [2, 3, 5, 6, 7,8,9]}
# Parametrelere bir çok değer verdik. Bu maliyetli tabi ama bu maliyeti azaltmak istiyorsak max_depth e öncelik vermeliyiz
# Bunlar üzerinden grid search yapalım

grid_model = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          scoring='f1_micro',   # accuracy de yazabilirdik ama aynı şeylerdi(Sonuçlar aynıydı)
                          cv=10,
                          n_jobs = -1)

grid_model.fit(X_train2, y_train2)  # modelimizi eğittik

grid_model.best_score_
grid_model.best_params_  # Bunların hangisi default sonucu getirdi hangisi benim değiştirdiğim değeri getirdi
# Bunu öğrenmek için alttaki kodu kullanabiliriz

grid_model.best_estimator_ # default ında olmayan parametreleri getiriyor
# Bunlar;
# DecisionTreeClassifier(min_samples_split=9,
# Bir tek bu default hariç seçilmiş

eval_metric(grid_model, X_train2, y_train2, X_test2, y_test2)
# Bu hypeparametrelerle test 0.99, train 0.99 .. Gayet iyi 

# Cross validation yapmamıza gerek yok ama yine de bakalım. Skorumuz 0.98 gelmiş. İyi
operations = [("OneHotEncoder", column_trans), ("DT_model", DecisionTreeClassifier(min_samples_split=9, random_state=101))]
scoring = ["accuracy", "precision_micro", "recall_micro", "f1_micro"]
model = Pipeline(steps=operations)
scores = cross_validate(model, X_train2, y_train2, scoring = scoring, cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]

################ Final Model
X = df.drop(columns=["species", "flipper_length_mm"]) 
y = df['species']
# Final modelimi en önemli feature ı(flipper_length_mm) düşürerek kuruyorum. Çünkü böyle yaparak daha iyi sonuçlar almıştık

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
operations = [("OrdinalEncoder", column_trans), ("DT_model", DecisionTreeClassifier(min_samples_split=9,random_state=101))]
pipe_final_model = Pipeline(steps=operations)
pipe_final_model.fit(X, y)
# Final modelimizi kurduk. Prediction alalım

######### Prediction
df.describe().T
samples = {"island": ["Torgersen", "Biscoe"],
           "culmen_length_mm": [39, 48],
           "culmen_depth_mm":[18,14],
           'flipper_length_mm':[180, 214],
           'body_mass_g': [3700,4900],
           "sex":["MALE","FEMALE"]}
# Sample için belirli değerler belirledik describe a bakarak

df_samples = pd.DataFrame(samples)
df_samples

pd.DataFrame(column_trans.transform(df_samples))

pipe_final_model.predict(df_samples)  # Tahmin sonucu

#################### DECISION TREE REGRESSION
# Ares Hoca: Benzer şeyleri yapacağız
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (9,5)
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

df=pd.read_csv("car_dataset.csv")
df.head()
# Arabaların çeşitli özellikleri olan bir datamız var
df.shape
df.isnull().any()
df.describe().T  # selling_price ve present_price da  std>mean .. outlierlar var
sns.heatmap(df.corr(), annot=True)
plt.show()
# 0.88 lik korelasyon görüyoruz ama multicollinearity önemli değil çünkü non-parametrik bir model kullanacağız
df.head()
df["vehicle_age"]=2022-df.Year  # Feature Engineering: Arabanın yaşını hesapladık. Year ı drop edeceğiz
df.head()
len(df.Car_Name.value_counts()) # Bu sütunun çok bir insight sağlamayacağını düşünüyoruz. Drop edelim
df.drop(columns=["Car_Name", "Year"], inplace=True)
df.head()
sns.histplot(df.Selling_Price, bins=50, kde=True)
# Selling_Price da outlier lar vardı. 10 değerinden önceki değerlerde bir yoğunluk var gibi
sns.boxplot(df.Selling_Price)
# Burada da Selling_Price da outlier lar görünüyor. Bunları atarak ya da atmayarak deneyebiliriz ama
# .. amacımız EDA değil burada
df2 = df.copy()

############# Train test split
X=df.drop("Selling_Price", axis=1)
y=df.Selling_Price

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
print("Train features shape : ", X_train.shape)
print("Train target shape   : ", y_train.shape)
print("Test features shape  : ", X_test.shape)
print("Test target shape    : ", y_test.shape)

########### OrdinalEncoder and Categoric Variable
cat = X_train.select_dtypes("object").columns
cat
X_train[cat]

from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
enc.fit_transform(X_train[cat])
# Nominal bir şekilde 0,1,2 şekline dönüştürmüş oldu

X_train[cat] = enc.fit_transform(X_train[cat])

X_train.head()

X_test[cat]=enc.transform(X_test[cat])

X_test.head()

############# Modeling with Pipeline
# Pipeline kuracağımız için baştan başlıyoruz. Yukarıda konuyu açıklamak için işlemler yaptık(Modelling kısmında)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

X_train.head(2)

X_test.head(2)

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
column_trans = make_column_transformer((ord_enc, cat), remainder='passthrough')
# handle_unknown='use_encoded_value' : Bir categorik column da bilinmeyen bir değer geldiği zaman encodlanmış değeri ver Yani --> -1 i ver
# unknown_value=-1 : Verilecek değer
# ord_enc, cat i yukarda tanımlamıştık. Bunlara dönüşüm yapacak
# remainder='passthrough' : Kategorik değişkenlere ord_enc yap. Diğerlerini pass geç

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
operations = [("OrdinalEncoder", column_trans), ("DT_model", DecisionTreeRegressor(random_state=101))]
# ("OrdinalEncoder", column_trans): Kategorik değişkenlere ordinal encoder yapacak ve bilinmeyen bir değer geldiği zaman -1 verecek
# ("DT_model", DecisionTreeClassifier(random_state=101) : Modelimizi uygulayacak
pipe_model = Pipeline(steps=operations)
pipe_model.fit(X_train, y_train)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_val(model, X_train, y_train, X_test, y_test):
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    scores = {"train": {"R2" : r2_score(y_train, y_train_pred),
    "mae" : mean_absolute_error(y_train, y_train_pred),
    "mse" : mean_squared_error(y_train, y_train_pred),                          
    "rmse" : np.sqrt(mean_squared_error(y_train, y_train_pred))},   
    "test": {"R2" : r2_score(y_test, y_pred),
    "mae" : mean_absolute_error(y_test, y_pred),
    "mse" : mean_squared_error(y_test, y_pred),
    "rmse" : np.sqrt(mean_squared_error(y_test, y_pred))}}  
    return pd.DataFrame(scores)

pd.options.display.float_format = '{:.3f}'.format

train_val(pipe_model, X_train, y_train, X_test, y_test)
# Classification modellerde 1.00 çıkmasını garipsemiyoruz ama regression modellerinde 1.00 çıkmasını
# .. direk overfit olarak yorumlayabiliriz
# Zaten test skoru 0.93 ama CV yapınca daha da düşecek

from sklearn.model_selection import cross_validate, cross_val_score
operations = [("OrdinalEncoder", column_trans), ("DT_model", DecisionTreeRegressor(random_state=101))]
model = Pipeline(steps=operations)

scores = cross_validate(model, X_train, y_train, scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv =10)
df_scores = pd.DataFrame(scores)
df_scores.mean()[2:]
# CV yapınca 0.87 çıkmış. Overfitting i açıkça görüyoruz
# Orion hoca NOT: burada label tranformationa gerek yok.yapsanızda yapmasanızda sonuç değişmez
#  Tree based modellerde outlier ları atsak bile skorlar çok iyileşmeyecek. Bunu göreceğiz

######### Removing Outliers
# Ares Hoca: Tree base modellerde scaling yapsanız da yapmasanızda sonuç değişmez
from yellowbrick.regressor import PredictionError
from yellowbrick.features import RadViz
visualizer = RadViz(size=(720, 3000))
model = pipe_model
visualizer = PredictionError(model)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show();
# Outlier ları atsak modelde(skorlarda) bir değişme(iyileşme) olur mu ?

from yellowbrick.regressor import ResidualsPlot
visualizer = RadViz(size=(1000, 720))
model = pipe_model
visualizer = ResidualsPlot(model)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show();     
# Bunlara bakmasakta olur ama göz atalım. X=10 un altındaki değerlerde model hata yapıyor
# .. buna göre outlier ları atacağız

len(df[df.Selling_Price > 10])

28/301
df_new = df[df.Selling_Price < 10]
df_new

# Benzer şeyleri yapacağız
X = df_new.drop(columns="Selling_Price")
y = df_new.Selling_Price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

operations = [("OrdinalEncoder", column_trans), ("DT_model", DecisionTreeRegressor(random_state=101))]
pipe_model = Pipeline(steps=operations)
pipe_model.fit(X_train, y_train)

train_val(pipe_model, X_train, y_train, X_test, y_test)
# Yukarda outlierları atmadan skorlarım 1 e 93 dü. Atarak da skorlar aynı geldi
# Çünkü tree base modeller outlier lardan etkilenmiyor

############# Visualizing trees
from sklearn.tree import plot_tree
list(X_train.select_dtypes("object").columns)
list(X_train.select_dtypes("number").columns)

features = list(X_train.select_dtypes("object").columns) + list(X_train.select_dtypes("number").columns)
features
# Ağacı çizdirmek için categorik ve nümerik column ları birleştirdik

pd.DataFrame(pipe_model["OrdinalEncoder"].fit_transform(X_train))

X_train.head(1)

plt.figure(figsize=(12,8), dpi=150)
plot_tree(pipe_model["DT_model"], filled=True, feature_names=features); #feature_names=X.columns
# Burada çok detaylı bilgi görünmüyor. Yorum yapamıyoruz

# train ve testler için predict ler alıp train_val ile değerlendirme yapıp sonra ağaç çizdireceğiz bu fonk. ile
def report_model(model):
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    print('\n')
    print(train_val(model, X_train, y_train, X_test, y_test))
    print('\n')
    plt.figure(figsize=(12,8),dpi=100)
    plot_tree(model["DT_model"],filled=True, feature_names=features, fontsize=10); #feature_names=X.columns

operations = [("OrdinalEncoder", column_trans), ("DT_model", DecisionTreeRegressor(max_depth=3, random_state=101))]
pruned_tree = Pipeline(steps=operations)
pruned_tree.fit(X_train,y_train)

report_model(pruned_tree)
# Default değerlere göre overfitting i çözdük. Skorlar birbirine daha yakınlaşmış oldu
# Sadece max_depth değil de diğer parametreleri de deneyelim(değiştirirsek neler olur diye) grid search ile
# class chat soru: hocam test sonucu değişmiyorsa burada overfittingi çözmenin bize ne faydası var?
# Orion: Overfitting i gidermeniz gerekiyor. Modelimizin genelleme yapmasını isteriz. O yüzden buradaki sonuç daha iyi

########### GridSearch
from sklearn.model_selection import GridSearchCV
operations = [("OrdinalEncoder", column_trans), ("DT_model", DecisionTreeRegressor(random_state=101))]
pipe_model = Pipeline(steps=operations)

param_grid = {"DT_model__splitter":["best", "random"],
              "DT_model__max_depth": [3,4,5],
              "DT_model__min_samples_leaf": [1, 2,3],  # Bir yaprakta bulunması gereken min örneklem sayısı
              "DT_model__min_samples_split": [2,3, 4], # Bir düğümün bölünmesi için min örneklem sayusı
              "DT_model__max_features":[5, 6, None]}

grid_model = GridSearchCV(estimator=pipe_model, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5, n_jobs = -1)

grid_model.fit(X_train,y_train)

grid_model.best_estimator_ # Default parametreler dışındaki seçilen parametreleri gösteriyordu
# Default dışında seçilmiş olan parametreler(grid search sonucunda) : max_depth=5, min_samples_split=3
grid_model.best_params_

grid_score =pd.DataFrame(grid_model.cv_results_)
grid_score
# Grid search içinde bütün parametrelerin kombinasyonu denendiği sonucunda gelen sonuçlar 
grid_model.best_index_
grid_model.best_score_
grid_score.loc[146]  # 146. index için üstteki tablonun dökümü

train_val(grid_model, X_train, y_train, X_test, y_test)
# Skorlarımız hem arttı hem de birbirine yakınlaşmış oldu. Modelimiz şu an artık genelleme yapabilecek diyebiliriz

y.mean()
0.636/3.4  # Hataya yüzdesel olarak bakalım. Selling pricedaki hatam 0.18

######### Cross Validation
from sklearn.model_selection import cross_validate, cross_val_score
operations = [("OrdinalEncoder", column_trans), ("DT_model", DecisionTreeRegressor(max_depth=5, min_samples_split=3, random_state=101))]
model = Pipeline(steps=operations)
scores = cross_validate(model, X_train, y_train, scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv = 5)
df_scores = pd.DataFrame(scores)
df_scores.mean()[2:]
# CV sonucunda 0.90 gelmiş. Önceki 0.95 ti(Altta). Overfitting var mı acaba diye yüzdesel hataları karşılaştıralım

train_val(grid_model, X_train, y_train, X_test, y_test)

0.636/3.4 
0.778/3.4  

############## Feature İmportance
# Classification ile benzer şeyleri yapacağız
operations = [("OrdinalEncoder", column_trans), ("DT_model", DecisionTreeRegressor(max_depth=5, min_samples_split=3, random_state=101))]
pipe_model = Pipeline(steps=operations)
pipe_model.fit(X_train, y_train)

X_train.head(1)

pd.DataFrame(pipe_model["OrdinalEncoder"].fit_transform(X_train))

list(X_train.select_dtypes("object").columns)
list(X_train.select_dtypes("number").columns)

features = list(X_train.select_dtypes("object").columns) + list(X_train.select_dtypes("number").columns)
features

df_f_i = pd.DataFrame(data = pipe_model["DT_model"].feature_importances_, index=features, 
                      columns = ["Feature Importance"]).sort_values("Feature Importance", ascending=False)
df_f_i

ax = sns.barplot(x = df_f_i.index, y = 'Feature Importance', data = df_f_i)
ax.bar_label(ax.containers[0],fmt="%.3f")
plt.xticks(rotation = 90)
plt.tight_layout()
# Modelim 3 feature ile insight sağlayabiliyormuş
# Ares hoca: En önemli feature ı atıp skorlara bakabilirsiniz

########### Compore real and predicted result
y_pred = grid_model.predict(X_test)
my_dict = { 'Actual': y_test, 'Pred': y_pred, 'Residual': y_test-y_pred }
compare = pd.DataFrame(my_dict)

comp_sample = compare.sample(20)
comp_sample

comp_sample.plot(kind='bar',figsize=(15,9))
plt.show()

########### Final Model
X=df_new.drop("Selling_Price", axis=1)
y=df_new.Selling_Price
X.head()

from sklearn.pipeline import Pipeline
operations = [("Ordinalcoder", column_trans), ("DT_model", DecisionTreeRegressor(max_depth=5, min_samples_split=3,                                                                                  random_state=101))]
pipe_model = Pipeline(steps=operations)
pipe_model.fit(X, y)
# Grid search sonucundaki hyperparametrelerimizi yazdık # DecisionTreeRegressor(max_depth=5, min_samples_split=3

########## Prediction
X.head()

samples = {"Present_Price": [7, 9.5],
           "Kms_Driven": [25000, 44000],
           "Fuel_Type":["Petrol", "Diesel"],
           'Seller_Type':['Dealer', 'Individual'],
           'Transmission': ['Manual', 'Automatic'],
           "Owner":[0,1],
           "vehicle_age":[1, 9]}

df_samples = pd.DataFrame(samples)
df_samples

pipe_model.predict(df_samples)
# 7.000	25000	Petrol	Dealer	Manual	0	1 özelliklerine sahip arabanın fiyatı 5.865 miş 
# 9.500	44000	Diesel	Individual	Automatic	1	9 özelliklerine sahip arabanın fiyatı 5.4075 miş 

# Class chat soru:Hocam yukarıda sordum ama yeniden sormak durumundayım. Bazı notebooklarda daha çok ccp_alpha üzerinde durmakta ve budama işlemini daha çok bu parametre üzerinden gerçekleştirmekte. Bunun bizim uyguladığımız max_depth, min_sample_leaf ve min_samples_split'ten farkı nedir???
# Ares hoca: Overfitting i biz max_depth i değiştirerek hallettik. ccp_alpha parametresini bilmiyorum deneyebilirsiniz

#%%
################LESSON 13####################
###############################################################################
#RANDOM FOREST (RF)(Bagging in özel bir hali)
#ENSEMBLE METHODS:
#Bir çok base model kullanarak ortaya bir meta_model çıkartmaktadır
#Yani;birçok ağaç oluşturulacak.Burdan  gelen sonuçların hepsi birleştirilerek 
#bir tane sonuç bulunur 
#*Bagging(Random Forest)  *Bossting(Ada Boost)

#Bagging ve Bossting:
#Ortak yöleri:>İkisi de voting üzerine çalışır.Yani mesela 100 ağaç gelir,hepsinin,bir 
#oranda oyu olur
#İkisinde de tek bir algoritma var 

#Farklılıkları:>Baggingde bütün modeller ayrı ayrı oluşturulur.(Bütün desicion treeler birbirinden bağımsız)
#Boosting de ise bir modelin hatasından diğer model haberdardır
#Bu yüzden bogging de her bir base modelin meta_odel üzerinde 1 oyu vardır.Boosting  de 
#iyi modellerin oyu fazladır

#Yani Ensemble ethods da bir sürü küçük basa model oluşturulur,bunlar bir araya toplanırlar,
#aralarından bir karar çıkar.Meta model bu toplantıdan çıkan karara göre 
#son karara varır.
#Bagging ile Boostingin bu karara gitme şekilleri farklıdır 

#Bagging=>Base modeller birbirinden habersiz(Her model 1 oy)
#Boosting=>Base-modellerin birbirinden haberi var (iyi modelin oyu fazla)
#!!!!!Bagging=>Strong algoritalar kullanır
#!!!!!Boosting=>Week algoritalar kullanır

#Tarihsel Gelişimi:
#Decision Tree Model in variance ilgili problemlerinden dolayı;variance sorunu 
#ile başa çıkabilmek için "bagging" ortaya atılmış ve bunun ile variance kontrol edilebilmiş
#Daha sonra Random Forest .ıkmış.Random Forest,tabanında Bagging kullanır ama 
#decision tree nin de bazı özelliklerini değiştirerek modelin 
#variance ını o da oynar ve daha güzel bir sonuç elde eder

#Daha sonra Boosting algoritmaları ortaya çıkıyor 
#Daha sonra Gradient Boosting Algoritması ortaya çıkmış.Bunun arkasında Gradient Descent Algoritası çalışır
#Enson olarak da bunların en iyi hali olan XGBoost Algoritaso geliştirilmiştir

#Bootstrapping("yeni data setleri oluşturma işlemi"):Datanın içinde,datanın boyutunda 
#subsamplelar oluşturur.
#Datadan bir örnek çeker,geri atar,çeker,geri atar(Duplicate de olabilir)
#Bu şekilde subsample lar oluşturur(classidier)
#Oluşan subsample lar aynı büyüklüktedirler ama farklı şekildedirler.Bu da varianceı
#kontrol etmemizi sağlar 
#Agregating kısmında gelir ve hepsine aynı "classifier"ı verir.Oluşan base classifier lar eta classifier üzerindeki oylarını kullanır
#Base classifierlar ile dataya farklı açılardan bakmış oluruz
#1-Modeller için subsample oluşturulur
#2-Bu subsample ları classifier lara yada regression algoritmalarına verilir
#3-Burdan çıkan sonuçlar 
#*Classification VOTING(MODE) üzerinden karar verir
#Regression ise AVERAGE(MEAN) üzerinden karar verir

###RANDOM FOREST
#Şidiye kadar Bagging etodundan bahsettik.Random Forest da bagging üzerine 
#kurulmuştur,fakat aralarında farklar vardır
#Random Forest,Decision tree kullanıyor,bu bagging olur;random forest olmaz

#Bagging ile Random Forest Farkı
#1-Subsample oluştururken subsample boyutu 2/3 tür.(Baggin de 3/3 tü)
#2-Featurelar arasında da seçim yapılır,hepsi alınmaz.
#"Verdiğim featurelar arasından şu kadar feature al" denir.Her seferinde farklı 
#featurelar seçtiği için bütün ağaçlar birbirinden farklı olur 
#Böylece he giren data değişir he de datanın üzerinde çalışacağı featurelar değişir
#Bütün ağaçların Root Node ları farklı olur.Bu şekilde Decision Tree nin sona kadar 
#gitesini engellemiş oluyoruz(Variance sona kadar gitmez)
#Sonrasında bütün subsamplelar birer prediction yapar ver bunların ortalaması alınır 

#Özetle:
#1-Subsample alarak datanın 2/3'sini alır 
#2-Aynı Decision Treeler oluşmasın diye bütün featureların belli bir kısını random olarak alır
#3-Böylece her seferinde Root node değişir 
#!Bütün subsample lar birer oya sahip
#4-Bu şekilde varianceı kontrol ederek Decision Treebnin overfittinge gitme problemini ortadan kaldırır.
#(Bazı featureların öne çıkması engellenir.)
#Root Node he subsample üzerinden hede featurelar üzerinden değiştirileye çalışılır
#(Her bir ağaç birbirinden farklı olsun)

#SWM, Rando Forest a göre daha uzun sürer

###HYPERPARAMETERS:
    
#1-n_estimators:"Kaç tane ağaç kurulsun?(default=100)"
#Çok ağaç olursa işlem çok uzun sürer

#2-ax_depth:Dalları budama işlemi.("Kaç adım gideyim?")(default=node)
#3-max_feature:"Featurelardan rasgele kaçını alayım"
#4-min_samples_split:"Bölmek için bir dalda kaç sample bulunsun?"(default=2)

#Olumlu yönleri:
#*Multi-class sorunlarında çok iyi
#*Hızlı bir şekilde prediktion alınabilir
#*Feature iportance ı çok güçlü

#Olumsuz Yönleri
#1-küçük datalar için kötü
#Maliyeti yüksektir.Güçlü cihaz gerektirir 

############################ ENSEMBLE LEARNING
# Ensemble Learning Random forest ve decision tree den yararlanan bir model
# Bagging kısmını bu gün
# Boosting kısmını yarın göreceğiz
# Ensemble: Birliktelik/Birleştirmek
# Önceden bir modelimiz vardı. Ensemble da birden fazla aynı metodları beraber kullanıyoruz
# Örneğin 100 tane decision tree kullanalım. Sonra ben bunları birleştireyim diyoruz
# Bunlar karışıkta(farklı modeller) seçilebilir ama genelde aynı modeller seçiliyor
# Alttaki karikatürde hepsinin bulduğu bir sonuç var. Bunları combine ediyoruz

# En son elde edilen modele metamodel de deniyor
# Burada 2 tane akım var kullanılan Bagging ve Boosting metodları

# Bahsettiğimiz şeyleri şekil olarak gösteren bir slide

# Temelde ayrıldıkları nokta. Biri paralel bir seri
# Bagging : Birisi input datayı alıyor bir sürü ondan yeni datalar üretiyor sonra farklı modeller sokuyoruz ve birleştiriyoruz. Modeller bağımsız çalışır burada
# Boosting: Burada da bir seri yapı var yani input datayı alıyor bundan bir şey üretip modele sokuyoruz o bir sonuç üretiyor
# .. sonra o sonuçtan bir data alıp sonra tekrar modele sokuyoruz sonra o tekrar bir sonuç üretiyor vs vs...
# Bagging en çok kullanılan yöntemleri: Random Forest
# Boosting en çok kullanılan yöntemleri : XGboost, Gradient Boost

# Anlattıklarımızı farklı bir şekilde gösterilmiş

# Karar verirken oylama yaparak output çıkarıyor
# ikisinde de modeller genelde same type(1000 modelin hepsi RF olsun gibi)
# Bagging: Modellere eşit ağırlık veriyor
# Boosting: Bir modelin katkısını performansına göre ağırlıklandırır. 

# Bagging Nedir? --> Bootstrap Aggregating
# Bootstrap işlemi yapılıyor sonra da aggregating işlemi yapılıyor
# Bootstrapte , torba içinde bilyeler var bu torbadan bilyeler çekip bir şey türetip bilyeleri tekrar içine atıyoruz
# .. sonra tekrar yeni bilyeler çekip ondan bir şey üretip tekrar torba içine atıyoruz
# Böylelikle bir sürü(Farklı) datasetler oluşuyor(100 tane dataset)
# Bu datasetler ile modellerimizi eğitiyoruz. Sonuçlarını oylama ile aggregate ediyoruz(birleştiriyoruz)
# Ortaya Ensemble classifier çıkıyor

# Aynı şeyi farklı bir şema ile görüyoruz altta
# Classification ise voting yapıyoruz
# Regression da ise ortalamasını alıyoruz
# Decision tree leri kullanarak kullanarak bu modelleri birleştirirsek buna biz random forest diyoruz
# Bagging ensemble ın en meşhur modeli random forest tır
# Yani Decision tree leri ensemble learning leri kullanırsanız model olarak
# Yani hem örneklerden sample hem featurelardan sample alırsak buna RF diyoruz
# John Hoca: Burada(ensemble da) KNN de kullanabilir , SVM de kullanılabilir vs vs
# Orion hoca: bagging classifier diyip altına estimator olarak istediğiniz modeli verebilirisniz
# Orion hoca: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html?highlight=bagging#sklearn.ensemble.BaggingClassifier

# 2. ders
# Benzer bilgiyi farklı şekilde gösteriyoruz
# Mesela ilk adımda 2 satıdan 2 kırmızıdan almış , almadıkları Turkuaz(B) ve Pembe(D) imiş vs ...
# John hoca: out-of-bag setleri test amaçlı kullanılabiliyor. Bu tür yaklaşımlarda karşınıza çıkabilir

# Ama bagging yöntemlerinden en meşhuru Random Forest tır. Genelde bununla karşılaşırsınız
# 2001 yılında ortaya çıkmış
# Gerçek data sette 100 feature varsa bundan 30 tane feature seçiyoruz random ve random sample seçilerek subsetler oluşturuluyor
# Farklı modeller (ensemble için) kullanılınca buna stacking yaklaşımı denir

# HYPERPARAMETERS
# n_estimators: Kaç modelimiz olacak
# max_depth : Kaç tane node eklensin. Kaç level olsun. RF ile aynı. Genelde overfitting i engellemek için kullanılan parametre
# max_features: Split yaparken kaç feature dan yararlansın
# min_samples_split: Split yapabilmesi için min kaç tane sample olması lazım
# bootstrapbool: False olursa verisetinin hepsini kullanır yani bootstrap i kullanmaz
# oob_scorebool: oob=out of bag(Set)(yukarda bahsedilmişti) . Bootstrap yapıldıktan sonra 
# .. out of bag ler validation için kullanılsın mı? True-False

# class chat : estimator ı 1 yapmakla bootstrap false yapmak aynı mı? 
# .. John Hoca: Değil. Bootstrap data ile ilgili estimatorla ilgili değil. Konseptler farklı

# Avantajları
    # Easy data preparation Missing value ları handle edebiliyor
    # Scaling e gerek yok
    # Outlier ları handle ediyor
    # Feature importances ları ranking yapabiliyoruz. Bu önemli
    # Decision tree ye göre overfitting için bu daha sağlam
# Dezavantajları
    # Küçük datasetlerinde iyi değil. Bu gün örnek göreceğiz
    # Çok büyük ağaçlarda computational cost yüksek olabilir.
    # Classification da performans olarak, regression a göre daha iyi çalışır
    # Yorumlama da sıkıntılar yaşanabilir. John Hoca: Çok da öyle diyemeyiz   
# Class chat soru: random forest alt yapisinda sonucta decisin tree kullaniyor. 
#peki burda decison tree regression icin de classificationda oldugu gibi iyi degil diyebilirmiyiz?
# Orion Hoca: Regression da outlierlar sonuçları etkiliyor. classificationda değil.  
#Decision tree için de geçerli

# NOT: Önce regression notebook undan başlanıldı anlatılmaya. Sonra classification

"""
Burada amaç veri kümesini birden fazla küçük parçaya bölüp her parçadan farklı 
bir karar ağacı oluşturmak ve sonrasında da bu karar ağaçlarının sonuçlarını birleştirmektir.
Majority Vote: Birden fazla tahmin algoritması çalışıyor ve çoğunluğun tahminini alır. 
Sınıflandırma da çoğunluğa bakılır. Prediction da tahmin edilen değerlerin ortalaması alınır.
Karar ağaçlarında veri neden küçük parçalara bölünüyor. Verinin tamamını gördüğünde daha 
iyi sonuç vermez mi? Karar ağaçlarında verinin artması durumunda başarının düştüğünü gösteren sonuçlar vardır.
Karar ağacındaki dallanma overfitting e gider. Bu bir tehlikedir. Ağaçlar dallanıp 
budaklandığında performans süresi artar. Birden fazla ağaç oluşturma farklı bir bakış açısı sağlanıyor. 
Random Forest veriyi farklı açılardan ele alarak sonucu birleştirir.
"""

#%%%
##############NOTEBOOK 13##############################

#################### Random Forest - Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (9,5)
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df=pd.read_csv("car_dataset.csv")
df.head()
# John Hoca: Önceki ders ile alttaki şeyler ortak burayı hızlı geçeceğiz

df.shape 
#(301, 9)
# Az bir data. Az datada neler olacağını göreceğiz
df.info()
df.isnull().any()
df.describe().T # std>mean ler var
sns.heatmap(df.corr(), annot=True)
plt.show()
df.head()
df["vehicle_age"]=2022-df.Year # Feature Engineering
df.Car_Name.value_counts()
df.drop(columns=["Car_Name","Year"], inplace=True)
df.head()

sns.histplot(df.Selling_Price, bins=50, kde=True)
# 15 ten sonrasını outlier olarak kabul edebiliriz
sns.boxplot(df.Selling_Price)

############## Train test split
X=df.drop("Selling_Price", axis=1)
y=df.Selling_Price

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=101)
print("Train features shape : ", X_train.shape)
print("Train target shape   : ", y_train.shape)
print("Test features shape  : ", X_test.shape)
print("Test target shape    : ", y_test.shape)

############# Modeling with Pipeline
cat = X.select_dtypes("object").columns
cat

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
column_trans = make_column_transformer((ord_enc, cat), remainder='passthrough') # Ordinal encoder la kategorical column ları dönüştür. Kalanları hiç ellemeden geç(passthrough)
# One-hot-encoder yerin OrdinalEncoder daha avantajlı demiştik. 3 konuda 
# .. 1.Hız olarak 2.Performance olarak 3.Feature importance belirlemekte

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
operations = [("OrdinalEncoder", column_trans), ("RF_model", RandomForestRegressor(random_state=101))]
pipe_model = Pipeline(steps=operations)
pipe_model.fit(X_train, y_train)

# If we didn't use pipeline, we would set up the model as follows.
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state=101)
rf_model.fit(X_train,y_train)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
def train_val(model, X_train, y_train, X_test, y_test):   
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train) 
    scores = {"train": {"R2" : r2_score(y_train, y_train_pred),
    "mae" : mean_absolute_error(y_train, y_train_pred),
    "mse" : mean_squared_error(y_train, y_train_pred),                          
    "rmse" : np.sqrt(mean_squared_error(y_train, y_train_pred))},  
    "test": {"R2" : r2_score(y_test, y_pred),
    "mae" : mean_absolute_error(y_test, y_pred),
    "mse" : mean_squared_error(y_test, y_pred),
    "rmse" : np.sqrt(mean_squared_error(y_test, y_pred))}} 
    return pd.DataFrame(scores)

train_val(pipe_model, X_train, y_train, X_test, y_test)
# Sonuçlara bakarak ovrfitting durumu olduğunu söyleyebiliriz
# Buradaki RF default değerleriydi
# Bunu CV yaparak gerçek skorlarına bakalım
# Class chat soru: overfittingden şüphelendiğimizde CV yapmasak ve direkt parametrelerle oynasak olur mu?
# Ares hoca: ("Doğru" işareti koymuş)
# John hoca: Buraya bakarak outlier durumdan bahsedebilir miyiz? --Cevap: mae ile rmse arasındaki farka bakarak tespit edebiliriz
# .. Outlier lar yoksa bunlar birbirine yakın olur. Outlier lar varsa fark büyür

from sklearn.model_selection import cross_validate, cross_val_score
operations = [("OrdinalEncoder", column_trans), ("RF_model", RandomForestRegressor(random_state=101))]
model = Pipeline(steps=operations)
scores = cross_validate(model, X_train, y_train, scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv =10)
df_scores = pd.DataFrame(scores)
df_scores.mean()[2:]
# 0.76 yerine aslında skor 0.88 miş

############ Removing Outliers
# Outlier ları çıkarıp bakalım sonuçlara
from yellowbrick.regressor import PredictionError
from yellowbrick.features import RadViz
visualizer = RadViz(size=(720, 3000))
model = pipe_model
visualizer = PredictionError(model)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show();
# Identity: İdeal olanı bu. Neden ideal olanı bu diyoruz--> Çünkü error 0 .. y_pred=y_actual

from yellowbrick.regressor import ResidualsPlot
visualizer = RadViz(size=(1000, 720))
model = pipe_model
visualizer = ResidualsPlot(model)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show();    

len(df[df.Selling_Price > 10])
28/301

df_new = df[df.Selling_Price < 10] # Outlier ları atıp ve yeni df oluşturmuş olduk
df_new.sample(10)

X = df_new.drop(columns="Selling_Price")
y = df_new.Selling_Price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

operations = [("OrdinalEncoder", column_trans), ("RF_model", RandomForestRegressor(random_state=101))]
pipe_model = Pipeline(steps=operations)
pipe_model.fit(X_train, y_train)

train_val(pipe_model, X_train, y_train, X_test, y_test)
# Outlierlar olmadan sonuçlarımız
# Yorum: Performans yükselmiş. RF normalde outlier ları handle ediyor ama küçük datada çok belirleyici değil bu
# .. Bunu Iris datasında da görmüştük. Datanın kçük olması test sonuçlarında çok büyük değişikliklere sebebiyet verebiliyor
# CV ile gerçek skorumuza bakalım

from sklearn.model_selection import cross_validate, cross_val_score
operations = [("OrdinalEncoder", column_trans), ("RF_model", RandomForestRegressor(random_state=101))]
model = Pipeline(steps=operations)
scores = cross_validate(model, X_train, y_train, scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv = 5)
df_scores = pd.DataFrame(scores)
df_scores.mean()[2:]

############# Visualizing trees
# John hoca: Burada farklı aşamalara bakıyoruz. Bunlar sadece ek bilgi. Normalde ağacı görselleştirme falan karşımıza çıkmaz
# .. modeli kurup, CV ye bakıp, (hyperparametreler bakıp) vs devam edebiliriz
features = list(X_train.select_dtypes("object").columns) + list(X_train.select_dtypes("number").columns)
features

from sklearn.tree import plot_tree
def report_model(model, number_of_tree):
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    print('\n')
    print(train_val(model, X_train, y_train, X_test, y_test))
    print('\n')
    plt.figure(figsize=(12,8),dpi=100)
    plot_tree(model["RF_model"].estimators_[number_of_tree],filled=True, feature_names=features, fontsize=8);

RF_model = RandomForestRegressor(n_estimators=250, max_depth=4, random_state=101) # Max_depth=4 olsun dedik.
operations = [("OrdinalEncoder", column_trans), ("RF_model", RF_model)]
pruned_tree = Pipeline(steps=operations)
pruned_tree.fit(X_train,y_train)

report_model(pruned_tree, 50)   # report_model(model, number_of_tree) # Ellinci ağacı gösteriyor çıktı(Orion Hoca: Buradaki ağaç tüm ağaçlardan sadece biri)
# Performans gayet iyi. Overfitting durumu yok.
# 4 tane dalımız olduğunu görüyoruz
# Class chat soru: modeldeki 250 ağacın ortalaması mı bu ağaç? --> Orion hoca: Hayır. 

########### GridSearch
from sklearn.model_selection import GridSearchCV
operations = [("OrdinalEncoder", column_trans), ("RF_model", RandomForestRegressor(random_state=101))]
model = Pipeline(steps=operations)

param_grid = {"RF_model__n_estimators":[ 64, 128, 250],
              "RF_model__max_depth": [4,5],
              "RF_model__min_samples_leaf": [1, 2, 3],
              "RF_model__min_samples_split": [2, 3, 5],
              "RF_model__max_features":['auto', X.shape[1]/3, 6]}

grid_model = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=5,
                          n_jobs = -1)

grid_model.fit(X_train,y_train)
grid_model.best_estimator_      # Pipeline yapısını gösteriyor
grid_model.best_params_
grid_model.best_score_   # RMSE

train_val(grid_model, X_train, y_train, X_test, y_test)
# Bu değerlerle gayet iyi bir sonuç üretmiş

# Grid search de bulunan sonuçlarla model kuralım
from sklearn.model_selection import cross_validate, cross_val_score
operations = [("OrdinalEncoder", column_trans), ("RF_model", RandomForestRegressor(max_depth=5, n_estimators=128,random_state=101))]
model = Pipeline(steps=operations)
scores = cross_validate(model, X_train, y_train, scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv = 5)
df_scores = pd.DataFrame(scores)
df_scores.mean()[2:]

############# Feature Importance
# 3. Ders
operations = [("OrdinalEncoder", column_trans), ("RF_model", RandomForestRegressor(max_depth=5, n_estimators=128,random_state=101))]
pipe_model = Pipeline(steps=operations)
pipe_model.fit(X_train, y_train)

features = list(X_train.select_dtypes("object").columns) + list(X_train.select_dtypes("number").columns)
features

pipe_model["RF_model"].feature_importances_

df_f_i = pd.DataFrame(data = pipe_model["RF_model"].feature_importances_, index=features,
                      columns = ["Feature Importance"]).sort_values("Feature Importance", ascending=False)
df_f_i

ax =sns.barplot(x = df_f_i.index, y = 'Feature Importance', data = df_f_i)
ax.bar_label(ax.containers[0],fmt="%.3f")
plt.xticks(rotation = 90)
plt.tight_layout()
# Bunlara bakarak owner, transmission featurelarını kullanmaya gerek yok
# John Hoca: Present_price bu kadar etkiliyse başka etkili ne olabilir?
# ... Bu feature a yakın bir feature üretip modele katkı sağlayabilir miyiz diye düşünmeliyiz

############ Feature Selection
X2 = X[["Present_Price", "vehicle_age", "Kms_Driven"]] # En iyi 3 feature ı alalım. Diğerlerini almayalım dedik
X2.sample(10)

X_train,X_test,y_train,y_test=train_test_split(X2, y,test_size=0.2, random_state=101)

# John Hoca: Cat column yok ama önceki akışla aynı devam edelim diye burada cat2 
# yazıyoruz ordina_encoder vs yazıyoruz ama gerek yok aslında
cat2 = []
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
column_trans = make_column_transformer((ord_enc, cat2), remainder='passthrough')
operations = [("OrdinalEncoder", column_trans), ("RF_model", RandomForestRegressor(max_depth=5, n_estimators=128, random_state=101))]
pipe_model = Pipeline(steps=operations)
pipe_model.fit(X_train,y_train)
train_val(pipe_model, X_train, y_train, X_test, y_test)
# 0.96 geldi pipeline sonucumuz. Feature selection öncesi ne kadardı? ---> 0.967(önceki sonuç) , 0.93(CV) 
# Diğer feature ların çok önemli olmadığını görüyoruz
# CV leri karşılaştırmak daha mantıklı. Altta CV ye bakarsak 0.92 bulacağız. Eskisi 0.93(cv) dü. Gayet iyi

pipe_model.feature_names_in_

operations = [("OrdinalEncoder", column_trans), ("RF_model", RandomForestRegressor(max_depth=5, n_estimators=128, random_state=101))]
model = Pipeline(steps=operations)
scores = cross_validate(model, X_train, y_train, scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv = 10)
df_scores = pd.DataFrame(scores)
df_scores.mean()[2:]

############### Final Model
X2.head()

pd.DataFrame(column_trans.fit_transform(X2)).head()

cat

cat2 = []
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
column_trans = make_column_transformer((ord_enc, cat2), remainder='passthrough')
operations = [("OrdinalEncoder", column_trans), ("RF_model", RandomForestRegressor(max_depth=5, n_estimators=128,random_state=101))]
pipe_model = Pipeline(steps=operations)
pipe_model.fit(X2, y)

############# Prediction
samples = {"Present_Price": [7, 9.5],
           "Kms_Driven": [25000, 44000],
           "Fuel_Type":["Petrol", "Diesel"],
           'Seller_Type':['Dealer', 'Individual'],
           'Transmission': ['Manual', 'Automatic'],
           "Owner":[0,1],
           "vehicle_age":[1, 9]}

df_samples = pd.DataFrame(samples)
df_samples

pipe_model.predict(df_samples)

# Class chat soru: hocam daha yüksek score almıştık ama bu son modeli tercih etmemizin sebebi?
# John Hoca: Maliyet açısından ya da websitesine kullanıcının 10 değer girmesi yerine 3 değer girmesi için vs

# Class chat soru: hocam yanlış hatırlamıyorsam biz DT de  en önemli feature u 
#silerek devam etmiştik diğerlerini baskılamasın diye ama burada en önemli 3 feature ile devam ettik
# John Hoca: Denemek lazım. Hangisinde performans iyi ise o seçilebilir ama denenmesi lazım

#######################################################################
#%%
############# Random Forest - Classification
"""
The Data
We will be using the same dataset through our discussions on classification with tree-methods (Decision Tree,Random Forests, and Gradient Boosted Trees) in order to compare performance metrics across these related models.
We will work with the "Palmer Penguins" dataset, as it is simple enough to help us fully understand how changing hyperparameters can change classification results.
Data were collected and made available by Dr. Kristen Gorman and the Palmer Station, Antarctica LTER, a member of the Long Term Ecological Research Network.
Gorman KB, Williams TD, Fraser WR (2014) Ecological Sexual Dimorphism and Environmental Variability within a Community of Antarctic Penguins (Genus Pygoscelis). PLoS ONE 9(3): e90081. doi:10.1371/journal.pone.0090081
Summary: The data folder contains two CSV files. For intro courses/examples, you probably want to use the first one (penguins_size.csv).
penguins_size.csv: Simplified data from original penguin data sets. Contains variables:
species: penguin species (Chinstrap, Adélie, or Gentoo)
culmen_length_mm: culmen length (mm)
culmen_depth_mm: culmen depth (mm)
flipper_length_mm: flipper length (mm)
body_mass_g: body mass (g)
island: island name (Dream, Torgersen, or Biscoe) in the Palmer Archipelago (Antarctica)
sex: penguin sex
(Not used) penguins_lter.csv: Original combined data for 3 penguin species
Note: The culmen is "the upper ridge of a bird's beak"
Our goal is to create a model that can help predict a species of a penguin based on physical attributes, then we can use that model to help researchers classify penguins in the field, instead of needing an experienced biologist
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")
#pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("penguins_size.csv")

df = df.dropna()
df.head()

############ Exploratory Data Analysis and Visualization
df.info()
df.describe().T

for feature in df.columns:
    if df[feature].dtype=="object":
        print(df[feature].unique())

df[df["sex"]== "."]

df.drop(index=336, inplace=True)

for feature in df:
    if df[feature].dtype=="object":
        print(df[feature].unique())

df["species"].value_counts()

ax = sns.countplot(x="species", data = df)
ax.bar_label(ax.containers[0]);

ax = sns.countplot(x="species", data = df, hue = "sex")
for p in ax.containers:
    ax.bar_label(p)

plt.figure(figsize=(12,6))
sns.pairplot(df,hue='species',palette='Dark2')

df.head()

plt.figure(figsize=(8,6))
sns.heatmap(df.select_dtypes("number").corr(),annot=True, cmap='viridis')
plt.title("Correlation Matrix")
plt.show()

############### OrdinalEncoder
#https://bookdown.org/max/FES/categorical-trees.html
#https://towardsdatascience.com/one-hot-encoding-is-making-your-tree-based-ensembles-worse-heres-why-d64b282b5769
############# Modeling with Pipeline
############# Train | Test Split
X = df.drop(columns="species")
y = df['species']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

X_train.head(2)

X_test.head(2)

cat = X_train.select_dtypes("object").columns
cat

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
column_trans = make_column_transformer((ord_enc, cat), remainder='passthrough')

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
operations = [("OrdinalEncoder", column_trans), ("RF_model", RandomForestClassifier(random_state=101))]
pipe_model = Pipeline(steps=operations)
pipe_model.fit(X_train, y_train)

"""
If we didn't use pipeline, we would set up the model as follows.
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=101)
rf_model.fit(X_train,y_train)
"""

############# Model Performance on Classification Tasks
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score,f1_score
def eval_metric(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test) 
    print("Test_Set")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print("Train_Set")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))

############## Random Forest
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import GridSearchCV
help(RandomForestClassifier)

eval_metric(pipe_model, X_train, y_train, X_test, y_test)

operations = [("OrdinalEncoder", column_trans), ("RF_model", RandomForestClassifier(random_state=101))]
model = Pipeline(steps=operations)
scores = cross_validate(model, X_train, y_train, scoring = ["accuracy", "precision_micro", "recall_micro", "f1_micro"], cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]

############ Evaluating ROC Curves and AUC
from yellowbrick.classifier import ROCAUC
model = pipe_model
visualizer = ROCAUC(model)
visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show();                      # Finalize and render the figure

########### RF Model Feature Importance
pipe_model["RF_model"].feature_importances_ # rf_model.feature_importances_

features = list(X_train.select_dtypes("object").columns) + list(X_train.select_dtypes("number").columns)
features

rf_feature_imp = pd.DataFrame(data = pipe_model["RF_model"].feature_importances_, index = features, #index=X.columns
                              columns = ["Feature Importance"]).sort_values("Feature Importance", ascending = False)
rf_feature_imp

ax = sns.barplot(x=rf_feature_imp["Feature Importance"], y=rf_feature_imp.index)
ax.bar_label(ax.containers[0],fmt="%.3f")
plt.title("Feature Importance for Random Forest")
plt.show()

############## Understanding Hyperparameters
from sklearn.tree import plot_tree
def report_model(model, number_of_tree):
    model_pred = model.predict(X_test)
    model_train_pred = model.predict(X_train)
    print('\n')
    print("Test Set")
    print(confusion_matrix(y_test, model_pred))
    print('\n')
    print(classification_report(y_test,model_pred))
    print('\n')
    print("Train Set")
    print(confusion_matrix(y_train, model_train_pred))
    print('\n')
    print(classification_report(y_train,model_train_pred))
    plt.figure(figsize=(12,8),dpi=100)
    plot_tree(model["RF_model"].estimators_[number_of_tree], feature_names=features, #features_names=X.columns
          class_names=df.species.unique(),
          filled = True,
          fontsize = 8);

RF_model = RandomForestClassifier(max_samples=0.5) #The sub-sample size is controlled with the max_samples parameter
operations = [("OrdinalEncoder", column_trans), ("RF_model", RF_model)]
pruned_tree = Pipeline(steps=operations) # pruned_tree = RandomForestClassifier(max_samples=0.5)
pruned_tree.fit(X_train,y_train)
# (max_samples=0.5): Sub_sample oranı. Gerçek data 500 tane ise sub sample lar 250 şer tane olsun

report_model(pruned_tree, 50)

############# Final Model and Prediction
# John hoca: Final modeli grid search ten sonra oluşturmak daha doğru olur. Yapmamak doğru olmaz.
# Bu kadar iyi sonuçlar olmasaydı grid search yapmalıydık
# class chat : belki aynı performansı veren daha az komplex bir model çıkabilir gridsearch ile -- john hoca: Evet 
X = df.drop(columns=["species"])
y = df['species']
X.head(2)

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
operations = [("transformer", column_trans), ("RF", RandomForestClassifier(random_state=101))]
pipe_model = Pipeline(steps=operations)
pipe_model.fit(X, y)

df.describe().T

observations = {"island": ["Torgersen", "Biscoe"], "culmen_length_mm":[39, 48], "culmen_depth_mm":[18, 14],
               "flipper_length_mm":[180, 213], "body_mass_g":[3700, 4800], "sex":["MALE", "FEMALE"]}

obs = pd.DataFrame(observations)
obs

pipe_model.predict(obs)

################ DT and RF Scoring for diabetes dataset
df = pd.read_csv("diabetes.csv")
df.head()

############## Cleaning Outliers
df=df[df.SkinThickness<70]
df=df[df.Glucose>0]
df=df[df.BloodPressure>35]
df=df[df.BMI>0]

df.info()
df.Outcome.value_counts()

############# Train | Test Split and Scalling
X = df.drop("Outcome",axis=1)
y = df["Outcome"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


################## Modelling and Model Performance
########### Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(class_weight = "balanced", random_state=101)
dt_model.fit(X_train, y_train)
eval_metric(dt_model, X_train, y_train, X_test, y_test)
# Train de 1 , Test 0.74. Overfitting durumu var gibi.
# CV de gerçek skorlara bakalım CV ile

model = DecisionTreeClassifier(class_weight = "balanced", random_state=101)
scores = cross_validate(model, X_train, y_train, scoring = ['accuracy', 'precision', 'recall', 'f1'],cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
# Overfittingden kurtarmak için grid search yapalım

param_grid = {"splitter":["best", "random"],
              "max_features":[None, 3, 5, 7],
              "max_depth": [None, 4, 5, 6, 7, 8, 9, 10],
              "min_samples_leaf": [3, 5, 6,7],
              "min_samples_split": [11, 12, 14,15,16,17]}

model = DecisionTreeClassifier(class_weight = "balanced", random_state=101)
dt_grid_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring='recall', n_jobs = -1, verbose = 2).fit(X_train, y_train)

dt_grid_model.best_params_
dt_grid_model.best_estimator_

eval_metric(dt_grid_model, X_train, y_train, X_test, y_test)
# Recall : 0.94 e , 0.89 . Modelde scoring='recall' dedik. Recall ları max etmeye çalışıyor
# Overfitting den kurtulduğunu söyleyebiliriz. En azından ezberleme olmadığını söyleyebiliriz
# class chat soru: Hocam grdisearch de class 0 ın recall değeri epeyce düşmüş. Bu durumda iyi bir classification yapıyor diyebilir miyiz?
# John Hoca: Bu problemle alakalı bir durumdur. Burada target(1 sınıfı) bizim için çok daha önemli. 
# .. Zaten ona göre grid search de yağtığımız için başarılı olduğunu söyleyebiliriz

model = DecisionTreeClassifier(class_weight='balanced', max_depth=4, max_features=3,min_samples_leaf=6, min_samples_split=14,random_state=101, splitter='random')
scores = cross_validate(model, X_train, y_train, scoring = ['accuracy', 'precision', 'recall', 'f1'], cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.iloc[:,2:]

df_scores.mean()[2:]
# En doğru sonuçlar bunlar
# Grid search içinde cv yapmak daha doğru olurdu

from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve, plot_roc_curve, roc_auc_score, roc_curve, average_precision_score
plot_precision_recall_curve(dt_grid_model, X_test, y_test);
# Çok iyi bir performansı olduğu söylenemez (0.42)
# Recall u arttırdık yukarıdaki sonuçlarda ama Precision çok düştü o yüzden sonuç bu şekilde
# Grid search de scoring ='F1' kullansaydık burada performans daha iyi olurdu 

############ Random Forest
# Benzer şeyleri yapıyoruz
rf_model = RandomForestClassifier(class_weight = "balanced", random_state=101)
rf_model.fit(X_train, y_train)
eval_metric(rf_model, X_train, y_train, X_test, y_test)
# Default parametreler ile sonuçlar
# Overfitting olduğu görünüyor

model = RandomForestClassifier(class_weight = "balanced", random_state=101)
scores = cross_validate(model, X_train, y_train, scoring = ['accuracy', 'precision', 'recall', 'f1'], cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]

df_scores

param_grid = {'n_estimators':[64, 128, 200],
             'max_features':[2, 4, "auto"],
             'max_depth':[2, 3, 4],
             'min_samples_split':[2, 3,4],
             'min_samples_leaf': [2,3,4],
             'max_samples':[0.5, 0.8]} # add 1

model = RandomForestClassifier(class_weight = {0:1, 1:4}, random_state=101)
rf_grid_model = GridSearchCV(model, param_grid, scoring = "recall", n_jobs = -1, verbose=2).fit(X_train, y_train)

rf_grid_model.best_params_
rf_grid_model.best_estimator_
eval_metric(rf_grid_model, X_train, y_train, X_test, y_test) # Skorlar daha iyi. RF nin daha iyi sonuçlar verdiğini söyleyebiliriz

model = RandomForestClassifier(class_weight={0: 1, 1: 4}, max_depth=2, max_features=2, max_samples=0.8, min_samples_leaf=2, n_estimators=128, random_state=101)
scores = cross_validate(model, X_train, y_train, scoring = ['accuracy', 'precision', 'recall', 'f1'],cv = 5)
df_scores = pd.DataFrame(scores, index = range(1, 6))
df_scores

df_scores.mean()[2:] # RF recall: 0.97 burada .. Önceki model de 0.91 di ..

plot_precision_recall_curve(rf_grid_model, X_test, y_test); # Önceden AP=0.41 di. Burada da performans iyileşmiş

# Random forest sonuç olarak ensemble learning olduğu için Decision tree ye göre daha iyi sonuç verdiğini söyleyebiliriz
# Hyperparametreler tune edilerek ideal sonuçlar alınabilir

#%%
################################################################################
####################LESSON 14########################
###############################################################################

##XGBOOST MODEL 
#Ensemble Methods(Boosting->Subsample,Bagging->Random Forest):Küçük weak 
#learner lar oluşturarak,bunlardan meydana gelen meta model ile daha iyi predictionlar
#elde etmek
#Bagginde her bir ağaca yeni bir model girer ama Boostingde aynı datayı ağaçlar 
#sırayla ele alır.Yani tek data üzerinden çalışır(ana data)(subsample olayı yok)
#Baggingde her bir model bağımsız olduğu için her birinin oyu eşit.Ama boostingde
#hepsi aynı datayı ele aldığı için ağırlıklı olanın oyu fazladır

#bagging variance ı düşürmeye çalışır (piller paralel bağlı gibi)
#boosting Bias ı düşürmeye çalışır (piller seri bağlı gibi)
#Boostingde bir modelden diğerine bilgi aktarımı vardır.

#boostingde errorlar ağırlıklandırılır.Bir sonraki adımda modelin bilemediklerine daha çok 
#ağırlık verilir
#Ağırlıklandırmadan kasıt;mesela datada yanlış bilinen değer 9 olsun.Birdaki modelde 
# ağırlıklandırılır.Yani 9 2 kere 3 kere yazılır 

#1-ADA BOOST ALGORITHMA:
#HYPERPARAMETERS
#base_estimator->default=None(Decision Tree kullanılır)
#n_estimator>Arka arkaya eklenecek ağaç adedi(default=50)
#!!Logistic,SVM.. herhangi bir model kullanılabilir 

#2-GRADIENT BOOSTING(GBM):
#Ağırlıklandırma üzerinden çalışmaz.Bir önceki modelin yaptığı hataları iyileştire 
#üzerinden çalışır.
#Gradient Descent gibi residualları aşağı çekmeye çalışarak hataları aşağı doğru çekerek sıfıra 
#yaklaştımaya çalışır.Tamamen sıfır yaparsa overfitting olur.Bu yüzden sıfır
#yapmak yerine sıfıra yaklaştırmaya çalışır
#Sıfıra ne kadar yaklaştıracağına ağaç sayısı ile karar verir.

#X i illk Tree ye sokar burda residual oluşur.Burdan sonra taget(y) ile bir işi
#kalmaz.Artık residualları bir dahaki ağaca aktarır ve o residualları tahmin 
#etmeye çalışır.Bu şekilde yeni çıkan residual ı bir sonraki ğaca ileterek,
#ağaç sayısı kadar işleme devam eder.Her ağaçta residualları azaltıp sıfıra 
#yaklaştırmaya çalışır ve optimum yerde bırakır 
#(y'yi sadece ilk Tree de residual hesabı için kullanır ve sonrasında residualları
#aşağı çekme hedefi ile devam eder.y'yi bir daha kullanmaz)

#Residual sıfıra kadar giderse zaten y'yi bulmuş olur.Bu da overfitting e sebep olur ki bunu 
#istemeyiz

#3-XGBoost(Extreme Gradient Boosting):
#Bir önceki Gradient Algoritması üzerine çalışarak XGBoost geliştirilmiştir
#*Öncekinden farklı olarak buna regularization eklenmiştir 
#Diğer modelde Tree sayısı ayarlanmazsa son adıma kadar gidip overfitting e 
#sebeb olabiliyordu
#XGBoost kendi içinde bir regularization uygular.(hata ekleme)
#Diyelim ki data da missing valuelar var ve çok hızlı sonuç almak istiyoruz.Missing valueları
#temizleeden XGBoost uygulanabilir
#Missing valueları hata olarak kabul eder ve diğer ağaca atar.Ama missing valuelar 
#baştan temizlenip verilirse çok daha iyi sonuçlar verir.Bu yüzden model öncesi
#EDA işlemi utlaka yapılmalıdır 
#!!!Bir ağaçtan diğerine geçerken kendi içinde Cross Validate işlemi yapar.Böylece
#extra Cross Validate işlemine gerek kalmaz.Biz yinede Cross Validate ve GridSearch
#yapmalıyız(hyperparametreler için)
#*Bütün treeleri paralelleştirir.Böylece işlemler hızlanış olur.
#*İşlemleri cache seviyesinde yaptığı için bir hız da burdan kazanırız 
#*Overfit e gitmeden kısa sürede yüksek skorların alınabildiği bir modeldir. 
#*Normal Gradient Boosting ile arasında mükemmel skor farkları yoktur ama ciddi bir 
#süre farkı vardır.(Random Forest ile de çok büyük bir skor farkı yoktur)
#!!!Logistic Regression,XGBoost tan daha hızlıdır

#XGBosst Hyperparameters:
#n_estimator:default=100
#subsample:default(bölmeden bütün veriyi kullan)=1.0
#max_depth:default=3(Ağaç ne kadar aşağı gitsin)
#learning_rate:default:0.1(Hear ağaç ne kadar katkıda bulunsun)

#####XGBoost faydaları:
#Büyük datalarda hızlı
#EDAsız bile hızlı güzel sonuçlar
#Feature Importance yapar
#Model performansı çok iyi 
#Çok fazla hyperparametresi var.Bunlarla oynanarak model iyileştirilebilir.

#####Negatif Yönleri:
#Görselleştirme yok(Çünkü açıklaması zor)
#Çok fazla hyperparametresi olduğu için yönetmesi zor.Biri bozulunca diğeri bozulabiliyor

######Özetle;
##Bagging:::-->>>>
#*Mesela Random Forest modelde;ana datadan subsampler üretir.Bunların her birine bir classifier 
#veriri,Treeler oluşur.Feature ların içinden de subsamplelar alır.Böylece hiçbir
#ağaç birbirine benzemez.Böylece varyansı azaltıp overfittingle mücadele eder 

##Boosting:::-->>>>
#Bütün data modele girer(Subsample yok)İlk model datayı manipüle eder,bir karar verir
#Burdan çıkan manipule edilmiş data sonraki modele gider,ilkinin üzerinden bir 
#karar verir.Bu şekilde sona kadar gidip bi prediction yapılır.Her modelde oluşan 
#ağırlıklara göre meta model karara gider 

#%%
####NOTEBOOK 14 cohort-9
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

#Bir internet sitesinde yayinlanan bir reklamin, o sitede gezen musteriler 
#tarafindan tiklanip tiklanmadigini analiz ediyor.
#Musterinin o sitede gecirdigi sure, kisilerin yaslari, kullanicinin 
#bulundugu bolgedeki yillik gelir, kisinin gunluk internet kullanim suresi, 
#reklamin basligi, sehir, cinsiyet, bulundugu ulke, kullanicinin siteyi 
#terkettigi saat bilgileri verilmistir. Target label olarak da kisinin 
#reklami tiklayip tiklamadigi bilgisi verilmistir.
df = pd.read_csv('advertising2.csv')
df.head()

#Exploratory Data Analysis and Visualization
df.info()
df.describe()   # Mean std degerlerinde sikinti yok. 
sns.pairplot(df, hue='Clicked on Ad') #veriler iyi ayrılmış

#Train | Test Split
for feature in df.columns:
    if df[feature].dtype=="object":
        print(feature, df[feature].nunique())
    
# Her bir kategorik sütunda kaç tane unique değer var
from sklearn.model_selection import train_test_split
cat = df.select_dtypes("object").columns
cat
list(cat)
cat2 = list(cat) + ['Clicked on Ad']
cat2

X = df.drop(columns=cat2)
y = df['Clicked on Ad']
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

X_train.head(1)
# Datamızın son hali

####ADABOOST CLASSIFIER
#Modelling and Model Performance
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_validate


def eval_metric(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    
    print("Test_Set")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print("Train_Set")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))
#Random Forest modelde her agac birbirinden bagimsizdir, her agacin verdigi 
#oy esittir.En son oylar sayilir ve oylar en fazla hangi class' a ait cikarsa tahmin o class' a atilir.
#Burda agaclar birbirini etkiler. Birinden cikan bilgi diger agaca aktatilir.
#base_estimator=DecisionTreeClassifier(max_depth=1)
####PARAMETERS

#BASE_ASTIMATOR -----> (Default= None) 'None' ise DecisionTreeClassifier 
#kullanir ve max_depth=1 olarak kabul eder.(Derinlik, yani asagi dogru bir kere dallansin.)
#Yani tek bir koku 2 tane de yapragi olur. Bu yuzden weak learner' dir. 
#max_depth ne kadar fazla olursa o kadar dallanirdi vr tahminler iyilesirdi.
#N_ESTIMATORS -----> Arka arkaya eklenecek agac sayisi (default=50)
#LEARNING_RATE -----> Bir sonraki agaca gozlemleri aktarirken ne kadar 
#agirliklandirma yapilacagina karar verir. (Default=1). Default'un 1 olmasi demek 
#gozlem sayisini bir sonraki agacta cok artir, bu yanlis tahmini bir an once bil 
#demek. Bu durumun modeli overfitting' e goturme tehlikesi var. Mesela 0.5 oldugu 
#zaman cok daha az artirir yani daha az agirliklandirma yapar.
#learning_rate duserse agac sayisi artar. Bu yuzden agac sayisi ile 
#learning_rate arasinda bir oranti olmali. Istenen durum; learning_rate 
#dusuk olsun, agac sayisi yuksek olsun. Default=1' deki gibi cok agresif davranmasin.
#! Bu modelin baska parametresi yoktur. Ama asagida goruldugu gibi 
#DecisionTreeRegressor import edilip olusan base_model AdaBoostClassifier' in 
#icine verilebilir ve boylece onun icindeki parametreler ile oynanabilir. 
#akat bu agac ne kadar kuvvetlendirilirse "weak learner" mantigindan 
#cikacaktir ve modeli daha kolay overfitting' e goturecektir. Mesela 
#max_depth=2 yapilabilir ama daha fazla artirmak boosting' in mantigina 
#ters duser. Ihtiyac duyuldugunda en fazla 2 yapilabilir ama buna da 
#gerek yok cunku bu modelin daha ust versiyonu olan Gradient Boosting ve XGBoost ile bu islemi yaparlar.

ada_model = AdaBoostClassifier(n_estimators=50, random_state=42)
ada_model.fit(X_train,y_train)

#Asagida cikan skorlarin birbiri ile dengeli gibi fakat Train ile Tes set 
#arasinda biraz fark var. Overfitting baslangici var diyebiliriz. 
#Bunun icin asagida parametrelerle oynayabiliriz..
eval_metric(ada_model, X_train, y_train, X_test, y_test)

#CrossValidate sonucu asinda skorlarin Train set sonuclarina yakin oldugunu 
#goruyoruz. Aldigimiz tek seferlik sonuclar bizi yaniltmis :

####Cross Validate
from sklearn.model_selection import cross_val_score, cross_validate

model = AdaBoostClassifier(n_estimators=50, random_state=42)

scores = cross_validate(model, X_train, y_train, scoring = ["accuracy", "precision", "recall", "f1"], cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
#test_accuracy    0.960
#test_precision   0.969
#test_recall      0.951
#test_f1          0.959

#Tree Visualization
from sklearn.tree import plot_tree
model = AdaBoostClassifier(n_estimators=3, random_state=42) # n_estimators=3 olsun şeklinde tanımlamışız
model.fit(X_train,y_train)

# Normal şartlarda bu adım yok ama ek bilgi olsun, farkındalık olsun diye gösteriyoruz
targets = df["Clicked on Ad"].astype("str")
plt.figure(figsize=(12,8),dpi=150)
plot_tree(model.estimators_[2], filled=True, feature_names=X.columns, class_names=targets.unique());
#Burada 1000 datamiz var. Butun data bir sonraki agaca tamamen aktarilir ama hepsi 
#ayni degildir. Ilk agacta hata yaptigi datalari agirliklandirir yani bir sonraki 
#modelde onlari birden fazla kere alir. Bir dahaki agacta isleme giren data sayisi 
#degismiyordu. Bunu da su sekilde ayarlar : Zaten daha once bildiklerinin 
#bazilarini bir dahaki modele almaz, onun yerine bilemediklerini daha fazla 
#sayida tekrar tekrar yazar. Yani duplicate yapar. Bu sekilde bir agacta 
#yanlis tahmin ettigi veriyi bir sonraki agacta tahmin etme olasiligi artar.

#Yani her agac bir onceki agactan aldigi datada tahmin edilemeyenlerin sayisini 
#artirir. Agirliklandirma islemini bu sekilde yapar.
#Skorlamayi ise soyle yapar ---> Asagida verilen ornekteki bilgilere gore 
#target class' ina nasil karar verdigini inceleyelim :

#Yukaridaki agacin kokunde Daily Time Spent on Site < 177.505 mi diye soruyor. 
#Degerimiz 177' den kucuk oldugu icin sol dala gider ve oradaki class tahmini 1 imis.

#Yukaridaki kodda model.estimators_[0] yerine 1 yazip ikinci agaca gecelim. 
#O da Daily Internet Usage ,= 64 mu diye soruyor. Cevap hayir oldugu icin sag 
#tarafa gitti. Buradaki tahmin class' i 0 mis.

#Kodda model.estimators_[0] yerine 2 yazarsak ilk kokte yas cikiyor. 
#Age <= 42.5 mu diye soruyor. Gozlemdeki yas 35' ti buna evet dedik sol dala gittik. Yine

#Bu 3 agactan aldigimiz degerlere gore Random Forest her agactan bir oy toplardi 
#bir tane 1; 2 tane 0 degeri ile bu gozlem icin class secimini 0 olarak yapardi. 
#Fakat AdaBoost agirliklandirma mantigi ile calisir. Asagidaki gibi model.
#estimatorerrors dersek bize her agacin yuzdesel olarak errorlerini dondurur. 
#Bu deger sifira ne kadar yakinsa o kadar mukemmel tahminler yapmis demektir.
model.estimator_errors_
#array([0.09666667, 0.15660636, 0.29526371])
#Her agacin AdaBoost uzerinde agirliklandirma formulu vardir. Bunu bizim 
#yerimize Python yapar fakat mantigini anlamak icin asagidaki islemlere bakalim. 
#log' un icinde 1' de cikardigimiz kisim, estimatorerrors' da buldugumuz degerdir. 
#Deger formulde yerine kondugunda ilk agacta 1 sinifina ait agirliklandirma 
#katsayisini 1.11 vermis; diger iki agacta 0 sinifina ait agirliklandirma 
#katsayilarini 0.84 ve 0.43 vermis. 0 sinifina ait olanlari kendi arasinda 
#toplar; (0.84 + 0.43 = 0.127). Bu deger ilkinden buyuk oldugu icin verdigimiz 
#gozlem icin class'i sifir olarak secer.

#Yani bazen tek bir agac, agirliklandirma katsatisi fazla oldugu icin diger 4' unun onune gecebilir.

1/2*np.log((1-0.09666667)/0.09666667) #1
#1.117411476360216
1/2*np.log((1-0.15660636)/0.15660636) #0
#0.8418492023096668
1/2*np.log((1-0.29526371)/0.29526371) #0
#0.43497739343711583

##Analyzing performance as more weak learners are added.

#n_estimator'a gore (agac sayisi), AdaBoostClassifier' da aldigimiz error_rate' 
#leri asagida gorsellestirelim. Her agac icin f1 skoru 1' den cikardik bu da 
#bize erroru verdi. Burdan sunu goruyoruz; belli bir noktaya gelip en dusuk 
#skoru aldiktan sonraki skorlar asagi yukari birbirine yakin olacaktir. Burda 
#en dusuk skoru 20-40 arasinda almis sonraki skorlardacok fazla bie degisim olmamis. 
#Bu yuzden GridSearch' de n_estimators sayisina 250-500 gibi buyuk araliklar da 
#verebiliriz. (Asagidaki sekilde degerler arasinda cok fark varmis gibi gorunuyor 
#ama y eksenine baktigimizda aralarinda sayisal olarak cok da bir fark olmadigini gorebiliriz.)
error_rates = []

for n in range(1,100):
    
    model = AdaBoostClassifier(n_estimators=n)
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    err = 1 - f1_score(y_test,preds)
    
    error_rates.append(err)
plt.plot(range(1,100), error_rates);
#Yukaridaki grafikte 20-30 arasindaki iyi olan degeri belki yakalariz 
#diye 20 ve 30 gibi kucuk sayilari da ekledik.


###Gridsearch

from sklearn.model_selection import GridSearchCV
model = AdaBoostClassifier(random_state=42)
param_grid = {"n_estimators": [20, 30, 100, 200], "learning_rate": [0.01, 0.1, 0.2, 0.5, 1.0]}
ada_grid_model = GridSearchCV(model, param_grid, cv=5, scoring= 'f1')
ada_grid_model.fit(X_train, y_train)

ada_grid_model.best_params_
#learning_rate default degeri 1 idi, n_estimators default degeri ise 50 idi. 
#Ama modelimiz default degerlerle hard tahminler yaparak hemen sonuca ulasmak 
#yerine bu yeni degerlerle daha soft tahmin yapacak, agac sayisi artacak.
#Bu da bizi overfitting' den kurtaracak.

ada_grid_model.best_score_
#0.969132103588408
#Yukarida Grid Search sonrasi aldigimiz skor f1 skorudur.Conku yukarida GridSearch 
#modelini kurarken skor olarak f1 skorunu sectik. GridSearch' den sonra tekrar bir 
#CrossValidate islemine gerek yok, istege bagli.

#Her model sonrasi aldigimiz f1, recall, roc_auc skorlarini karsilastirmak amaci 
#ile asagidaki degiskenleri tanimladik :

y_pred = ada_grid_model.predict(X_test)

ada_f1 = f1_score(y_test, y_pred)
ada_recall = recall_score(y_test, y_pred)
ada_auc = roc_auc_score(y_test, y_pred)
eval_metric(ada_grid_model, X_train, y_train, X_test, y_test)
#GridSearch sonucu alinan skorlarda oncekine gore pek bir degisim yok. 
#Sadece f1 skoru Test setinde %94 cikmis. Ama GridSearch' un CrossValidate' i 
#f1 skorunu __ada_grid_model.bestscore=%96_ olarak bulmus, yani 
#burada overfitting durumu olmadigina karar veriyoruz.

#####Feature_importances
#AdaBoostClassifier' a gore tahminlemeye en fazla katkisi olan feature' i secelim. 
#Daha sonra asagida diger modellerin feature importance' lari ile kiyaslama yapacagiz.

model = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
model.feature_importances_

feats = pd.DataFrame(index=X.columns, data= model.feature_importances_, columns=['ada_importance'])
ada_imp_feats = feats.sort_values("ada_importance")
ada_imp_feats
#Bu modelde cinsiyetin target' a hicbir katkisinin olmadigini goruyoruz.

plt.figure(figsize=(12,6))
sns.barplot(data=ada_imp_feats ,x=ada_imp_feats.index, y='ada_importance')

plt.xticks(rotation=90);

###Evaluating ROC Curves and AUC
#! Modelin genel performansina mutlaka bakilmali !!

from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve, plot_roc_curve, roc_auc_score
###Modelin genel performansi %99, yani AdaBoost modelden guzel bir skor elde ettigimizi soyleyebiliriz.

plot_roc_curve(ada_grid_model, X_test, y_test);

####Gradient Boosting Modelling and Model Performance

#!! Linear modellerin arti yonleri ile Tree modellerin arti yonlerinin birlesimi. 
#Feature importance'i XGBoost modele gore zayiftir. XGBoost model bunun bir ustu.!!

#Gradient Boosting Model' in Ada Boostingile calisma mantigi tamamen farklidir.
#Agirliklandirma uzerinden calismaz. Bir onceki modelin yaptigi hatalari iyilestirme 
#uzerine calisir. AdaBoost' da prediction actual degere yaklastirilmaya calisiliyordu; 
#Gradient Boosting ise residual' lari (y-ypred) 0' a yaklastirarak gercek deger ile 
#tahmin degeri arasindaki farki kapatmayi amaclar.

#Gradient Boosting ile AdaBoost' un yaptigi sey aynidir fakat calisma mantiklari farklidir. 
#Gradient Boosting residual (hatalar) uzerinden calisarak her yeni agacta onlari 0' a yaklastirmaya calisir.

#Yukaridaki data orneginde feature ' lara gore Heart Disease belirlenmeye calisiliyor. Gradient 
#Boosting her gozlem icin baslangic olasiligi belirler. Yukaridaki target labelda 4 
#tane 1 degeri var. Buna gore logaritmik bir hesaplama ile 1 sinifinin 0 sinifina 
#oranini hesaplayarak ornegin 0.69 gibi bir deger bulur. Bu cikan degeri, Logistic 
#Regression' in olasilik formulune sokar ve bu degeri bir olasiliga donusturur. 
#Mesela bu deger 0.67 gibi bir olasiloga donusur. Bu bizim baslangic 
#olasiligimiz olur. Bundan sonra modele predict ettirecegimiz butun gozlemler 
#icin baslangic olasiligi 0.67 olur. Bunun anlami; bir predicti egitim icin ilk 
#agaca verdigimizde 0.67 olasilikla 1 class' ina ait olacak. (Treshold=0.5 idi).
#b Yani butun gozlemler yoluna 1 ile baslayacak.
#Ilk agac icin residual hesaplamasini su sekilde yapar :
#1 classi icin residual farki --->> 1 - 0.67 = 0.33
#0 classi icin residual farki --->> 0 - 0.67 = -0.67
#Yukaridaki degerler ilk agac icin residual degerleridir. Yukaridaki data 
#orneginde son sutuna bakarsak; 1 sinifindaki her gozlem icin 0.33, 0 sinifindaki 
#her gozlem icin ise -0.67 residual degerleri yazilmis.

#Yukaridaki gozlemler uzerinden residual' lari bir sonraki agacta tekrar tahmin 
#etmeye calisir yani residual'lari minimize etmeye calisir. Asagida bir sonraki 
#agacta residual degerlerinin dustugunu goruyoruz. Residula' lar bu sekilde her 
#yeni agacta biraz daha 0 degerine yaklasir.

np.log(4/2)  
#0.6931471805599453
#Baslangictaki olasilik (1 classinin 0 classina orani). Her hasta 0.67 olasilikla 
#kalp hastasi olarak yola basliyor :

(np.e**np.log(4/2))/(1+np.e**np.log(4/2))   # Logistic Regression' in olasilik 
#formulu (Her gozlem icin baslangic olasiligi)
0.6666666666666666
0.67 + (0.1* 0.33) + (0.1* 0.13)    # 1 sinifi icin hesaplama (Buradaki 0.1 learning_rate default degeri)
0.7160000000000001
#Belirlenen learning_rate ile o agactaki residual farklarini carparak her agactan 
#gelen degeri toplar. Yukarida bu islemi 3 agac icin yapmis ve %71.6 degerini 
#bulmus. Yani 0.5' den buyuk bir deger bukdugu icin bu gozlemi 1 classina atip kalp hastasi diyecektir.

0.67 + (0.1* -0.67) + (0.1* -0.44)  

# 0 sinifi icin hesaplama. - deger aldigi icin amaci 0' a yaklastirmak. 0.5' in 
#altina dusememis. Agac sayisi yetersiz kalmis.
# Bu yuzden hata yapti, gozlem 1 sinifina gitti.
0.5589999999999999
#Yukaridaki orneklerde goruldugu gibi Gradient Boosting, her classa ait olanlarin 
#baslangic olasiliklarini belirleyerek (buradaki ornetkte 0.69) arka planda Logistic 
#Regression' in olasiliklarini kullanarak 1 class' indakileri 1' e, 
#0 class' indakileri 0' a yaklastirmaya calisir. Bu sekilde residuallari 
#minimize etmeye calisarak gercek ile tahmin arasindaki farki kapatir ve her 
#agacta dogru tahmine dogru adim adim gider. Belli bir agac sayisindan sonra 
#da residullar sabitlenir ve model minimum global hatayi bulmus demek olur; artik egitime ihityac yoktur.

from sklearn.ensemble import GradientBoostingClassifier
grad_model = GradientBoostingClassifier(random_state=42)
 
#PARAMETERS

#LOSS -----> (Default='deviance'). Bu parametrenin aciklama kismina bakarsak 
#deviance=Logistic regression yaziyor. Yani oranlari olasiliga donustururken 
#arka planda Logistic Regression kullanir. Yani Gradient Descent bir model ile 
#Tree based modelleri birlestirir. Fakat burda Logistic Regression icindeki 
#Ridge Lasso gibi parametreleri oynayamayiz. XGBoost modelde bunlari da oynayabilecegiz.

#N_ESTIMATOR -----> (Default=100). Arka arkaya eklenecek agac adedi.

#LEARNING_RATE -----> (Default=0.1) Her agacin ne kadar katkida bulunacagini 
#gosterir. Bu deger buyudugu zaman agac sayisi azalir.

#SUBSAMPLE ------> (Default=1) Yani ilk agaca butun datayi ver. Kaynaklarda 
#GridSearch' de 0.5 ile 0.8' in de denenmesi tavsiye edilir.

#CRITERION -----> (Default='friedman_mse'). Agaclari bolme kriteri. mse' den 
#bir farki yoktur. Bu bir regression parametresidir fakat Friedman tarafindan 
#class' lar icin optimize edilmistir. Classification' da kullanilan Gini yerine 
#bunu kullanir cunku arka planda Gradient Descent tabanli calisiyor.

#MIN_SAMPLE_SPLIT -----> (Default=2). Bolunme icin yaprakta bulunmasi gereken sample sayisi

#MIN_SAMPLES_LEAF ------. (Default=1). Bir yapragin yaprak olarak kabul 
#edilebilmesi icin o yaprakta bulunmasi gereken min gozlem sayisi.

#MAX_DEPTH -----> (Default=3). Agac ne kadar derinlige insin? Default 
#degerinin 3 olmasinin sebebi; agacin daha cok dallanmasi weak learning 
#mantigina ters duser ve modelimiz weak learner bir model. Bu deger bazen 
#2 yapildiginda sonuclar daha da iyilesebiliyor fakat artirmak tavsiye edilmez. 
#Weak learner oldugu icin model daha zayif hale getirildiginde tahminler daha iyi bir hale gelebilir.

#MAX_FEATURE -----> (Default=None). Ornegin 5 degeri girildiginde feature' lar 
#icinden rastgele 5 tane secer ve bunlar icindeki en iyi agaci kok yapar.

#Yukaridaki kirpma hyperparametreleri ile oynamak yerine min_weight_fraction_leaf=0.0 
#ve min_impur'ty_decrease=0.0 gibi sayisal parametreler ile de oynanabilir. 
#Fakat bunlari oynayabilmek icin de arkadaki matematige hakim olmak gerekir. Su asamada oynamak tehlikeli olabilir.

#Modelin overfitting' e gitmemesi icin icerideki diger hyperparametrelerle oynayabiliriz.

grad_model.fit(X_train, y_train)
#Bu modelin default degerleri ile skorlari aldigimizda modelin overfitting' e dogru bir gidisat oldugunu soyleyebiliriz.

eval_metric(grad_model, X_train, y_train, X_test, y_test)


#######Cross Validate

#Cross Validate ile de yukaridaki tek seferlik skorlarimiz birbirine yakin. 
#Asagida GridSearch ile parametrelerle oynayarak Train ile Test set skorlarini birbirine yaklastirmaya calisacagiz.
model = GradientBoostingClassifier(random_state=42)
scores = cross_validate(model, X_train, y_train, scoring = ['accuracy', 'precision','recall','f1', 'roc_auc'], cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
#test_accuracy     0.952222
#test_precision    0.958870
#test_recall       0.945960
#test_f1           0.951603
#test_roc_auc      0.987972
#dtype: float64
#Gridsearch
param_grid = {"n_estimators":[100, 200, 300],
             "subsample":[0.5, 1], "max_features" : [None, 2, 3, 4]} #"learning_rate": [0.001, 0.01, 0.1], 'max_depth':[3,4,5,6]
gb_model = GradientBoostingClassifier(random_state=42)
grid = GridSearchCV(gb_model, param_grid, scoring = "f1", verbose=2, n_jobs = -1).fit(X_train, y_train)
grid.best_params_
#{'max_features': 3, 'n_estimators': 100, 'subsample': 0.5}
#f1 skoruna gore calistirdigimizda en iyi parametre degerlerinin yukaridaki gibi oldugunu goruyoruz.

grid.best_score_     #f1 skoru (GridSearch' den sonra Test skorlari yukseldi.)
#0.9658340599400116
#GridSearch' den sonra Test set skorlarimiz biraz daha yukseldi; Train set ile aradaki variance farki biraz daha kapandi.

y_pred = grid.predict(X_test)

gb_f1 = f1_score(y_test, y_pred)
gb_recall = recall_score(y_test, y_pred)
gb_auc = roc_auc_score(y_test, y_pred)


#####Feature importances
#Bu model icin de feature importance' i aldik ve asagida tum model sonuclarini kiyaslayacagiz.

model = GradientBoostingClassifier(max_features= 3, n_estimators = 100, subsample = 0.5, random_state=42)
model.fit(X_train, y_train)

model.feature_importances_

feats = pd.DataFrame(index=X.columns, data=model.feature_importances_, columns=['grad_importance'])
grad_imp_feats = feats.sort_values("grad_importance")
grad_imp_feats
sns.barplot(data=grad_imp_feats, x=grad_imp_feats.index, y='grad_importance')
plt.xticks(rotation=90);

 
####Evaluating ROC Curves and AUC
#Modelin genel performansi oldukca yuksek (0.99)

plot_roc_curve(grid, X_test, y_test);

###XG Boosting Modelling and Model Performance
#Linear Regression ile Tree based modellerin en guclu yonlerini birlestirerek olusturulmus guclu bir modeldir.

from xgboost import XGBClassifier
xgb = XGBClassifier(random_state=42).fit(X_train, y_train)
###PARAMETERS
#max_depth=3, learning_rate=0.1, n_estimator=100 parametreleri Gradient Boosting 
#ile ayni ve burada da onemli parametreler.

#OBJECTIVE -----> (Default='binary:logistic') Arka planda Logostoc Regression' i 
#kullanir. ! Gredient Boost modelde bu parametrenin adi loss idi. !
#BOOSTER -----> (Default='gbtree') Tree modellerle Gradient tabanli modelleri 
#birlestirmeye yardim eden parametre.
#REG_ALPHA -----> (Default=0) Lasso degeridir (l1)
#REG_LAMBDA -----> (Default=1) Ridge degeridir. (l2) Arka planda Ridge ve Lasso 
#ile regularization islemi de yapabiliriz.
#!!!! Ridge ve Lasso degerleri Gradient Boosting' de yoktu. Bu modelde var. 
#XGBoost model, Gradient Boosting modelin gelistirilmis halidir. !!!!
#Datamizda overfitting oldugu durumlarda lasso degeri ile oynayip modele 
#cok sert mudahalelerde bulunabiliriz veya Ridge degeri ile oynayabiliriz.
#SCALE_POS_WEIGTH -----> (Default=1). Onceki modellerde dengesiz veri setleri 
#icin 'balanced' islemi vardi. Burda onun yerine bu parametre var. (Gradient Boosting' de bu ozellik yok)
#Diyelim ki datada 10 tane bir sinifa ait, 100 tane diger sinifa ait veri var. 
#100/10=10 sonucuna gore scale_pos_weight=10 diyebiriz. Eger skorlar cok iyi 
#cikmazsa 7, 8, 9, 11, 12, 13 gibi degerler de verilip skorlar gozlenebilir.
#MIN_CHILD_RATE -----> (Default=1). Overfitting durumu varsa kullanilabilir; 
#Lasso gibi davranir. 1 den daha buyuk degerler de verilebilir. Feature 
#importance da bu parametreye yuksek degerler verilirse bazi feature' larin
#cok onemsizlestigi veya tamamen sifirlandigi gozlemlenebilir. Bu deger ile 
#oynayarak da overfitting ile mucadele edilebilir. (Biz ornek olarak 100 degerine kadar degerler verdik)
#SUBSAMPLE -----> (Default=1). Her agacta gozlemin ne kadarini kullansin? 0.5 
#ile 0.8 degerleri de GridSearch' de denenmeli.
#COL_SAMPLE_BYTREE -----> (Default=1). max_feature yerine bu parametre var. 
#0 ile 1 arasinda deger alir. Feature' larin yuzde kacini kullanmasini 
#istedigimizi soyleriz. Bu feature'lari rastgele secer, aralarindan en iyisi ile isleme baslar.
#COL_SAMPLE_BYLEVEL -----> (dHer bolme isleminde kullanilan gozlem sayisini 
#degistirmek istersek kullaniriz. Subsample ve col_sample_bytree kullaniliyorsa 
#bu parametreye gerek yok. Kaynaklarda bu parametrenin degistirilmesi tavsiye edilmez.

#COL_SAMPLE_BYNODE -----> (Default=1) Bir agacta en son kalacak max leafe sayisi 
#kac olsun? Default degereinde hepsi demis. Bunun yerine max_depth ile oynamak daha mantikli.

#GAMMA -----> (Default=0) 0 ile sonsuz arasinda bir deger alir. Hangi datada 
#hangi degeri alacagini bilemiyoruz. Bazi datalarinda 1 degeri, bazi datalarda 
#1 milysr degeri overfitting' i engeller. Kaggle yarismalarinda yarismacilar 
#tarafindan cok kullanilir. Diyelim ki overfitting sorunu var, once min_child_rate i 
#kullandik fakat bir sonuc alamadik gamma' ya cok cok buyuk degerler de dahil olmak 
#uzere degerler verip overfitting ile mucadele etmesini saglayabiliriz.

#Bu modelin cok fazla parametresi oldugu icin overfitting durumlarinda GridSearch 
#isleminden once max_depth, min_child_rate, gamma parametreleri kullanilarak 
#manuel olarak overfitting ile mucadele etmeye calisip sonra GridSearch' e gitmek mantikli olur.

#Verbosity -----> Model arkada calisirken rapor yazsin mi yazmasin mi? (Onemli bir parametre degil.)

eval_metric(xgb, X_train, y_train, X_test, y_test)
#Tek seferlik skorlarda overfitting durumu gorunmuyor. Cross Validate sonrasi da skorlar tutarli.

####Cross Validate

model = XGBClassifier(random_state=42)

scores = cross_validate(model, X_train, y_train, scoring = ['accuracy', 'precision','recall',
                                                                   'f1', 'roc_auc'], cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
#test_accuracy     0.957778
#test_precision    0.968757
#test_recall       0.945909
#test_f1           0.956573
#test_roc_auc      0.988441
####Gridsearch
param_grid = {"n_estimators":[50, 100, 200],'max_depth':[3,4,5], "learning_rate": [0.1, 0.2],
             "subsample":[0.5, 0.8, 1], "colsample_bytree":[0.5,0.7, 1]}
xgb_model = XGBClassifier(random_state=42)
#from sklearn.metrics import make_scorer 
#Eger istersek make scorer ile de istedigimiz label' in istedigimiz metrigini make_scorer 
#icine tanimlayip GridSearch islemi yaptirabiliriz.

#xgb_grid = GridSearchCV(xgb_model, param_grid, scoring = make_scorer(precision_score,  average=None, pos_label=1), verbose=2, n_jobs = -1).fit(X_train, y_train)
xgb_grid = GridSearchCV(xgb_model, param_grid, scoring = "f1", verbose=2, n_jobs = -1).fit(X_train, y_train)
#Fitting 5 folds for each of 162 candidates, totalling 810 fits
xgb_grid.best_params_
#{'colsample_bytree': 0.5,
# 'learning_rate': 0.1,
# 'max_depth': 3,
# 'n_estimators': 100,
# 'subsample': 0.8}
xgb_grid.best_score_    # Cross Validate skoru. Asagidaki f1 skoru ile (%94) birbirine daha yakin. Overfitting durumu yok. 
0.9682125077654098
y_pred = xgb_grid.predict(X_test)

xgb_f1 = f1_score(y_test, y_pred)
xgb_recall = recall_score(y_test, y_pred)
xgb_auc = roc_auc_score(y_test, y_pred)

eval_metric(xgb_grid, X_train, y_train, X_test, y_test)

###Feature importances
model = XGBClassifier(random_state=42, colsample_bytree = 0.5, subsample= 0.8)
model.fit(X_train, y_train)

model.feature_importances_

feats = pd.DataFrame(index=X.columns, data=model.feature_importances_, columns=['xgb_importance'])
xgb_imp_feats = feats.sort_values("xgb_importance")
xgb_imp_feats
xgb_importance
sns.barplot(data=xgb_imp_feats, x=xgb_imp_feats.index,y='xgb_importance')

plt.xticks(rotation=90);

#Feature importance comparison
#Yukaridaki butun feature importance sonuclarini asagida kiyaslarsak en dogru 
#sonucu veren modelin XGBoost oldugunu gorebiliriz. Diger modellerin feature 
#importance' larinin cok guvenilir olmadigi goruluyor. XGBoost tum feature'lara 
#agirlik vermis, digerleri bazi feature' lara cok fazla agirlik verirken bazilarina hic vermemisler.

#Kaynaklarda feature importance bakimindan en guvenilir modelin Random Forest 
#oldugu soylenir, ikinci sirada XGBoost geli. Tree based modeller icin Random Forest, 
#boosting modeller icin XGBoost" un feature importance' lari kullanilabilir.

pd.concat([ada_imp_feats, grad_imp_feats, xgb_imp_feats], axis=1)
ada_importance	grad_importance	xgb_importance

####Evaluating ROC Curves and AUC
#Modelin genel performansi oldukca yuksek.

plot_roc_curve(xgb_grid, X_test, y_test);

####Random Forest
#Default degerler ile Random Forest skorlarini alalim :

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train,y_train)
y_pred = rf_model.predict(X_test)
rf_f1 = f1_score(y_test, y_pred)
rf_recall = recall_score(y_test, y_pred)
rf_auc = roc_auc_score(y_test, y_pred)
eval_metric(rf_model, X_train, y_train, X_test, y_test)

#Decision Tree
#Default degerler ile Decision Tree skorlarini alalim : (Default deger 
#kullandigimiz icin overfitting baslangici var gibi gorunuyor.)

from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
dt_f1 = f1_score(y_test, y_pred)
dt_recall = recall_score(y_test, y_pred)
dt_auc = roc_auc_score(y_test, y_pred)
eval_metric(dt_model, X_train, y_train, X_test, y_test)

#Logistic Regression
#Default degerler ile Logistic Regression skorlarini alalim :

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
log_model=LogisticRegression()
log_model.fit(X_train_scaled, y_train)
y_pred=log_model.predict(X_test_scaled)
log_f1 = f1_score(y_test, y_pred)
log_recall = recall_score(y_test, y_pred)
log_auc = roc_auc_score(y_test, y_pred)
eval_metric(log_model, X_train_scaled, y_train, X_test_scaled, y_test)

####KNN
#Default degerler ile KNN skorlarini alalim :

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_scaled, y_train)
y_pred = knn_model.predict(X_test_scaled)
knn_f1 = f1_score(y_test, y_pred)
knn_recall = recall_score(y_test, y_pred)
knn_auc = roc_auc_score(y_test, y_pred)
eval_metric(knn_model, X_train_scaled, y_train, X_test_scaled, y_test)


######SVM
#Default degerler ile SVM skorlarini alalim :

from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)
y_pred = svm_model.predict(X_test_scaled)
svc_f1 = f1_score(y_test, y_pred)
svc_recall = recall_score(y_test, y_pred)
svc_auc = roc_auc_score(y_test, y_pred)
eval_metric(svm_model, X_train_scaled, y_train, X_test_scaled, y_test)


#####Comparing Models
#Karsilastirma icin genel formul : Model isimleri, F1, Recall, Roc_Auc skorlari 
#(Yukarida bunlari degiskenlere atamistik) bir DataFrame icinde verilmis.

#Ikinci def kismi sayilarinnsagda gorunmesi icin.
#plt.subplot(311) : 3--> Satir sayisi; 1---> Sutun sayisi; 1----> Ilk tablo

#Bunun altina da compare' daki skorlari F1 skora gore, 2. tabloda Recall' a gore, 
#3. tabloda ise Roc_Auc'a gore azalan oranda sirala dedik.

compare = pd.DataFrame({"Model": ["Logistic Regression", "KNN", "SVM", "Decision Tree", "Random Forest", "AdaBoost",
                                 "GradientBoost", "XGBoost"],
                        "F1": [log_f1, knn_f1, svc_f1, dt_f1, rf_f1, ada_f1, gb_f1, xgb_f1],
                        "Recall": [log_recall, knn_recall, svc_recall, dt_recall, rf_recall, ada_recall, gb_recall, xgb_recall],
                        "ROC_AUC": [log_auc, knn_auc, svc_auc, dt_auc, rf_auc, ada_auc, gb_auc, xgb_auc]})

def labels(ax):
    for p in ax.patches:
        width = p.get_width()                        # get bar length
        ax.text(width,                               # set the text at 1 unit right of the bar
                p.get_y() + p.get_height() / 2,      # get Y coordinate + X coordinate / 2
                '{:1.3f}'.format(width),             # set variable to display, 2 decimals
                ha = 'left',                         # horizontal alignment
                va = 'center')                       # vertical alignment
    
plt.figure(figsize=(14,10))
plt.subplot(311)
compare = compare.sort_values(by="F1", ascending=False)
ax=sns.barplot(x="F1", y="Model", data=compare, palette="Blues_d")
labels(ax)

plt.subplot(312)
compare = compare.sort_values(by="Recall", ascending=False)
ax=sns.barplot(x="Recall", y="Model", data=compare, palette="Blues_d")
labels(ax)

plt.subplot(313)
compare = compare.sort_values(by="ROC_AUC", ascending=False)
ax=sns.barplot(x="ROC_AUC", y="Model", data=compare, palette="Blues_d")
labels(ax)
plt.show()

#1.satir F1 skoru, 2. satir Recall satiri, 3. satir ise Roc_Auc skorudur.
#Ilk satirda F1 skorlara gore sonuclara baktigimizda, KNN' nin en yuksek skoru verdigini goruyoruz.

#Ikinci satirda Recall skorlari arasinda da en yuksek skorun KNN' de oldugunu goruyoruz. 
#(F1 ve Recall skorlari yuksek ise Precision skorlari da yuksektir diyerek ayrica ona bakmadik.)

#En yuksek KNN gibi dursa da modellerin skorlari genel olarak birbirine cok yakin. 
#Bu yuzden modelin guvenilirligini olcen ROC_AUC skorlarina bakiyoruz. Burada da 
#Logistic Regression one cikmis. Burada KNN veya Logistic Regression tercih edilmeli 
#fakat KNN modelin maliyeti cok yuksek oldugu icin Logistic Regression secmek daha 
#mantikli olacaktir. Cunku en hizli calisan algoritmadir. Skorlar yakin oldugu icin calisma maliyetlerini kiyasladik.

#Feature importance gibi aciklamalar KNN uzerinden yapilamaz. Bu yuzden de tercih 
#etmemek mantikli olur. Logistic Regression' da katsayilara gore hangi feature' in 
#daha onemli oldugunu soyleyebliriz. Aciklamasi cok daha kolaydir. KNN modelin neye 
#gore bir secim yaptigini aciklamak ise zordur.

#XGBoost gibi modelller cok guclu modeller olmasina ragmen, bazen burda oldugu gibi 
#diger modeller on plana cikabilir. Bu yuzden tum modeller mutlaka denenmelidir.


#%%
##########NOTEBOOK 14 cohort-11

#AdaBoosting, Gradientboosting, XGBoosting Regressor
# Aynı işlemleri regression için yapacağız
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (9,5)
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
df=pd.read_csv("car_dataset.csv")
df.head()

df.shape
df.info()

df.isnull().any().any()
df.describe().T
plt.show()

df.head()

df["vehicle_age"]=2022-df.Year
df.Car_Name.value_counts()

df.drop(columns=["Car_Name", "Year"], inplace=True)
df.head()

sns.histplot(df.Selling_Price, bins=50, kde=True)

sns.boxplot(df.Selling_Price)

 
###Train test split
from sklearn.preprocessing import OrdinalEncoder
df_new = df[df.Selling_Price < 10]
df_new.sample(10)

X=df_new.drop("Selling_Price", axis=1)
y=df_new.Selling_Price
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=101)

print("Train features shape : ", X_train.shape)
print("Train target shape   : ", y_train.shape)
print("Test features shape  : ", X_test.shape)
print("Test target shape    : ", y_test.shape)

###Modeling with Pipeline for Adaboost Regressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
def train_val(model, X_train, y_train, X_test, y_test):
    
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    scores = {"train": {"R2" : r2_score(y_train, y_train_pred),
    "mae" : mean_absolute_error(y_train, y_train_pred),
    "mse" : mean_squared_error(y_train, y_train_pred),                          
    "rmse" : np.sqrt(mean_squared_error(y_train, y_train_pred))},
    
    "test": {"R2" : r2_score(y_test, y_pred),
    "mae" : mean_absolute_error(y_test, y_pred),
    "mse" : mean_squared_error(y_test, y_pred),
    "rmse" : np.sqrt(mean_squared_error(y_test, y_pred))}}
    
    return pd.DataFrame(scores)
cat = X.select_dtypes("object").columns
cat
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder


ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

column_trans = make_column_transformer((ord_enc, cat), remainder='passthrough')

# make_column_transformer: Hangi sütunlar kategorik hangileri nümerik bunun bilgisini tutuyor. Ilerde işimize yarayacak
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostRegressor

#base_estimator=DecisionTreeRegressor(max_depth=3)
operations = [("OrdinalEncoder", column_trans), ("Ada_model", AdaBoostRegressor(random_state=101))]

pipe_model = Pipeline(steps=operations)

pipe_model.fit(X_train, y_train)

train_val(pipe_model, X_train, y_train, X_test, y_test)

y_pred_ada = pipe_model.predict(X_test)
y_pred_ada

from sklearn.model_selection import cross_validate, cross_val_score

operations = [("OrdinalEncoder", column_trans), ("Ada_model", AdaBoostRegressor(random_state=101))]

model = Pipeline(steps=operations)

scores = cross_validate(model, X_train, y_train, scoring=['r2','neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv =10)
pd.DataFrame(scores)
pd.DataFrame(scores).mean()[2:]

from sklearn.tree import plot_tree
features = list(X.select_dtypes("object").columns) + list(X.select_dtypes("number").columns) 
features

Ada_model = AdaBoostRegressor(n_estimators=3, random_state=101) # n_estimators=3 : 3 tane ağaç

operations = [("OrdinalEncoder", column_trans), ("Ada_model", Ada_model)]

model = Pipeline(steps=operations)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.figure(figsize=(12,8),dpi=100)
plot_tree(model["Ada_model"].estimators_[0], filled=True, feature_names=features, fontsize=7);
# model["Ada_model"].estimators_[0] : 0 nolu ağaç

plt.figure(figsize=(12,8),dpi=100)
plot_tree(model["Ada_model"].estimators_[1], filled=True, feature_names=features, fontsize=7);

plt.figure(figsize=(12,8),dpi=100)
plot_tree(model["Ada_model"].estimators_[2], filled=True, feature_names=features, fontsize=7);

y_pred

np.array(y_test)

 #X_test
X_test.loc[[33]]

pipe_model["OrdinalEncoder"].fit_transform(X_test.loc[[33]])

####Gridsearch for Adaboosting
from sklearn.model_selection import GridSearchCV
param_grid = {"Ada_model__n_estimators":[50, 100, 200, 300, 500],
              "Ada_model__learning_rate":[0.1, 0.5, 0.8, 1],
              "Ada_model__loss": ["linear", "square"]
            }

# "Ada_model__loss": Adaboost a özel bir hyperparameter. Her bir iterasyonda ağırlıkları update ederken kullanılan yöntem ne olmalı(Lineer,square,exp...)
operations = [("OrdinalEncoder", column_trans), ("Ada_model", AdaBoostRegressor(random_state=101))]
model = Pipeline(steps=operations)
grid_model = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=5,
                          n_jobs = -1)
grid_model.fit(X_train,y_train)

grid_model.best_params_

grid_model.best_estimator_

grid_model.best_score_
train_val(grid_model, X_train, y_train, X_test, y_test)

# 0.74 lerde olan rmse yi 0.60 lara indirmiş olduk

#Feature importance
operations = [("OrdinalEncoder", column_trans), ("Ada_model", AdaBoostRegressor(loss='square', random_state=101))]

pipe_model = Pipeline(steps=operations)

pipe_model.fit(X_train, y_train)

pipe_model["Ada_model"].feature_importances_

imp_feats = pd.DataFrame(data=pipe_model["Ada_model"].feature_importances_,columns=['ada_Importance'], index=features)
ada_imp_feats = imp_feats.sort_values('ada_Importance', ascending=False)
ada_imp_feats
ada_Importance

plt.figure(figsize=(12,6))
ax = sns.barplot(data=ada_imp_feats, x=ada_imp_feats.index, y='ada_Importance')
ax.bar_label(ax.containers[0],fmt="%.3f")
plt.xticks(rotation=90);

 
####Modeling with Pipeline for Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor

operations = [("OrdinalEncoder", column_trans), ("GB_model", GradientBoostingRegressor(random_state=101))]

pipe_model = Pipeline(steps=operations)

pipe_model.fit(X_train, y_train)

# 100 -- 150 -- (-50) -- (-30) --> 150 + 0.1 * (-50) + 0.1*(-30)

# 200 -- 150 -- (50) -- (25)  --> 150 + 0.1 * 50 + 0.1*25
train_val(pipe_model, X_train, y_train, X_test, y_test)

operations = [("OrdinalEncoder", column_trans), ("GB_model", GradientBoostingRegressor(random_state=101))]

model = Pipeline(steps=operations)
scores = cross_validate(model, X_train, y_train, scoring=['r2', 
            'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv =10)
pd.DataFrame(scores)
pd.DataFrame(scores).mean()[2:]

#### Gridsearch for Gradientboosting
param_grid = {"GB_model__n_estimators":[64, 128], 
              "GB_model__subsample":[0.5, 0.8], 
              "GB_model__max_features" : [3, 5, 6],
              "GB_model__learning_rate": [0.1, 0.2], 
              'GB_model__max_depth':[1,2]}
operations = [("OrdinalEncoder", column_trans), ("GB_model", GradientBoostingRegressor(random_state=101))]

model = Pipeline(steps=operations)

grid_model = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=5,
                          n_jobs = -1)

# Class chat soru : grid model de çıkan sonuç 5 cv nin ortalaması mı yoksa en iyisi mi--> Johnson Hoca: CV ortalama skor döndürür
grid_model.fit(X_train, y_train)

grid_model.best_params_

grid_model.best_estimator_

grid_model.best_score_
train_val(grid_model, X_train, y_train, X_test, y_test)

# Adaboost a göre sonuçlar daha iyi
# Rmse 0.47, R2:0.97 gelmiş

operations = [("OrdinalEncoder", column_trans), ("GB_model", GradientBoostingRegressor(max_depth=2, max_features=6,
                                           n_estimators=128, random_state=101, subsample=0.5))]

model = Pipeline(steps=operations)

scores = cross_validate(model, X_train, y_train, scoring=['r2', 
            'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv =10)
pd.DataFrame(scores).mean()[2:]

# Hoca derste cv =10 u , cv =5 olarak değiştirdi grid search ile cv sonuçları farklı olduğu için düzeltmek adına
# John Hoca: Notebook baştan çalıştırılıp denenebilir
# Johnson Hoca: bir tanesinde cv=10 diğerinde cv= 5 olduğu için fark ortaya çıkıyor hocam

###Feature importance
operations = [("OrdinalEncoder", column_trans), ("GB_model", GradientBoostingRegressor(max_depth=2, max_features=6,
                                           n_estimators=128, random_state=101, subsample=0.5))]

pipe_model = Pipeline(steps=operations)

pipe_model.fit(X_train, y_train)

pipe_model["GB_model"].feature_importances_

imp_feats = pd.DataFrame(data=pipe_model["GB_model"].feature_importances_,columns=['grad_Importance'], index=features)
grad_imp_feats = imp_feats.sort_values('grad_Importance', ascending=False)
grad_imp_feats
grad_Importance

ax = sns.barplot(data=grad_imp_feats, x=grad_imp_feats.index, y='grad_Importance')
ax.bar_label(ax.containers[0],fmt="%.3f")
plt.xticks(rotation=90);

###Modeling with Pipeline for XG Boost Regressor
#!pip install --upgrade pip
#!pip install xgboost==0.90
import xgboost as xgb


from xgboost import XGBRegressor

operations = [("OrdinalEncoder", column_trans), ("XGB_model", XGBRegressor(random_state=101, 
                                                                           objective="reg:squarederror"))]

pipe_model = Pipeline(steps=operations)

pipe_model.fit(X_train, y_train)

train_val(pipe_model, X_train, y_train, X_test, y_test)
operations = [("OrdinalEncoder", column_trans), ("XGB_model", XGBRegressor(random_state=101, 
                                                                           objective="reg:squarederror"))]

model = Pipeline(steps=operations)

scores = cross_validate(model, X_train, y_train, scoring=['r2', 
            'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv =10)
pd.DataFrame(scores).iloc[:, 2:].mean()

###Gridsearch for XGBoost
param_grid = {"XGB_model__n_estimators":[100, 300],
              "XGB_model__max_depth":[1, 2], 
              "XGB_model__learning_rate": [0.01, 0.05, 0.1],
              "XGB_model__subsample":[0.5, 1], 
              "XGB_model__colsample_bytree":[0.5, 1]}
operations = [("OrdinalEncoder", column_trans), ("XGB_model", XGBRegressor(random_state=101, 
                                                                           objective="reg:squarederror"))]

model = Pipeline(steps=operations)

grid_model = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=5,
                          n_jobs = -1)
grid_model.fit(X_train, y_train)

grid_model.best_params_
{'XGB_model__colsample_bytree': 1,
 'XGB_model__learning_rate': 0.05,
 'XGB_model__max_depth': 2,
 'XGB_model__n_estimators': 300,
 'XGB_model__subsample': 0.5}
grid_model.best_estimator_

grid_model.best_score_
train_val(grid_model, X_train, y_train, X_test, y_test)

###Feature importance
operations = [("OrdinalEncoder", column_trans), ("XGB_model", XGBRegressor(learning_rate=0.05, max_depth=2,
                              n_estimators=300, objective='reg:squarederror', random_state=101,subsample=0.5))]

pipe_model = Pipeline(steps=operations)

pipe_model.fit(X_train, y_train)

pipe_model["XGB_model"].feature_importances_

imp_feats = pd.DataFrame(data=pipe_model["XGB_model"].feature_importances_, columns=['xgb_Importance'], index=features)
xgb_imp_feats = imp_feats.sort_values('xgb_Importance', ascending=False)
xgb_imp_feats
xgb_Importance

plt.figure(figsize=(12,6))
ax = sns.barplot(data=xgb_imp_feats, x=xgb_imp_feats.index, y='xgb_Importance')
ax.bar_label(ax.containers[0],fmt="%.3f")
plt.xticks(rotation=90);

# class chat soru: Hocam domine eden ve zayıf olanları elimine ediyoruz bu durumda # John hoca: Denenebilir

Feature importance comparison
pd.concat([xgb_imp_feats, ada_imp_feats, grad_imp_feats], axis=1)

####Feature Selection
# Mesela bir websitesinde kullanıcının sadece 3 bilgi girip bir sonuç almasını sağlamak için feature selection yapabiliriz
X_new = df_new[["vehicle_age",  "Present_Price", "Seller_Type"]]
X_new.head()

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=101)
cat2 = ["Seller_Type"]

ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

column_trans = make_column_transformer((ord_enc, cat2), remainder='passthrough')


operations = [("OrdinalEncoder", column_trans), ("XGB_model",XGBRegressor(learning_rate=0.05, max_depth=2,
                              n_estimators=300, objective='reg:squarederror', random_state=101,subsample=0.5))]

pipe_model = Pipeline(steps=operations)
pipe_model.fit(X_train,y_train)
train_val(pipe_model, X_train, y_train, X_test, y_test)

operations = [("OrdinalEncoder", column_trans), ("XGB_model",XGBRegressor(learning_rate=0.05, max_depth=2,
                              n_estimators=300, objective='reg:squarederror', random_state=101,subsample=0.5))]

model = Pipeline(steps=operations)

scores = cross_validate(model, X_train, y_train, scoring=['r2', 
            'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv =10)
pd.DataFrame(scores).iloc[:, 2:].mean()

# Modeli 3 feature ile de kurduğumuzda 0.93 e düşmüş hatamız. Biz modeli 3 feature ile mi 6-7 feature ile mi yapalım bu sizin vereceğiniz karar

###Final Model
X_new.head()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer

cat2 = ["Seller_Type"]

ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

column_trans = make_column_transformer((ord_enc, cat2), remainder='passthrough')


operations = [("OrdinalEncoder", column_trans), ("XGB_model",XGBRegressor(learning_rate=0.05, max_depth=2,
                              n_estimators=300, objective='reg:squarederror', random_state=101,subsample=0.5))]

pipe_model = Pipeline(steps=operations)
pipe_model.fit(X_new, y)

pd.DataFrame(column_trans.fit_transform(X_new)).head()

####Prediction
samples = {"Present_Price": [7, 9.5],
           "Kms_Driven": [25000, 44000],
           "Fuel_Type":["Petrol", "Diesel"],
           'Seller_Type':['Dealer', 'Individual'],
           'Transmission': ['Manual', 'Automatic'],
           "Owner":[0,1],
           "vehicle_age":[1, 9]}

# Pipe modelimizi normalde 3 feature ile eğitmiştik()"Present_Price"  "vehicle_age"'Seller_Type')
# Ancak pipe model içinde make_column_transform featureları indexlediği için kullanıcı fazladan 
# .. değer girse bile bize bir çıktı veriyor.
# Make_column_transform yapılmamış bir model oluştursaydık hata alırdık. Burada sıra bile önemli değil
# .. önemli olan traindeki sütunların isimleriyle buranın isimleri aynı olması

# Ancak mesela indexte tutulan "Present_Price": [7, 9.5] sütununu samples a vermezsek hata alırız
df_samples = pd.DataFrame(samples)
df_samples

# 2 adet unseen data için bir sonuç alabiliriz

pipe_model.predict(df_samples)
# Istersek pickle dosyasına aktarım yapabiliriz
# John Hoca: Supervised learning i bitirmiş olduk
# Class chat : gradient boosting residuları azaltmak için reziduya neden olan datalara daha fazla mı öğrenme yapıyor?
# 2. modelin görevi o residual i minimize etmeye çalışmak. Residuleri hangi datalar yüksek tutuyorsa onları
# .. classify etmeye çalışıyor. Sonuç olarak modeller seri çalışır bir sonraki model bir öncekinin hatalarını minimize
# .. etmek için çalışır. En sonunda da minimum bir hata ile sonuç elde etmiş oluruz

# Johnson Hoca: Gradient descent tabanlı modeller tahmin edilen değeri her iterasyonda gerçek değere 
# .. yakınlaştırmaya çalışırken, gradient boosting algortiması ise residualleri minimize ederek tahmin
# .. edilen değeri gerçek değere yakınlaştırmaya çalışıyor.



#%%
#################################################################
###LESSON 17
##################################################################
###UNSUPERVISED LEARNING
#Train datası yok,target label yok.Bütündata algoritmaya verilir ve kümeler 
#ortaya çıkar 
#1-Clustering 2-Dimensionality Reduction
#Unsupervised learning,domain knowledge gerektirir

####Clustring
 #*Customer Segmentation:Customerlar gruplandırılır.Sonra bu kümelere bakılarak hangi reklamlar verilir,
#buna karar verilir
 #*Targeted Marketing
 #Recomender Systems 
 
###Dimensionality Reduction
#3 Boyuttan sonra görselleştirme yapılamaz.3 feature dan fazla feature ı olan big dataları
#görselleştirebilmek,daha anlamlı hale getirebilmek için kullanılır

####Clustering
#Benzer özelliği olan şeyleri aynı gruba atarak kümeleme işlemi yapar
#Eğer cluster sayısı çok arttırılırsa birbirine çok benzeyen çok fazla clusterlar
#oluşur ve kümelerin hiçbirisi birbirinden ayırt edilemez 
#Cluster sayısı çok az olursa da alınması gereken bilgi kaçırılmış olur.Fazla genelleme yapar
#Bu yüzden cluster sayısı çok önemlidir 

####K-means Algorithm
#k->Kaç cluster oluşacağını belirler(Supervised learningde komşuluğu belirtiyordu)
#K-means algoritmasının amacı;her bir küe içindekiler birbirine benzesinler ve her küme birbirinden
#farklı olsun 
#K-means algoritması iterasyon mantığı ile çalışır 
#*k sayısı kadar rastgele "centroids" atar
#*bu centroidsler her iterasyonda hareket ederek en iyi yeri bulurlar ve 
#çevresindekileri kendi grubu olarak belirleyip kümelerler
#Cenroids->Clusterların merkezi

#!!!İlk nokta random atandığı için,istenen sonuçlar alınmazsa k-means algoritması birkaç defa çalıştırılabilir
#k-means algoritması random olarak yoluna başladıktan sonra çevresini hesaplaması ve 
#duracağı yeri bilmesi gerekir.Bunu da "Distance Function" ve "Optimization Criteria"
#ile yapar
#Distance Function(Çevre hesabı)
#Optimization Criteria(ALgoritma ne zaman duracak?):Bölgenin varyansının sıfıra 
#en yakın olduğu noktada durur.(çevresindekilerin uzaklığının karesi min olduğunda)

#k Sayısının Tespiti:
#1-Domain Knowledge   2-Data Driven Approach(Elbow method)

#Domain Knowledge:Datayı bilen birisi kaç küme olduğunu söyler ve k sayısı ona 
#göre belirlenir 

#Elbow Method:Optimal k sayısı matematiksel hesaplarla bulunur 

###Clustering Evaluation
#1-Clustering Tendency(Hopkins Test)
#2-Optimal Number of Clusters(Elbow Method)
#3-Clustering Quality(Externel Metric(Domain Knowledge)-Internal Metrics(No domain Knowledge))

###1-Clustering Test
#Hopkings Test:Bir datanın clusterlanıp clusterlanamayacağı hakkında bilgi verir
#İki hipotez oluşturur

#*-Null Hypothesis(Ho):Data,non-random,uniform distibution dır
#(No meaningful clusters)

#*-Alternative Hypothesis(Ha):Data random dağılmıştır.(Presence of cluster)
#X=datasets.load_iris().data 
#hopkins(X,150) =>>>Buran çıkan değer 0.5 in üzerindeyse Ho kabul.
#0.5 in altındaysa Ha kabul 

#Ho:Data random değil,clustering olmaz.(>0.5)
#Ha:Data random,clustering tendency olabilir.(<0.5)


###2-Elbow Method
#Domain knowledge imkanı yoksa bu method kullanılır

###3-Clustering Quality
#En iyi cluster=>>Cluster içindeki mesafe minimum,Clusterlar arası mesafe max.
#Bunun için 2 metrik kullanılır:

    #External Metrics:(Domain Knowledge)
#*Adjusted Rand İndex(biz bunu kullanacağız)
#*Fowlkes-Mallows İndex
#*Jaccard İndex/coefficient

   #Internal Metrics:(No domain Knowledge)
#*Silhoutte Coefficient(biz bunu kullanacağız)
#*Davies-Bouldin İndex
#*Dunn Index

###Adjusted Rand Index:(Benzerlik üzerinden çalışır)
#Clusterların birbirleri ile benzerliği üzerinden bir sayı döndürür.
#(Kümeler birbirleri ile benzer i değil mi?)
#Domain bilgisi olan birisi hepsini bölmesede datanın bir kısmını labelar.
#Bunun üzerine clusterların birbirleri ile olan benzerliği ortaya konur
#R2 scoru 0 veya negatif ise kötü clustering,1 e yakın ise iyi clustering 
#yapılmış demektir.Küme için mesafe kısa,kümeler arası mesafe uzun demektir

###Silhouette Coefficient:(Mesafe üzerinden çalışır):
#Domain bilgisi olan biri yoksa bu method kullanılır
#S=(b-a)/max(a,b) -->1-e yaklaşırsa süper,-1 yaklaşırsa kötü 
#a->Küme için ortalama mesafe 
#b->Diğer clusterlara olan ortalama mesafe 


#%%
#####################NOTEBOOK 17#########################################

####K_Means Clustering
##Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings('ignore')
#pd.set_option('display.max_rows', 500)
#Ciceklere ait yaprak olculerinin verildigi bir data seti. Bu datanin bir 
#label' i yok. Oncelikle domain bilgisi olan birinden bilgi almak gerekir. 
#Boyle birisi yoksa bitkiler hakkinda uzman birine danisilir. O kisiden bu 
#datada 3 farkli bitki oldugu bilgisini alinabilirse; bu 3 kume uzerinden 
#cluster' lar nasil kaliteli hale getirilebilir, bunun uzerine calismak 
#gerekir. Uzman destegi yok ise de en iyi cluster' i yapabilmek icin 
#matematiksel hesaplamalara basvurulur.
#k-means gercek dunya verileri uzerinde gruplandirma yaparken cok 
#basarili olmadigi icin mutlaka bir uzman destegi alinmalidir.

df = pd.read_csv("iris1.csv")
df.head()

df.info()

df.describe()
#Asagidaki pairplot grafiginde verilerin cok net bir sekilde ikiye ayrildigini 
#ve aralarindaki mesafenin fazla oldugunu gorebiliyoruz. Yogun kisimlardan yeni 
#bir kume olusabilir mi, bunu bu grafikten cikaramiyoruz.

sns.pairplot(df)
plt.show()

####Scaling Data
#The K-means algorithm definitely needs scaling. However, if all our features 
#are in the same/similar range, there is no need to scale the data. 
#For these data, scaling may cause worse results in some cases. You must try 
#both with and without scale and continue with whichever one has good results.

####K_Means Clustering
#Bir egitim asamasi yok; benzer paternleri bulma mantigi uzerine calisir. 
#k-means algoritmasi kullanirken mutlaka scale islemi yapilmali, cunku mesafe 
#tabanli calisan bir algoritma. Fakat bu datada sayilar birbirine cok yakin 
#oldugu icin scale islemi skorlari kotulestirdi. Bu yuzden bu data icin 
#scale islemi uygulamayacagiz.

#Egitim icin kullandigimiz bir data yok; butun datayi kullanacagiz.

X = df.copy()
X.head()
#   sepal_length  sepal_width  petal_length  petal_width
#0           5.1          3.5           1.4          0.2
#1           4.9          3.0           1.4          0.2
#2           4.7          3.2           1.3          0.2
#3           4.6          3.1           1.5          0.2
#4           5.0          3.6           1.4          0.2
X.shape
#(150, 4)
from sklearn.cluster import KMeans
K_means_model = KMeans(n_clusters=5, random_state=42)

####PARAMETERS
#N_CLUSTERS -------> Kac kume olussun? (Default=8)
#N_INIT -------> Algoritma ilk bolmeyi random olarak yaptigi icin skorlar kotu cikabilir. 
#Bu yuzden ayni islemi defalarca yaptirmak basari oranini artirir. 
#Bu parametre ile random olarak verilen sayida baslangic noktasi belirlenir. 
#Hangisi en iyi kumelemeyi sagliyorsa algoritma onunla devam eder. 
#(10 kere baslangic noktasi belirle, en basarili hangisiyse onunla yoluna devam et) (Default=10)

#INIT --------> (init='k-means++') Random noktalari verilerin yogunlastigi 
#yerlerin orta noktalarindan secer. En iyi kumeyi bulma islemini hizlandirir.

#MAX_ITER ------> Iterasyon sayisini manuel olarak ayarlamamizi saglar (Default=300). 
#Cok buyuk datalarda 300 yetersiz kalir. Bu durumlarda k-means algoritmasi uyari verir.

K_means_model.fit_predict(X)

#Supervised modellerde oldugu gibi burada ayri ayri fit ve prediction islemi yok. 
#Fit ya sadece asagidaki gibi tek basina kullanilir ya da yukaridaki gibi predict ile beraber kullanilir.
#Sadece fit kullanilirsa, feature' lardaki ozelliklerin birbirine yakinligina gore 
#paternler tespit edilmis olur. Bu islemden sonra predict kullanilamaz. 
#Bunun yerine K_means_model.labels kullanilir. Hangi gozlem hangi 
#kumeye atanmis, bu islemle gorulebilir.
#fit_predict kullanimi tercih edilirse labels kullanmaya gerek yok; 
#hangi kumeye atama islemi yaptigini gosterir. Kullanimi daha pratik oldugu icin biz bunu tercih edecegiz.

K_means_model.fit(X)
KMeans(n_clusters=5, random_state=42)
K_means_model.labels_

#X' e tahmin ettigimiz class' lari yeni bir feature olarak ekleyelim :

X["Classes"] = K_means_model.labels_
X
###Hopkins test
#Data kumeleme islemi icin uygun mu? Random olarak mi dagilmis? Bunun icin Hopkins test yapilir.
#Eger data non random, uniform ise, dummies feature' lardan olusuyor ise 
#noktalarin birbirlerine olan uzakliklari ayni olur ki boyle datalar 
#clustering icin uygun degildir. Eger datada dummy edilmis 
#feature' lar varsa bunlarin atilmasi gerekir.

#Yukarida ekledigimiz class' i cikarip datayi eski haline getirelim :

X = X.drop("Classes", axis =1)
#!pip install pyclustertend
from pyclustertend import hopkins
X.shape
#(150, 4)
#Hopkins testin icine data seti ve sampling_size verilir. Cok buyuk datalarda 
#islem hizini artirmak icin sampling_size yerine tum datayi temsil eden bir orneklem verilebilir. 
#Bu datada gozlem sayisi az oldugu icin tum datayi verecegiz :

hopkins(X, X.shape[0])
#0.16892803397149367
#Hopkins degerinin 0.5' in altinda olmasi, cluster islemi yapabiliriz anlamina geliyor. 0' a 
#ne kadar yakinsa data, cluster islemi icin o kadar uygun demektir. 0.5' in uzerinde 
#deger yukseldikce, datanin clustering islemine uygunlugu azalir. Alinan deger 0.5' in 
#uzerinde ciksa bile mutlaka sonraki islemler de uygulanip skorlar alinmalidir.

#Burda degerimizi 0.16 cikmis. Demek ki random bir dagilim var ve cluster islemi yapilabilir.

hopkins(X, 60)   # Subsample ile de yakin bir skor elde ettik. 
0.17476540775140603
####Choosing The Optimal Number of Clusters
#Elbow metod
#Optimal k sayisi Elbow Metod ile belirlenir.

ssd = []

K = range(2,10)                                          # k icin aralik.

for k in K:
    model = KMeans(n_clusters =k, random_state=42)
    model.fit(X)                                         # Her yeni k sayisina gore modeli egit.
    ssd.append(model.inertia_)

#model.inertia_ --------> Olusturulan her kume icin kume elemanlarinin merkeze 
#olan uzakliklarini olcer ve bunlarin karesini alir. Cikan degerleri toplar ve 
#ortalamasini alir. Kumelerden hesaplanan degerler ne kadar kucukse, kume 
#elemanlari merkeze o kadar yakin demektir. Inertia degeri yuksek cikarsa kume 
#elemanlari genis alana yayilmis demektir.

#mse hesaplanirken tahmin deger ile gercek degerlerin farkinin karesini aliyordu, 
#bu sekilde cezalandirma islemi uyguluyordu. Burda ise kume icinde centroid ile 
#gozlem arasindaki mesafenin karesini alir. Bu yuzden inertia degeri buyur. 
#En az inertia degeri hangi k degerinde cikarsa o k degerini secmek 
#mantikli olur. Cok ic ice girmis datalarda inertia degerinin cok iyi olmasi beklenmez.

#Elbow metoduna gore; keskin dususun durdugu ilk noktadaki k degerini secmek gerekir. 
#Sonraki sert dususler dikkate alinmaz.

plt.plot(K, ssd, "bo-")
plt.xlabel("Different k values")
plt.ylabel("inertia-error") 
plt.title("elbow method") 
Text(0.5, 1.0, 'elbow method')

ssd # sum of squared distance

#Yukaridaki inertia degerleri arasindaki difference' lara bakarsak 
#ilk keskin dusus noktasinin k=3 oldugunu goruyoruz. Elbow metodunun belirledigi 
#k noktasi bu, asagida bir de yellowbrick' in sectigi k noktasina bakip hangisini 
#almamiz gerektigine karar verecegiz :

pd.Series(ssd).diff()   

# diff fonksiyonu sadece series ve DataFrame' lerde kullanilabilir. Bu yuzden degerleri series' e cevirdik. 

#Yukaridaki negatif degerleri pozitife cevirmek icin basina (-) isareti koyduk. 
#Kume sayisini gorebilmek icin rename ile indexi 1' den baslattik :

df_diff =pd.DataFrame(-pd.Series(ssd).diff()).rename(index = lambda x : x+1)
df_diff

#Keskin dususun durdugu noktanin baslangicinin k=3 oldugunu asagida da goruyoruz :

df_diff.plot(kind='bar');

#Asagida yellowbrick icine yeniden modelimizi kurup k icin aralik verdik. 
#Modeli yeniden fit ettik. Yellowbrick k=4 degerini secmis. k=3 icin sert bir 
#dusus var fakat k=4' deki dususu daha anlamli bulmus :

from yellowbrick.cluster import KElbowVisualizer

model_ = KMeans(random_state=42)
visualizer = KElbowVisualizer(model_, k=(2,9))

visualizer.fit(X)        # Fit the data to the visualizer
visualizer.show();

#Iki farkli algoritmadan sonuc olarak iki farkli k degeri geldi. 
#Bazen bu yontemlerle buldugumuz k degerleri bile guvenilir olmayabilir. 
#Datada cok daha fazla kume olabilir, ic ice gecmis datalarda bulunan k degeri 
#yaniltici olabilir. Boyle durumlarda uzman destegi almak gerekir.

####Silhouette analysis
#Silhouette, yapilan kumelemenin kalitesini olcen bir skorlamadir.

#Inertia, her kume icindeki verilerin merkez etrafinda ne kadar siki kumelendiginin 
#skorunu olcuyordu. Silhouette ise kumelerin hem kendi iclerinde ne kadar siki 
#bir sekilde kumelendigini hem de diger kumeye ne kadar uzak oldugunu olcer. 
#Bu yuzden modelimizin kalitesini olcmek icin bu metodu kullanacagiz :

from sklearn.metrics import silhouette_score
#silhouette_score icine once data verilir, sonra da kac label alindiysa bu verilir. 
#Biz yukarida 5 label secmistik (n_clusters=5), silhouette_score icine bunu verdik. 
#Degerimiz 0.488 cikti. Bu deger 1' e ne kadar yakinsa kumeleme o kadar iyi demektir. 
#Bu skorun iyi olup olmadigini diger kume sayilari ile kiyaslayarak karar verecegiz :

#0.48874888709310654
#Silhouette analizinde k icin 2 ile 9 arasinda aralik verip hepsi icin skor 
#almasini istedik. Modeli kurup fit islemini yaptik. Her fit isleminden sonra 
#label' lari aldik. Label sayisina gore de ortalama silhouette_score sonucunu dondurmesini istedik :

range_n_clusters = range(2,9)
for num_clusters in range_n_clusters:
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    # silhouette score
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f"For n_clusters={num_clusters}, the silhouette score is {silhouette_avg}")
#For n_clusters=2, the silhouette score is 0.6810461692117465
#For n_clusters=3, the silhouette score is 0.5528190123564102
#For n_clusters=4, the silhouette score is 0.49805050499728815
#For n_clusters=5, the silhouette score is 0.48874888709310654
#For n_clusters=6, the silhouette score is 0.36483400396700366
#For n_clusters=7, the silhouette score is 0.34974816211612186
#For n_clusters=8, the silhouette score is 0.3574536925852728
#silhouette_score en yukse puani n_clusters=2' de vermis. Fakat yellowbrick 
#sonucuna gore 2' nin cok genis bir alana yayildigini, kendi icinde cok siki 
#kumelenmedigini biliyorduk. Bu yuzden bu degeri dikkate almiyoruz. Sonraki 
#en yuksek skoru n_clusters=3' de almis. Elbow metodun sonucuna guvenip bu degeri seciyoruz.

#Diyelim ki 3 ve 4 degerleri arasinda kararsizlikta kaldik. O zaman basta 
#cizdirdigimiz pairplot' a bakip ordaki bariz kume sayilarina gore bir karar verebiliriz.

#Asagida n_clusters=3 secerek yellowbrick' te visualizer' a baktik. Bu gorsel bize 
#her class' a ait silhouette skorunu verdi. 1 class' inin kendi icinde iyi bir 
#sekilde ic ice girdigini ve diger kumelerden de cok guzel ayrildigini 
#soyleyebiliriz. Ortadaki kirmizi cizgi yukarida n_clusters=3 icin buldugumuz 
#3 kumenin silhouette_score' unun ortalamasidir (%55).

#Gorseldeki renklerin baslangicta kalin olmasi, o kumeye dusen gozlem sayisini 
#gosteriyor. Ne kadar kalinsa o kumeye o kadar fazla gozlem dusmus demektir :

from sklearn.cluster import KMeans

from yellowbrick.cluster import SilhouetteVisualizer

model3 = KMeans(n_clusters=3, random_state=42)
visualizer = SilhouetteVisualizer(model3)

visualizer.fit(X)    # Fit the data to the visualizer
visualizer.poof();

####Building the model based on the optimal number of clusters
#n_clusters=3 degerine karar kildik ve buna gore modelimizi kuracagiz :

model = KMeans(n_clusters =3, random_state=42)
model.fit_predict(X)

model.labels_

#Tahminlerimiz olan model.labels_' i bir degiskene atadik ve bunu asagida X' e feature olarak ekledik :

clusters = model.labels_
X.head()

X["predicted_clusters"] = clusters
X
#Asagida gercek kume degerlerimizi okuttuk. Kurdugumuz model ile bu degerler 
#uyusuyor mu buna bakacagiz. Bu bilgiler elde yok ise datanin uzmanlar 
#tarafindan label' lanmasi istenir :

labels = pd.read_csv("label.csv")
labels

#X datasina label' i feature olarak ekliyoruz :

X["labels"] = labels
X
####CrossTab Function
#crosstab fonksiyonu carpraz dogrulama yapar (Confusion Matrix ile ayni mantikta). 
#Bu fonksiyonun icine ilk olarak tahminlerin oldugu feature' in ismi, 
#ikinci olarak da gercek degerlerin oldugu label verilir (veya uzmanlardan alinan label) :

ct = pd.crosstab(X.predicted_clusters, X.labels)
ct

#0 -------> 48 tanesini versicolor, 14 tanesini virginica olarak tahmin etmis. 
#Demek ki 0 kumesi versicolor.
#1 ------> 50 tanesini setosa turu olarak tahmin etmis. 
#1 class' inin setosa oldugunu anliyoruz, hepsini yakalamis.
#2 ------> 36 tanesini virginica turu olarak tahmin etmis. Demek ki bu da virginica.

#Butun turlerden 50' ser tane vardi. Toplam 16 tane hatali tahmin var. 
#Sadece setosa turunde tam tahmin yapabilmis. Demek ki setosa turu digerlerinden 
#net bir sekilde ayrilmis; versicolor ve virginica turlerinin bilgilerinin ic ice oldugu noktalar var.

####Adjust Rand Score
from sklearn.metrics import adjusted_rand_score
#adjusted_rand_score icine musteriden alinan gercek label ve modelin tahmin 
#ettigi label sirayla verilir :

adjusted_rand_score(X.labels, X.predicted_clusters)
#0.7302382722834697
#Adjust Rand Score' u Accuracy score gibi dusunebiliriz. Fakat Accuracy score 
#boyle bir durumda %90 gibi bir skor verir. Fakat burda %73' luk bir skourumuz var. 
#Cunku Adjust Rand Score cezalandirarak skorlama yapar. 16 hata icin %73' luk bir skor verdi.

#Adjust Rand Score' un alinabilmesi icin konuyla ilgili bir uzmandan gercek 
#label' larin verilmesi gerekir. Aksi taktirde bu skor kullanilamaz.

#####Visualization Clusters
#Bilgilerin ic ice oldugu kisimlari gorebilmek icin datamizi gorsellestirelim. 
#Uzmandan gelen degerler string idi. Fakat gorsellestirmede string deger 
#kullanamayacagimiz icin bunlari sayisal degerlere donusturduk :

X_labels = X.labels.map({"versicolor":0, "setosa":1,"virginica":2})

#sepal_length ve sepal_width' i hem X_labels (uzmandan gelen gercek sonuclar) 
#hem de X.predicted_clusters (yaptigimiz tahminler)' e gore gorsellestirecegiz :

plt.figure(figsize = (20,6))

plt.subplot(121)
plt.scatter(X["sepal_length"], X["sepal_width"], c = X_labels, cmap = "viridis", alpha=0.7)  
# alpha --> Saydamlik. Ust uste binen degerlerin gorunmesi icin.
plt.title("Actual")

plt.subplot(122)
plt.scatter(X["sepal_length"], X["sepal_width"], c = X.predicted_clusters, cmap = "viridis", alpha=0.7)
plt.title("Predicted");

#Actual grafigine baktigimizda ic ice gecmis bir kisim goruyoruz. 
#Modelimiz tahmin yaparken bu kisimlarda hata yapti. Gercekte sari olan verilerin 
#cogunu mor yapmis. Mavi kisimda hic hata yok cunku veriler birbirinden ayri ise 
#k-means algoritmasi bunlari cok iyi ayirir.

#Bir de sepal_width ve petal_length' e gore bir gorsellestirme yaptik. Verilerin 
#ic ice gectigi kisimlarin yine yanlis tahminlerin yapildigi kisimlar oldugunu gorebiliyoruz :

plt.figure(figsize = (20,6))

plt.subplot(121)
plt.scatter(X["sepal_width"], X["petal_length"], c = X_labels, cmap = "viridis", alpha=0.7)
plt.title("Actual")

plt.subplot(122)
plt.scatter(X["sepal_width"], X["petal_length"], c = X.predicted_clusters, cmap = "viridis", alpha=0.7)
plt.title("Predicted");

#clustercenters her feature' a ait her class' in koordonatlarini verir.
#Her sutun bir feature' i temsil eder. Mesela 5.9016129 degeri, 1. feature' in ilk class' inin center' idir :

centers = model.cluster_centers_
centers
#feature' in center degerleri :
centers[:,0] # centers of sepal_length feature
#array([5.9016129, 5.006    , 6.85     ])
#feature' in center degerleri :
centers[:,1] # centers of sepal_width feature
#array([2.7483871 , 3.428     , 3.07368421])
#ve 1. center degerlerini asagida gorsellestirdik. Degerlerden ilk olanlardan biri 
#X eksenine, digeri Y eksenine karsilik geliyor. Mesela X ekseninde 5.9016129 degerine karsilik 2.7483871 degeri geliyor.
plt.scatter(centers[:,0], centers[:,1], c = "black")

#Yukarida buldugumuz bu centroid' leri asagida 0. ve 1. siradaki feature' larin 
#uzerine bindirelim. Center' larin yaptigimiz tahminlere gore orta noktaya 
#yakin bir yerde oldugunu goruyoruz :

plt.scatter(X["sepal_length"], X["sepal_width"], c = X.predicted_clusters, cmap = "viridis", alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

###Remodeling according to discriminating features
#k-means ic ice gecmis datalarda basarisiz kaldi. Ic ice olan datalari birbirinden 
#en iyi ayiran feature' lari tespit edip, ona gore yeniden model kurarsak 
#skorlarimizi artirabiliriz. Turler arasi en iyi ayrimi yapan feature' lar 
#ile daha basarili bir model kurulabilir. Bir nevi feature selection islemi.
X.head()
#Yeniden model kurmadan once gercek label' lari datadan cikaralim :

X.iloc[:, :-1].head()
#Tahminlerimizin oldugu feature olan predicted_clusters' a gore 
#groupby yapip bunlarin ortalamasini aldik. Yani her feature' in yaptigimiz 
#tahminlere gore ortalamalarini aldik. Ortalamalar uzerinden, 
#en fazla tespiti yapacak olan feature' i sececegiz :

clus_pred = X.iloc[:, :-1].groupby("predicted_clusters").mean().T
clus_pred

#Yukarida buldugumuz degerleri lineplot uzerine cizdirdik. 
#sepal_length' te turler cok ic ice olmasa da iyi bir ayrim yok; 
#sepal_width' te cok ic ice gecmisler; petal_length' te ise birbirlerinden çok 
#iyi ayrilmislar; petal_width' te de nispeten guzel bir ayrim var. 
#Son iki feature uzerinden yeni bir k-means algoritmasi kurarak skorlarimizi iyilestirebiliriz :

sns.lineplot(data = clus_pred)

#Asagida son iki feature' in center' larina bakinca cok fazla ic ice olmadiklarini goruyoruz :

plt.scatter(X["petal_length"], X["petal_width"], c = X.predicted_clusters, cmap = "viridis", alpha =0.7)
plt.scatter(centers[:, 2], centers[:, 3], c='black', s=200, alpha=0.5)

#Son iki feature' i sectik, bunlarla yeni bir model kuracagiz :

X2 = X.iloc[:, [2,3]]
X2

###Hopkins test
#Tum feature' lar ile Hopkins test skoru 0.16 idi. Burda ise 0.10 cikmis. 
#Yani kumelenmeye meyillilik artti :

hopkins(X2, X2.shape[0])
#0.10049815035992225
###Elbow metod
ssd = []

K = range(2,10)

for k in K:
    model3 = KMeans(n_clusters =k)
    model3.fit(X2)
    ssd.append(model3.inertia_)
plt.plot(K, ssd, "bo-")
plt.xlabel("Different k values")
plt.ylabel("inertia-error") 
plt.title("elbow method")

#Elbow metoduna gore en keskin dususun durdugu nokta yine 3 degerinde.

df_diff = pd.DataFrame(-pd.Series(ssd).diff()).rename(index = lambda x : x+1)
df_diff

#Yellowbrick' e gore de n_clusters=3 :

from yellowbrick.cluster import KElbowVisualizer

model_ = KMeans(random_state=42)
visualizer = KElbowVisualizer(model_, k=(2,9))

visualizer.fit(X2)        # Fit the data to the visualizer
visualizer.show();

####Silhouette analysis
#n_clusters=3' e gore aldigimiz Silhouette skoru 0.55 idi. Burda ise 0.66' ya yukseldi :

range_n_clusters = range(2,9)
for num_clusters in range_n_clusters:
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X2)
    cluster_labels = kmeans.labels_
    # silhouette score
    silhouette_avg = silhouette_score(X2, cluster_labels)
    print(f"For n_clusters={num_clusters}, the silhouette score is {silhouette_avg}")

#Class' larin kendi icindeki Silhoutte skorlarina bakarsak butun class' larin 
#%65' i gectigini gorebiliriz. Tum class' larda bir iyilesme var :

from sklearn.cluster import KMeans

from yellowbrick.cluster import SilhouetteVisualizer

model3 = KMeans(3, random_state=42)
visualizer = SilhouetteVisualizer(model3)

visualizer.fit(X2)    # Fit the data to the visualizer
visualizer.poof();

#Building the model based on the optimal number of clusters
#Aldigimiz 2 feature ile ve n_clusters=3 ile modelimizi yeniden kurduk, egitimi yaptik :

final_model = KMeans(n_clusters =3, random_state=42)
final_model.fit_predict(X2)

#final_model.labels_
#Musteriden gelen label' lari X2 datamiza ekledik :

X2["labels"] = labels           
#Tahminlerimizi de X2' ye ekledik. Bunlara gore yeniden bir R2 skoru alacagiz :

X2["predicted_clusters"] = final_model.labels_
X2.head()

#Tum feature' lar ile kurdugumuz modelde R2 skoru %73 idi, yeni modelimizde %88 oldu :

adjusted_rand_score(X2.labels, X2.predicted_clusters)
#0.8856970310281228
###Compare results
#crosstab ile carpraz dogrulamamizi yaptik. Hatamiz yukarida 16 idi, bu modelde 6' ya dustu :

# ct for 2 features
pd.crosstab(X2.predicted_clusters, X2.labels)

# ct for all features
ct
###Prediction cluster of new data
##Prediction icin modele bir sample verdik ve modelimiz bu degeri 1 sinifina atadi :

#new_data = [[1.7, 0.2]]
#final_model.predict(new_data)
#array([1])
 

#%%
##################################################################################
#LESSON 18 
##################################################################################

####Hierarchical Clustering Theory(Büyük Datalarda Maliyeti Yüksek)
#K-Means Algoritması gibi clustering için kullanılan bir algoritmadır.Önce bir 
#dendogram yapar,bu dendogram üzerinden ne kadar cluster yapılacağını ortaya koyar.
#Datanın ne kadar cluster'a ayrılacağı önceden belirlenmez.Dendogram yapılır,buna
#karar verilir

#Dendogram:Similarity ne kadar azsa observationlar birbirine o kadar yakındır.
#Kümeler arasındaki uzaklığın ne kadar olduğunu dendogram üzerinde görerek kaç cluster
#olması gerektiğine karar verilir

#Dendogram iki şekilde oluşturulur.(Agglomerative(En çok kullanılan),Divisive)

##Agglomerative:Aşağıdan yukarıya doğru çalışır.Önce bütün observationları birer
#cluster olarak kabul eder;yukarı doğru çıkarak en son datanın hepsini bir cluster 
#olarak kabul eder 

##Divise:Yukarıdan aşağıya doğru çalışır.Tüm datayı alır,aşağı doğru ikiye ayırarak
#gider.K-means algoritmasını kullanır.Çok kullanışlı değildir.Bunun yerine 
#k-mean algoritması kullanmak daha mantıklı.

#K-means Algoritmasında olduğu gibi Hierarchical Clustering de amaç:
    #Cluster içindeki mesafe min olsun.
    #Clusterlar arasındaki mesafe max olsun 
# x
#7|         p6
#6| p2    p4 p5          x:Amount of money 
#5|                      y:Money mobility
#4|   p1             
#3|
#2| p0     p3
#1|____________y
# 1 2 3 4 5 6 7

#Dendogram,her bir data noktasını bir cluster olarak alır.Sonra herbirinin birbirlerine
#olan mesafelerini hesaplar.En yakın iki tanesinin orta noktasını bir cluster olarak alır.
#(P5 ile P6).Bu iki nokta yerine tek bir cluster alır.Yeniden ölçüm yapar.En yakındaki
#P4ü alır ve orta noktayı bularak birleştirir.Sonra da P3 ü alır.
#P1 ve P2 birbirine en yakın iki nokta olduğu için bunları birleştirip ayrı bir 
#cluster yapar.Sonra bunlara Po ı katar.En son aşamada kalan son 2 clusterı birleştirir 

#Clusterlar arası mesafe yukarıdaki gibi belirlenir.Mesafe hangisinde daha büyük ise 
#araki clusterlar birbirinden daha uzaktır.

#                 _________________________________
#                |                                 |
#           _____|_____                       _____|_____
#          |           |                     |           |
#      ____|____    ___|___             _____|_____   ___|___
#     |         |  |       |           |           | |       |
#     |         |  |       |           |           | |       |
#
#Clusterlar arası esafe yukarıdaki gibi belirlenir.Mesafe hangisinde daha büyük 
#ise oradaki clusterlar birbirinden daha uzaktır.
#Hangi iki çizgi arası mesafe en uzaksa,o iki çizgi arasından hayali bir çizgi 
#çizilir ve o çizgi üzerine kesen dikey çizgiler sayılır.Bu sayı bize clustır 
#sayısını verir.
#Bu bilgi dayadan elde ettiğimiz bilgi kesin doğruluğu yoktur.Bir uzman bizim 
#bulduğumuzdan farklı bir sayı söyleyebilir.O zaman o kabul edilir.

#Hierarchial Clustering in k-mean den farkı;hiçbir işlem yapadan dendograma bakarak 
#k sayısını tahmin edebiliyor olamızdır 

#Hyperparameters:
#sklearn.cluster.AgglomerativeClustering

#1-affinity(Default='euclidean'):
#Distance parametresidir
#{'euclidean','manhattan','cosine','precomputed'}
#!!!! 'euclidean' sadece 'ward' ile kullanilir

#2-Linkage(Default='ward')
#Datanin hangi clust a gidecegini belirler(Clusterlar arasi mesafe olcer)
#{'ward','complete','average','single'}
#average: data noktasini hang' clusterin meanine daha yakinsa ona atar
#complete:Clusterlarin en uzak noktasini bulur.Bunlar icin en yakin olana atama yapar
#Single:En yakin nokta hangisi ise ona atar
#!!!!!Complete:Noise datalarda iyi calisir fakat buyuk clusterlar olur
#!!!!!Single grift datalarda kotu calisir(Complete in aksine)
#ward:Kumelerin icinin varyansinin en az olmasini saglayacak sekilde calisir
#Datayi hangisine eklendiginde varyans az olacaksa ona ekler 


#%%
#####NOTEBOOK 18-1#############

#Hierarchical Clustering
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
#%matplotlib notebook
plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")
#pd.set_option('display.float_format', lambda x: '%.3f' % x)
#k-means algoritmasinda kullandigimiz iris datasetini Hierarchial Clustering' de de kullanacagiz :

df = pd.read_csv("iris1.csv")
df.head()

df.info()

df.describe()

#Gorselde 2 kumenin birbirinden net bir sekilde ayrildigini goruyoruz. 
#Daha fazla kume sayisi varsa pairplot ile bunun tespiti zor fakat bu kumeden 
#birinin net bir sekilde ayrildigini ve kume ici varyansin dusuk oldugunu; 
#digerinin ise genis bir alana yayildigini ve varyansin yuksek oldugunu goruyoruz. 
#Varyansi yuksek olan kumeden baska bir kume daha cikabilir :

sns.pairplot(df);

####Scaling Data

#The K-means algorithm definitely needs scaling. However, if all our features are 
#in the same/similar range, there is no need to scale the data. For these data, 
#scaling may cause worse results in some cases. You should try data both with and 
#without scale and continue with whichever one has good results.

#K-means algoritmasinda oldugu gibi scale islemi yapmayacagiz. Cunku data zaten 
#scale edilmis gibi. Yine de scale islemi yapilip hangisinin sonuclari daha iyi gorulebilir.

######Hopkins Test
#Hopkins test, a statistical test for randomness of a variable.
#Null Hypothesis (Ho) : Data points are generated by non-random, uniform distribution (implying no meaningful clusters)
#Alternate Hypothesis (Ha): Data points are generated by random data points (presence of clusters)
#The more close 0, the more available to separate clusters
#!pip install pyclustertend
from pyclustertend import hopkins
X = df.copy()
#Hopkins test icinde ilk kisma datanin kendisi, ikinci kisma sampling_size verilir. 
#Cok buyuk datalarda Hopkins test yavas calisacagi icin icine belli bir gozlem sayisi verilebilir.

hopkins(X, X.shape[0])   
#0.16847390567991735
#Hopkins test skoru 0 ile 0.5 arasinda ise clusturing icin uygundur, random bir 
#dagilim var demektir. 0.5' ten sonra uniform dagilima dogru gider ve clustering zorlasir.

#####Dendrogram
#K-means Algoritmasi gibi clustering icin kullanilan bir algoritmadir. 
#Once bir dendogram cizer, bu dendogramdaki cluster' lar arasi mesafeye gore 
#datanin kac cluster' a ayrilmasi gerektigine karar verir.

#Tree-like hierarchical representation of clusters is called a dendrogram.
#It illustrates the arrangement of the clusters produced by the corresponding analyses.
from scipy.cluster.hierarchy import dendrogram, linkage
#dendrogram()
#“linkage” parameter: (default= “ward”)
#{‘ward’, ‘complete’, ‘average’, ‘single’}

#Which linkage criterion to use. The linkage criterion determines which distance 
#to use between sets of observation.

#Ward minimizes the variance of the clusters being merged.
#Average uses the average of the distances of each observation of the two sets.
#Complete or maximum linkage uses the maximum distances between all observations of the two sets.
#Single uses the minimum of the distances between all observations of the two sets.

######PARAMETERS
#Z -----> Dendogram icine datanin kendisi ile birlikte hangi metodu kullanacigini da ister. 
#Cluster secimini hangi metod ile yapacagini belirtmemiz gerekir. Datayi dendograma 
#metoduyla birlikte vermek icin de linkage fonksiyonu kullanilir.

#*LINKAGE -----> Datanin hangi cluster' a gidecegini belirler. 
#(Cluster' lar arasi veya ici varyansi olcen parametreleri var.)

#Linkage icine sirasiyla; data, method (default='single'), metric (default='euclidean') parametrelerini alir.

#ward --> Kumelerin varyansinin en az olmasini saglayacak sekilde calisir. 
#Datayi hangi cluster' a eklediginde varyans en az olacaksa datayi ona ekler. 
#Bir data iki kumeye de ayni uzakliktaysa, varyansi zaten dusuk olan bir kumenin 
#kalitesini bozmak istemez, bu datayi varyansi yuksek olan kumeye atmayi tercih eder. 
#Hedefi varyansi dusuk tutmaktir. (Linkage icin default deger 'ward')

#complete --> Cluster' larin en uzak noktasini bulur. Bu en uzak noktalardan en 
#yakini hangisi ise datayi o cluster' a atar.

#average --> Datanin her bir cluster elemanina olan uzakliklarini olcer ve ortalama alir. 
#Bu ortalama deger hangi cluster' da en kucukse datayi ona atar.

#single --> Datanin cluster' larin en yakin elemanina olan uzakliklarini olcer; 
#en yakin olan cluster' a atama islemini yapar.

#*p --> min su kadar kumeyi goster. (Dendogramdaki goruntu karisikligini onlemek icin) (Default=30)

#*truncate_mode --> Eger bir p degeri giriyorsak bunu da default deger yerine 
#'lastp' olarak degistirmemiz gerekir. (Sondaki belirttigim kadar sayida p' yi goster) (Default=None)

#*AFFINITY --> Gozlemlerin birbirine olan uzakligini olcen parametre. (Default='euclidean')

#!!!! 'euclidean' sadece 'ward' ile kullanilir. !!!!

hc_ward = linkage(y = X, method = "ward")  # Datayi hangi kumeye atarsan varyans en en dusuk olacaksa o kumeye at.
hc_complete = linkage(X, "complete")       # Kumelerdeki en uzak noktalari bul, bunlar icinde en yakin olana atama yap.
hc_average = linkage(X, "average")         # Datanin kumenin her elemanina olan uzakliklarinin ortalamasini al, en az olan kumeye ata ata.
hc_single = linkage(X, "single")           # Kume elemanlarinin dataya en yakin noktasini olc, dataya en yakin olana atama yap.
#hc_ward --> Yukarida varyansa gore olcen parametreyi sectik. Buna gore cluster islemi asagida yapildi. y ekseni bize cluster' lar arasi mesafeyi veriyor. y eksenine gore mesafesi en uzak olan cluster' lara gore k sayisi belirleyecegiz. Asagida en uzak mesafelere gore k=2 olur. Dendogram karar verdigi cizgiyi mavi ile cizer. Burada, en iyi kumelenmenin 2 class ile olacagina karar vermis.

#K-means algoritmasinda oldugu gibi kumelerin kendi icinde ne kadar yogun olarak 
#kumelendigini olcen bir inertia degeri bu algoritmada yok. Burada kumeler 
#arasi mesafeye gore bir karar verilir. Bu yuzden dendogram ile alinan kumeleme kalitesi k-mean' den daha dusuk olabilir.

plt.figure(figsize = (20,8))
plt.title("Dendrogram")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_ward, leaf_font_size = 10);   

#Yukaridaki gorselde dendogram bize 2 cluster olmasi gerektigini soyledi fakat 
#biz 3 degerini secersek de en alttaki 3 kumenin de birbirinden guzel ayristigini 
#gorebiliyoruz.'3 degerini secersek de kaliteli bir clustering olabilir.' cikarimini 
#da gorsele bakarak yapabiliriz.

#Yukarida linkage icindeki tum parametrelere gore degiskenler belirlemistik. 
#Bunlarin hepsine gore dendogramlarimizi cizdirelim. Butun parametreler bize 2 
#cluster olmasi gerektigini soyluyor. y eksenlerine baktigimzda distance en 
#fazla 'ward' da; en dusuk ise single' da. Default deger olan ward' in digerlerine 
#gore cok daha kaliteli bir kumeleme yaptigini goruyoruz :

plt.figure(figsize = (20,12))

plt.subplot(221)
plt.title("Ward")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_ward, leaf_font_size = 10)

plt.subplot(222)
plt.title("Complete")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_complete, leaf_font_size = 10)

plt.subplot(223)
plt.title("Average")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_average, leaf_font_size = 10)

plt.subplot(224)
plt.title("Single")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_single, leaf_font_size = 10);

#Yukarida feaute sayisinin cok olmasindan dolayi karisik bir goruntu olustu. 
#Cok daha buyuk datalarda bu goruntu karisikliga sebep olacaktir. Bu yuzden 
#asagida bazi hyperparametreler ile oynayacagiz.

#p=10 yaparak yukaridaki son 10 dallanmayi goster diyerek bir kirpma islemi yapmis 
#olduk. Boylece gorselimiz daha sade bir hal aldi. Asagidaki tum dendogramlarda 
#kalan observation sayisinin 10 oldugunu gorebiliriz. :

plt.figure(figsize = (20,12))

plt.subplot(221)
plt.title("Ward")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_ward, leaf_font_size = 10, truncate_mode='lastp', p=10)   # leaf_font_size --> X eksenindeki yazilari buyutmek icin.

plt.subplot(222)
plt.title("Complete")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_complete, leaf_font_size = 10, truncate_mode='lastp', p=10)

plt.subplot(223)
plt.title("Average")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_average, leaf_font_size = 10, truncate_mode='lastp', p=10)

plt.subplot(224)
plt.title("Single")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_single, leaf_font_size = 10, truncate_mode='lastp', p=10);

#####Hierarchical Clustering (Agglomerative)
#Tumevarim mantigi ile islem yapar. En alttaki observation' larin hepsini birer 
#cluster olarak kabul eder ve bunlari en yakinlarindaki gozlem ile birlestirerek 
#yeni bir orta nokta belirler. Bu sekilde yukariya kadar datalari birlestirir 
#ve en son datanin tamamina ulasir.

#Ideal clustering is characterised by minimal intra cluster distance and maximal inter cluster distance
from sklearn.cluster import AgglomerativeClustering
#S(Silhouette) Score
#If the ground truth labels are not known, evaluation must be performed using the model itself. 
#(One of the evaluation method is Silhouette Coefficient)
#A higher Silhouette Coefficient score relates to a model with better defined clusters.
#a :The mean distance between a sample and all other points in the same class. 
#b: The mean distance between a sample and all other points in the next nearest cluster.
#s = (b-a) / max(a,b)

#Bu algoritmada kumelenme kalitesini olcen bir inertia degeri olmadigi icin kume 
#icindeki datalarin birbirleri ile ne kadar siki bir iliski icinde olduklarini 
#olcemiyoruz. Bu yuzden Silhouette Score' dan destek alacagiz :

from sklearn.metrics import silhouette_score
K = range(2,10)

for k in K:
    model = AgglomerativeClustering(n_clusters = k)
    model.fit_predict(X)
    print(f'Silhouette Score for {k} clusters: {silhouette_score(X, model.labels_)}')
#Silhouette Score for 2 clusters: 0.6867350732769781
#Silhouette Score for 3 clusters: 0.5543236611296426
#Silhouette Score for 4 clusters: 0.48896708575546993
#Silhouette Score for 5 clusters: 0.48438258927906036
#Silhouette Score for 6 clusters: 0.359237619260309
#Silhouette Score for 7 clusters: 0.34220719366205077
#Silhouette Score for 8 clusters: 0.3435906599119544
#Silhouette Score for 9 clusters: 0.3304886352874667
#Silhouette Score' u bize bariz bir sekilde datayi 2 cluster' a bolmemiz gerektigini soyluyor. 
#Eger bir uzman destegimiz yoksa bu skorlara gore sececegimiz deger 2 olmali. 
#K-means algoritmasi da cluster degerini 2 olarak bulmustu fakat orada 2 cluster' da 
#da inertia degerinin (kume ici elemanlarin merkeze olan uzakligi) cok yuksek oldugu 
#bilgisini bize vermisti. Burada bu bilgiyi saglayacak bir skor olmadigi icin hem 
#dendogram hem de Silhouette Score sonuclarina guvenerek cluster=2 olarak kabul ediyoruz. 
#Simdiye kadarki gozlemlerimizde cluster sayisini 3 secersek de kaliteli sonuclar 
#alma ihtimalimiz oldugunu gozlemledik.

#Uzmandan destek aldigimizi ve bu data icin 3 cluster oldugu bilgisini aldigimizi 
#farzederek cluster=3 olarak kabul edip modelimizi kuruyoruz. affinity ve linkage 
#parametrelerini degistirmek cok fazla onerilmez fakat datada cok fazla outlier var 
#ise complete denenebilir :

model1 = AgglomerativeClustering(n_clusters=3, affinity = "euclidean", linkage = "ward")
model1.fit_predict(X)
#K-means algoritmasinda fit predict' ten ayri kullanilabiliyordu. 
#Fakat burada ayri ayri kullanilmaz, hata verir.

#sklearn kaynaklarinda fit_predict islemi; "cluster' larin merkez noktalarinin 
#hesaplanmasi" olarak gecer. Bu yuzden unsupervised algoritmalarinda ikisinin 
#birlikte kullanilmalari tavsiye edilir. Hierarchical Clustering' de ise istense 
#bile ayri ayri kullanilamazlar.

model1.labels_     # Yukaridaki sonuclarin aynisini dondurur. 

#Asagida 2 feature' i gorsellestirdik. Agglomerative 'in diger bir negatif yonu; 
#K-means algoritmasindaki gibi cluster merkezlerini belirleyemiyoruz :

plt.scatter(X["petal_length"], X["petal_width"], c = model1.labels_, cmap = "viridis", alpha =0.7)

#Olusturdugumuz 3 cluster' a gore tahminlerimizi almistik (model1.labels_).
#Bunlari bir degiskene atadik. Uzmandan aldigimiz labelleri da bir degiskene atadik. 
#Asagida bu iki bilgiyi feature olarak datamizin sonuna ekledik :

clusters = model1.labels_
labels = pd.read_csv("label.csv")
labels


X["predicted_clusters"] = clusters
X["labels"] = labels
X


#Crosstab islemi ile kendi degerlerimiz ile uzmandan gelen degerleri kiyasladik. 
#Aldigimiz sonuclar K-means algoritmasindan aldigimiz sonuclar ile hemen hemen
#ayni cikti. Orada da 16 hata yapmistik, burada da 16 hata yaptik. 
#Kumeleme kalitesinde herhangi bir sikinti yok. Sadece hatalarin yerleri degismis. 
#Fakat bu algoritmada karsilastirma metrikleri az oldugu icin uzman birinden 
#bilgi almadigimiz surece imkanlarimiz kisitlidir.

ct = pd.crosstab(X["predicted_clusters"], X["labels"])
ct
#Model burada 0 class' i icin fazla tahminleme, 2 class' i icin de eksik tahminleme yapmis.

########ARI Score
#The Adjusted Rand Index computes a similarity measure between two clusterings by 
#considering all pairs of samples and counting pairs that are assigned in the same 
#or different clusters in the predicted and true clusterings.
#The value of ARI indicates no good clustering if it is close to zero or negative, 
#and a good cluster if it is close to 1.
from sklearn.metrics.cluster import adjusted_rand_score
#R2 skoru da K-means' de aldigimiz skora cok yakin cikti. (R2 skoru Accuracy skoru 
#gibidir fakat yapilan hatalar uzerinden cezalandirma yapmasi Accuracy score' a gore 
#daha dusuk bir skor almamiza sebep olur) :

adjusted_rand_score(X.labels, X.predicted_clusters)
#0.7311985567707746
######Visualization Clusters
#Visualization islemi icin str olan class isimlerimizi numeric hale cevirdik :

X_labels = X.labels.map({"versicolor":0, "setosa":1,"virginica":2})
X_labels

#c = X_labels ---> Renklendirmeyi kac farkli kategoriye gore yapayim?

#Hem gercek degerler olan X_labels hem de tahmin degerlerimiz olan X.predicted_clusters' a 
#gore 2 feature' a ait gorsellestirme islemini yaptik :

plt.figure(figsize = (20,6))

plt.subplot(121)
plt.scatter(X["sepal_length"], X["sepal_width"], c = X_labels, cmap = "viridis", alpha=0.7, s=100)
plt.title("Actual")

plt.subplot(122)
plt.scatter(X["sepal_length"], X["sepal_width"], c = X.predicted_clusters, cmap = "viridis", alpha=0.7, s=100)
plt.title("Predicted");

#Actual kisimda mor ve sari datalarin ic ice gectigini, bu datalarin predicted 
#kisminda yanlis class' a atandigini goruyoruz. K-means' de oldugu gibi grift 
#kisimlari ayirmada Agglomerative de yetersiz kaldi.

#Baska 2 feature secerek gercek ve tahmin degerlere baktik. Yine grift olan kisimlar 
#oldugunu ve algoritmanin bunlari ayirmada yetersiz kaldigini goruyoruz :

plt.figure(figsize = (20,6))

plt.subplot(121)
plt.scatter(X["sepal_width"], X["petal_length"], c = X_labels, cmap = "viridis", alpha=0.7, s=100)
plt.title("Actual")

plt.subplot(122)
plt.scatter(X["sepal_width"], X["petal_length"], c = X.predicted_clusters, cmap = "viridis", alpha=0.7, s=100)
plt.title("Predicted");

#Remodeling according to discriminating features
X.head()

#Elimizde uzmandan gelen gercek degerler oldugu icin bunun uzerinden gruplandirma 
#yaparak ortalama degerlerimize baktik. Elimizde cogu zaman musteriden gelen veriler 
#olmayacagi icin, oyle durumlarda degerlerimize predicted_clusters uzerinden 
#bakmamiz gerekir. K-means algoritmasinda predicted label' i kullanmistik.

#Mean degerlere gore en iyi ayrimin petal_length ve petal_width' te oldugunu goruyoruz :

clus_pred = X.iloc[:, [0, 1, 2, 3, 5]].groupby("labels").mean().T
clus_pred

#Yukarida elde ettigimiz degerleri asagida gorsele aktardir ve sonuc olarak 
#K-means' de aldigimiz degerlere yakin degerler elde ettik. En iyi ayrisimi yapan 
#feature' larin petal_length ve petal_width oldugunu goruyoruz :
sns.lineplot(data = clus_pred)

#Asagida boxplot' a gore butun feature' larin tek tek labels ile olan iliskisine baktik 
#ki birbirine grift durumda olan degerler var mi :

plt.figure(figsize = (20,6))

plt.subplot(141)
sns.boxplot(y = "sepal_length", x = "labels", data = X)

plt.subplot(142)
sns.boxplot(y = "sepal_width", x = "labels", data = X)

plt.subplot(143)
sns.boxplot(y = "petal_length", x = "labels", data = X)

plt.subplot(144)
sns.boxplot(y = "petal_width", x = "labels", data = X)

#sepal_length' te class' lara gore iyi bir ayrim olmus. Whiskers' larda az cok 
#grift bir durum var fakat datalarin yogun toplandigi bolgelerde cok fazla grift bir durum yok.

#sepal_width' te datalarin yogun oldugu bolgelerde verscolor ve virginica 
#turlerinin birbirine grift oldugunu soyleyebiliriz.

#petal_length ve petal_width' te datalarin yogun oldugu bolgelerede class' lara 
#gore cok iyi bir ayrisim olmus. Bu iki feature kullanilarak bir model kurmamiz 
#gerektigini anladik. Dendogram' da inertia gibi yardimci degerler olmasa da boxplot' a b
#akarak boyle bir inside saglayabiliriz.

#Sadece petal_length ve petal_width feature' larini alarak yeniden modelimizi kuracagiz.

X2 = X.iloc[:, [2,3]]
X2


######Hopkins Test
#2 feature secerek aldigimiz Hopkins test skoru dustu. Iyi bir clustering islemi yapabilecegimizi anladik :
hopkins(X2, X2.shape[0])
#0.10589104835654066
#####Dendrogram
#Tekrar butun metodlar icin degiskenlerimizi tanimladik. Bu sefer data olarak X2' yi kullaniyoruz :

hc_ward = linkage(y = X2, method = "ward")
hc_complete = linkage(X2, "complete")
hc_average = linkage(X2, "average")
hc_single = linkage(X2, "single")
#Dendogram cluster sayisi olarak hala 2' yi secmemiz gerektigini soyluyor. Asagida bir de Silhouette Score' a bakacagiz :

plt.figure(figsize = (20,12))

plt.subplot(221)
plt.title("Ward")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_ward, leaf_font_size = 10, truncate_mode='lastp', p=10)

plt.subplot(222)
plt.title("Complete")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_complete, leaf_font_size = 10, truncate_mode='lastp', p=10)

plt.subplot(223)
plt.title("Average")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_average, leaf_font_size = 10, truncate_mode='lastp', p=10)

plt.subplot(224)
plt.title("Single")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_single, leaf_font_size = 10, truncate_mode='lastp', p=10);

########S(Silhouette) Score
#Silhouette Score' da en yuksek skorun cluster=2' nin oldugunu soyluyor fakat 
#yukaridaki skorlara bakarsak cluster=3' un skorlarinin cok daha fazla artis 
#gosterdigini goruyoruz. Yine de burada yaptigimiz yorumlar kesin yorumlar degil, bir uzman destegine ihtiyac var :

K = range(2,10)

for k in K:
    model = AgglomerativeClustering(n_clusters = k)
    model.fit_predict(X2)
    print(f'Silhouette Score for {k} clusters: {silhouette_score(X2, model.labels_)}')
#Silhouette Score for 2 clusters: 0.7669465622770762
#Silhouette Score for 3 clusters: 0.6573949269287823
#Silhouette Score for 4 clusters: 0.5895284480910935
#Silhouette Score for 5 clusters: 0.5781917218437669
#Silhouette Score for 6 clusters: 0.5747380906148477
#Silhouette Score for 7 clusters: 0.5830818097709548
#Silhouette Score for 8 clusters: 0.5678904784921739
#Silhouette Score for 9 clusters: 0.5469910001848306

##########Final model
#Cluster=3 secerek ve en iyi parametrelerimizi tanimlayarak X2 datasina gore 
#modelimizi tekrar olusturduk. Musteriden gelen labellari da ekledik :

final_model = AgglomerativeClustering(n_clusters=3, affinity = "euclidean", linkage = "complete")
clusters = final_model.fit_predict(X2)
X2["predicted_clusters"] = clusters
X2["labels"] = labels
X2

#Musteriden gelen labellara gore Crostab islemi yaptigimizda hatanin 6' ya dustugunu goruyoruz :

X2_ct = pd.crosstab(X2["predicted_clusters"], X2["labels"])
X2_ct

ct  # Daha once butun feature' lar ile aldigimiz crosstab skorlari.

#Gorsellestirme icin str olan class isimlerimizi tekrar numeric hale cevirdik :

X2_labels = X2.labels.map({"versicolor":0, "setosa":1,"virginica":2})
X2_labels


plt.subplot(121)
plt.scatter(X2["petal_length"], X2["petal_width"], c = X2_labels, cmap = "viridis", alpha=0.7, s=100)
plt.title("Actual")

plt.subplot(122)
plt.scatter(X2["petal_length"], X2["petal_width"], c = X2.predicted_clusters, cmap = "viridis", alpha=0.7, s=100)
plt.title("Predicted");

#Actual ve predicted grafiklerine baktigimizda, grift olan kisimlarda hala hatalar yapildigini goruyoruz.

######ARI Score
#2 feature ile R2 skorumuzun da arttigini goruyoruz :

adjusted_rand_score(X2.labels, X2.predicted_clusters)
#0.8857921001989628
###Prediction cluster of new data
#Olusturdugumuz model icin iki tane gozlem iceren bir data olusturduk.
#Fakat bu datayi modele verdigimizde asagidaki gibi hata aliyoruz. Agglomerative 
#Clustering algoritmasinin predict ozelligi yoktur. Prediction yapilamamasi bu modelin 
#en kotu ozelliklerinden biridir. Modele tahmin yaptirabilmenin yolu; dataya bu 
#gozlemleri eklemek ve modeli yeniden olusturarak bu datalarin hangi class' a ait 
#olduklarini o sekilde gormek olabilir. Bu da zor bir islem oldugu icin genelde 
#K-means Algoritmasi tercih edilir.

new_data = [[1.7, 0.2], [2.3, 0.5]]
final_model.predict(new_data)
#---------------------------------------------------------------------------
#AttributeError                            Traceback (most recent call last)
#~\AppData\Local\Temp/ipykernel_8308/3499392862.py in <module>
#----> 1 final_model.predict(new_data)
#AttributeError: 'AgglomerativeClustering' object has no attribute 'predict'
 

#%%
#####NOTEBOOK 18-2#############

#######Hierarchical Clustering
#This data set contains statistics, in arrests per 100,000 residents for assault, 
#murder, and rape in each of the 50 US states in 1973. Also given is the 
#percent of the population living in urban areas.
#A data frame with 50 observations on 4 variables.
#Murder numeric Murder arrests (per 100,000)
#Assault numeric Assault arrests (per 100,000)
#UrbanPop numeric Percent urban population
#Rape numeric Rape arrests (per 100,000)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#%matplotlib notebook
plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")
#pd.set_option('display.float_format', lambda x: '%.3f' % x)
#Basari, harcama, suc oranlari gibi icerigi olan datalarda gruplandirma yaparken 
#musteri cok fazla yol gosterici olmaz, bizim yorumlarimiz daha cok anlam kazanir. 
#Bu gruplandirma islemleri, clustering algoritmalarinin en cok kullanildigi alanlardir.

#Mesela basarinin olculdugu bir datada basarili, basarisiz, orta seviyede basarili
#gibi 3 cluster islemi yapildiginda Silhouette skoru cok kotu ise daha fazla 
#cluster islemi yapilip cok basarili, basarili gibi yeni cluster' lar eklenebilir.

#Elimizdeki data, 1973 yilinda Amerika' daki 50 eyaletin her biri icin gerceklesen 
#tutuklamanin ne kadarinin cinayet, ne kadarinin fiziki saldiri, ne kadarinin 
#Tecavuz suclamalarindan gerceklestigini veren bir data.

#UrbanPop --> Bir eyaletin ilcesindeki nufusun genel eyaletin nufusuna olan orani.
#Datadaki eyaletler isimleri de bir feature' di. index_col=0 islemi ile bu feature' i index olarak atamis olduk :

df = pd.read_csv("USArrest.csv", index_col=0)
df.head()

#Exploratory Data Analysis and Visualization
df.info()

#Describe' a bakarsak datamizin bir scale islemine ihtiyaci oldugunu goruyoruz :

#df.describe().T

#Asagidaki dagilim bize cok net bir inside saglamiyor. Belki Murder ve Assault arasinda 
#2 class' lik bir ayrim yapilacagi soylenebilir. Pairplot bize cok fazla bir inside 
#saglamadigi icin asagida bir de barplot grafiklerine bakacagiz :
sns.pairplot(df);

#Asagida Murder, Assault ve Rape'e gore eyalet isimlerini yazdirdik. Siralamayi 
#da bu feature' lara gore yapmasini istedik. Bunlara bakarak; sucun cok oldugu, 
#az oldugu, orta derecede oldugu eyaletler gibi bir ayrim yapabiliriz. 
#Sadece bu gorsellere bakarak bile datayi 3 cluster' a bolmemiz gerektigi bilgisini 
#aldik. Bunu metriklerimiz ile teyit edecegiz :

plt.figure(figsize = (14,6))
sns.barplot(y = "Murder", x = df.index, data = df, order = df.Murder.values.sort())
plt.xticks(rotation = 90);

plt.figure(figsize = (14,6))
sns.barplot(y = "Assault", x = df.index, data = df, order = df.Assault.values.sort())
plt.xticks(rotation = 90);

plt.figure(figsize = (14,6))
sns.barplot(y = "Rape", x = df.index, data = df, order = df.Rape.values.sort())
plt.xticks(rotation = 90);

#####Hopkins Test
#Hopkins test, a statistical test for randomness of a variable.
#Null Hypothesis (Ho) : Data points are generated by non-random, uniform distribution (implying no meaningful clusters)
#Alternate Hypothesis (Ha): Data points are generated by random data points (presence of clusters)
#The more close 0, the more available to separate clusters
#!pip install pyclustertend
from pyclustertend import hopkins
#Hopkins test skoru 0.5' in altinda cikti, iyi bir kumeleme yapabilmemiz olasi gorunuyor :
hopkins(df, df.shape[0])
#0.3166210893583002
from sklearn.preprocessing import StandardScaler, MinMaxScaler
Hem MinmaxScaleer hem de StandardScaler islemi ile skorlara bakmak istedigimiz icin df' in iki tane kopyesini aldik :

df1 = df.copy()
df2 = df.copy()
hopkins(MinMaxScaler().fit_transform(df1), df1.shape[0])
0.20693446968811494
hopkins(StandardScaler().fit_transform(df2), df2.shape[0])
0.2161343187125487
#Iki scaler isleminde de asagi yukari ayni sonuclar dondu. Biz burada MinMaxScaler' 
#i tercih ettik ve buna gore egitimimizi yaptik :

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
X = pd.DataFrame(df_scaled, columns=df.columns)

###Dendrogram
#Tree-like hierarchical representation of clusters is called a dendrogram.
#It illustrates the arrangement of the clusters produced by the corresponding analyses.
from scipy.cluster.hierarchy import dendrogram, linkage
#dendrogram()
#“linkage” parameter: (default= “ward”)
#{‘ward’, ‘complete’, ‘average’, ‘single’}
#Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation.
#Ward minimizes the variance of the clusters being merged.
#Average uses the average of the distances of each observation of the two sets.
#Complete or maximum linkage uses the maximum distances between all observations of the two sets.
#Single uses the minimum of the distances between all observations of the two sets.
#Linkage degiskenlerimizi tanimladik. Bunlarin hepsine gore birer dendogram cizdirip en iyi olan skoru sececegiz :

hc_ward = linkage(y = X, method = "ward")
hc_complete = linkage(X, "complete")
hc_average = linkage(X, "average")
hc_single = linkage(X, "single")
plt.figure(figsize = (14,7))
dendrogram(hc_ward, leaf_font_size = 10);   

#Ward, cluster=2' yi tavsiye etti. Fakat yukaridaki gorsellerden 3 cluster yapabilecegimizi 
#ongormustuk ve boyle bir gurplandirma datasinda musteriye bagimli degiliz, daha ozgur secimler yapabiliriz.

#Yukaridaki koda show_contracted = True parametresini ilave ettik. Bu parametre 
#cluster' larin sonuna centikler atar ki bu centik sayisi bize aslinda asagida kac tane daha cluster oldugunu gosterir :

plt.figure(figsize = (14,7))
dendrogram(hc_ward,
           truncate_mode = "lastp",
           p = 10,
           show_contracted = True,
           leaf_font_size = 10);

#Yukarida tanimladigimiz butun degiskenlere gore asagida dendogramlari cizdirdik :

plt.figure(figsize = (20,12))

plt.subplot(221)
plt.title("Ward")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_ward, truncate_mode = "lastp", p = 10, show_contracted = True, leaf_font_size = 10)

plt.subplot(222)
plt.title("Complete")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_complete, truncate_mode = "lastp", p = 10, show_contracted = True, leaf_font_size = 10)

plt.subplot(223)
plt.title("Average")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_average, truncate_mode = "lastp", p = 10, show_contracted = True, leaf_font_size = 10)

plt.subplot(224)
plt.title("Single")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_single, truncate_mode = "lastp", p = 10, show_contracted = True, leaf_font_size = 10);

#Complete ve Average dendogramindaki mavi cizgiler, cluster=3 almamiz gerektini 
#soyluyor. Bunlar gorselde aldigimiz bilgiyi desteklediler. Bunlari da dikkate 
#alarak cluster=3 sececegiz. Bunlardan net bir bilgi alamasak bile 3 cluster 
#secebilirdik. Cunku yukaridaki barplot grafikleri bize bu bilgiyi sagladi.

######Hierarchical Clustering (Agglomerative)
#Ideal clustering is characterised by minimal intra cluster distance and maximal inter cluster distanc
from sklearn.cluster import AgglomerativeClustering
#S(Silhouette) Score
#If the ground truth labels are not known, evaluation must be performed using the model itself. (One of the evaluation method is Silhouette Coefficient)
#A higher Silhouette Coefficient score relates to a model with better defined clusters.
#a :The mean distance between a sample and all other points in the same class. b: The mean distance between a sample and all other points in the next nearest cluster.
#s = (b-a) / max(a,b)

Yukaridaki bilgilere bakarak cluster=3 secmeye karar vermis olsak bile Silhouette skoruna mutlaka bakmaliyiz.

from sklearn.metrics import silhouette_score
#Gorsellerden cluster=3 cikarimini yapmis olsak bile, asagidaki Silhouette skoru 
#bize cluster=2' yi secmemiz gerektigini soyluyor. Gorseller ile aldigimiz 
#skorlar farkli bilgiler veriyor. Bu durumda, acaba datamizin kalitesini 
#dusuren birseyler mi var sorusu aklimiza gelmelidir.

K = range(2,11)

for k in K:
    model = AgglomerativeClustering(n_clusters = k)
    model.fit_predict(X)
    print(f'Silhouette Score for {k} clusters: {silhouette_score(df, model.labels_)}')
#Silhouette Score for 2 clusters: 0.45421907210414214
#Silhouette Score for 3 clusters: 0.3180902156279954
#Silhouette Score for 4 clusters: 0.13602999771899232
#Silhouette Score for 5 clusters: 0.16195986543258523
#Silhouette Score for 6 clusters: 0.22945281939930115
#Silhouette Score for 7 clusters: 0.24214267818856985
#Silhouette Score for 8 clusters: 0.3051367809988358
#Silhouette Score for 9 clusters: 0.29774346557180853
#Silhouette Score for 10 clusters: 0.2936409237035898
X.head()

#cluster=3 secerek modelimizi olusturduk. tahminlerimizi X datasina feature olarak ekledik. 
#Daha sonra Murder ve Assault feature' larini tahminlerimizin oldugu cluster' a gore gorsellestirdik.

model = AgglomerativeClustering(n_clusters=3, affinity = "euclidean", linkage = "ward")
clusters = model.fit_predict(X)
X["cluster"] = clusters
#Murder ve Assault feature' lari arasinda yuksek bir corr iliskisi oldugunu goruyoruz, 
#ikisi de artma egiliminde. 3 cluster' a gore az suc isleyenler ile cok suc isleyenlerin 
#iyi ayirt edildigini (mor ve sari renk); orta seviyede suc isleyenlerin ise (yesil renk) 
#diger cluster' lar ile grift durumda oldugunu goruyoruz :

plt.scatter(X.Murder, X.Assault, c = clusters, cmap = "viridis")

#Asagida Murder' a gore populasyona bakma istedik. x ekseni Murder, y ekseni 
#populasyona karsilik geliyor. Bu grafikte class' larin birbirine cok grift 
#oldugunu goruyoruz. Bu bilgiye gore populasyonu incelememiz gerekir :

plt.scatter(X.Murder, X.UrbanPop, c = clusters, cmap = "viridis")

#Elimizde musteriden gelen bir label olmadigi icin tahminlerimize gore bir gruplandirma yaparak bunu gorsellestirdik :

clus_pred = X.groupby("cluster").mean().T
clus_pred

#cluster=3' e gore Murder ve Assault ve Rape' de class' larin birbirinden cok 
#iyi ayristigini goruyoruz. UrbanPop diger feature' lar kadar iyi bir ayrisim 
#yapamamis ama yine de nispeten iyi bir ayrim oldugunu soyleyebiliriz. 
#Bu gorselden herhangi bir inside elde edemedik. Asagida bir de boxplot' a bakalim.

sns.lineplot(data = clus_pred)

#Asagida class' lara gore boxplot' lari cizdirdik :

plt.figure(figsize = (20,6))

plt.subplot(141)
sns.boxplot(y = "Murder", x = "cluster", data = X)

plt.subplot(142)
sns.boxplot(y = "Assault", x = "cluster", data = X)

plt.subplot(143)
sns.boxplot(y = "Rape", x = "cluster", data = X)

plt.subplot(144)
sns.boxplot(y = "UrbanPop", x = "cluster", data = X);

#Tahminlerimiz olan cluster' lara gore Murder, Assault ve Rape' de datalarin 
#yogun oldugu kisimlarin birbirleriye grift olmadiklarini goruyoruz. 
#Whiskers' larda hafif sarkmalar var ama bunlar cok da onemli degil,onemli olan 
#yogun bolgeler. Fakat UrbanPop' da datalarin yogun oldugu kisimlarin birbirlerine 
#grift durumda oldugunu goruyoruz. Yukaridaki mean degerlerinden bir inside elde 
#edememistik fakat boxplot' lari inceleyerek bir inside saglayabildik. Bu modelde 
#gorsellestirme cok onemlidir.

X.cluster.value_counts()

# Modelimiz 3 cluster' a gore 50 eyaletten 26' sini suclarin dusuk oldugu yere, 
#15 tanesini orta, 9 tanesini de suclarin cok yuksek oldugu yere atamis.

#Modelimizin kalitesini dusuren UrbanPop sutununu datadan atarak modelimizi yeniden kuracagiz :

X2 = X.iloc[:, [0,1,3]]
X2.head()

#####Hopkins test
#Datamizin yeni hali ile Hopkins test skorumuz iyilesti :

hopkins(X2, X2.shape[0])
#0.07237476567669528
hc_ward = linkage(y = X2, method = "ward")
hc_complete = linkage(X2, "complete")
hc_average = linkage(X2, "average")
hc_single = linkage(X2, "single")
plt.figure(figsize = (20,12))

plt.subplot(221)
plt.title("Ward")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_ward, truncate_mode = "lastp", p = 10, show_contracted = True, leaf_font_size = 10)

plt.subplot(222)
plt.title("Complete")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_complete, truncate_mode = "lastp", p = 10, show_contracted = True, leaf_font_size = 10)

plt.subplot(223)
plt.title("Average")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_average, truncate_mode = "lastp", p = 10, show_contracted = True, leaf_font_size = 10)

plt.subplot(224)
plt.title("Single")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_single, truncate_mode = "lastp", p = 10, show_contracted = True, leaf_font_size = 10);

#Datamizin yeni haliyle sadece single parametresi cluster=3 secmemizi soyluyor. 
#Diger degerlercluster=2 secmemizi soyluyor. Yeni modele gore yeniden Silhouette 
#skorlarina bakmamiz ve gorsellerden destek almamiz gerekiyor :

K = range(2,11)

for k in K:
    model = AgglomerativeClustering(n_clusters = k)
    model.fit_predict(X2)
    print(f'Silhouette Score for {k} clusters: {silhouette_score(df, model.labels_)}')
#Silhouette Score for 2 clusters: 0.5843563041221426
#Silhouette Score for 3 clusters: 0.5408473507473215
#Silhouette Score for 4 clusters: 0.49680242574181405
#Silhouette Score for 5 clusters: 0.4610301160129345
#Silhouette Score for 6 clusters: 0.45309851046082883
#Silhouette Score for 7 clusters: 0.39604153006494913
#Silhouette Score for 8 clusters: 0.39684807539987044
#Silhouette Score for 9 clusters: 0.3700971634671143
#Silhouette Score for 10 clusters: 0.36251487312336017
#Yeni modelde Slihouette skorlarinin yukseldigini goruyoruz. cluster=3 skoru 
#neredeyse cluster=2 skoruna yaklasmis, 23 puan birden artmis. cluster=3' teki 
#kalitemiz cok arttigi icin, bastaki gorsellerden de cluster=3 secmemiz gerektigi 
#bilgisini elde ettigimiz icin cluster=3 secerek yolumuza devam edecegiz :

final_model = AgglomerativeClustering(n_clusters=3, affinity = "euclidean", linkage = "ward")
clusters = final_model.fit_predict(X2)
X2["cluster"] = clusters
X2.cluster.value_counts()

# Yeni modelde eyaletler arasi dagilim da daha orantili bir hale geldi. 
#Secimimizden emin olmak icin her feature icin boxplot grafiklerimizi de inceleyelim :

plt.figure(figsize = (20,6))

plt.subplot(131)
sns.boxplot(y = "Murder", x = "cluster", data = X2)

plt.subplot(132)
sns.boxplot(y = "Assault", x = "cluster", data = X2)

plt.subplot(133)
sns.boxplot(y = "Rape", x = "cluster", data = X2)

#Ilk modelde datalarin yogun olduklari bolgeler birbirinden iyi ayrismisti fakat 
#whiskers' larda grift bir durum vardi. Yeni modelimizde whiskers' larin da birbrilerinden 
#oldukca iyi ayristigini gorebiliyoruz. Kumelenme kalitesi oldukca artti.

#Dendogram bize cluster=2 secmemiz gerektigini soylese bile biz gorseller ve 
#Silhouette score ile cluster=3 secmemiz gerektiginde karar kildik. Clusteering 
#metodlarinda kullanabilecegimiz butun yontemleri kullanarak en kaliteli 
#clustering' i yaptigimizdan emin olmak cok onemli.

######Evaluation
#0 : states with high crime rates
#1 : states with low crime rates
#2 : states with medium crime rates
#Yukaridaki boxplot degerlerine bakarak 0 class' ina high, 1 class' ina low, 
#2 class' ina da medium atamasini yaptik. Yukaridaki modelde bu degerler farkliydi, 
#her yeni kurulan modelde class' larin yerleri degisebilir :

#Model sonucu elde ettigimiz predict' leri de X2 datasina ekledik :

X2["crime_level"] = X2["cluster"].map({0:"high", 1:"low",2:"medium"})
X2.head()

df.index

#Eyalet isimleri datada index idi. Bunlari da datamizin sonuna ekledik :

X2["States"] = df.index
X2

X2[X2.crime_level=="low"]["States"]


#%%
#################################################################################
#LESSON 19 
#################################################################################

#####Principal Compnent Analysis(PCA)
#The Curse Of Dimensionality(Buyuk datalarla bas etme yolu)
#*Dataya eklenen her feature maliyet olarak geri doner.
#*Feature sayisi arttikca gorsellestirme imkani zorlasir
#*Data noktalari arttikca datanin egitilmesi zorlasir
#*Yani dataya eklenen herbir sutun ,zaman,maliyet olarak bize geri doner 
#*Bu sorunlardan kurtulmak icin dimension reduction tekniklerini uyguluyoruz

#*EDA zorlugubu ortadan kaldirmak icin,
#*Datayi 2D ve 3D boyuta indirip gorsellestirmek icin,
#*Sutun sayisi arttikca,complexity,overfitting.multicollinearity sorunlari
#ortaya cikar.Bunlari ortadan kaldirmak icin ,
#!!!!Dimension reduction ile feature sayisi dusuruldugunde bu sorunlarin hepsi 
#ile mucadele edilir.Sutun sayisi az,satir sayisi fazla oldugunda modeller 
#daha iyi ogrenir.Dimension rediction yapmanin bir sebebi de budur

#####Principal Compnent Analysis(PCA):
#Dimensioni dusurmek icin kullanilan populer bir tekniktir
#PCA; yuksek boyuttaki datalarda bir takim bilgilerden feragat ederekmbir nevi 
#onlarin izdusumleri ile daha az feature(boyut) ile onlari tanimlar

#3 featureli bir datadan sadece ortadaki kesisen kisim alinirsa data,1d boyuta iner.
#Butun bilginin sadece yaklasik %70 i alinir gibi olur.(1 feature ile)

#Teknik duzeyde PCA 'varyansi paylasan degisken kumelerini' tanimlar ve bu 
#varyansi temsil edecek bir degisken olusturur.
#2 feature li datada v1,datanin varyansini en iyi tanimlayan component olur.
#Fakat line dan uzakta kalan tanimlanamayan bazi noktalar kalir.daha fazla component
#istiyorsak bir de v2 cizeriz, o eksendeki iz dusumlerini aliriz
#Amacimiz en az feature ile en cok varyansi aciklamak

#Olusan her bir componente 'Princibal Component' denir.Princibal component 
#uyguladigimizda modele herhangi bir sinir koymazsak,feature adedinde
#component olusur
#Bu componentlerde belli oranlarda total varyansi karsilar
#18 feauteli bir datadan 18 component alinirsa hicbirinin birbiri ile corr
#iliskisi olmaz.Hepsinin birbiriyle coo u sifir olur.Bu componentler birer feature gibi davranir
#Yani bir dataya PCA yapildiktan sonra multicollinearity den soz edilmez
#3 feautre li bir dataya PCA uygulandiginda birbiriyle corr iliskisi olmayan 
#3 component olusur.(Her component icindeki featurelar arasi corr ilislisi olabilir)
#Her component icinde her feature icinden belli oranda bilgi vardir


#####Steps of PCA
#1-Once data standardize edilir.(z score a gore scale islemi).Boylece mean noktasi
#bulunmus olur
#2-Mean noktasi uzerinden her bir sample in birbirlerine gore Covariance Matrixleri 
#ortaya cikar
#3-Linear transformasyon ile Eigenvectors ve Eigenvalues hesaplanir
#Eagenvectorden gelen degerler,total varyansin kacta kacini karsiladigini soyler
#Eger bir sinir koymazsak feature sayisi kadar komponent olusur
#4-Eger bir 'k' sayisi veya 'varyans' belirlenirse o kadar component olusturulur.
#O zamanda %100 varyans karsilanamaz.Secilen component kadar varyans karsilanir

#!!!!!Gorsellestirme icin component sayisi 2 veya 3 e dusurulmek zorunda.
#Eger gorsellestirme istenmiyorsa,amac overfitting veya multicolineartiy 
#ile mucadele ise cok daha fazla component secilebilir
#1-Standardization
#2-Coverion Matrix
#3-Compute Eigenvectors and Eigenvalves
#4-Choose 'k' eigenvectors with the largest eigenvalues

######PCA AVANTAJLARI:
#Featurelar arasi corr u yok eder
#Algoritmanin performansini artirir
#Overfitting azaltir
#Gorsellestirme yapilir

###DEZAVANTAJLARI
#*Yorumlanmasi zor(Her componentte her featuredan bilgi var)
#*Standardization gerekli.(Meane gore bir vektor ciziliyor ki covariance matrixleri duzgun olsun)
#*Bilgi kaybi.(Featurelarin boyutunu dusurmek icin bir takim bilgilerden vazgeciliyor)
#*PCA,total varyansin belli bir kismini almayarak boyut dusurme islemidir



#######Principal Component Analysis-Unsupervised
#PCA bir boyut kucultme islemi oldugu icin hem supervised hem de unsupervised 
#algortmalarinda kullanilabilir. Bir nevi bir feature selection islemidir.

###############################################################################
#%%%NOTEBOOK
###############################################################################
###PCA Asamalari
#1. Data standardize edilir. (StandartScale ile)
#2. Mean noktasi uzerinden her bir sampl' in birbirlerine gore Covariance Maxrix degerleri ortaya cikar.
#3. Linear transformasyon ile Eigenvectors ve Eigenvalues hesaplanir.

#Eigenvector' den gelen degerlere gore componentler siralanir. Eigenvector' den 
#gelen degerler, total varyansin kacta kacini karsiladigini soyler. Eger bir sinir 
#koyulmaz ise feature sayisi kadar component olusur.

#4. 'k' sayisi veya 'varyans' belirlenerek component sayisi ayarlanir. Bu 
#sekilde de varyansin %100' u karsilanamaz, bir takim bilgilerden feragat 
#edilmis olur. Secilen component kadar varyans karsilanmis olur.

#Gorsellestirme yapilmak isteniyorsa component sayisi 2 veya 3' e dusurulmek 
#zorundadir. Eger amac gorsellestirme degil, overfitting veya multicollinearity 
#ile mucadele etmek ise cok daha fazla component secilebilir.

 
#####Imports

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (10,6)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
#Bu calismada sklearn kutuphanesinden import ettigimiz gogus kanseri ile ilgili
#bir data setini kullanacagiz. Data setinin isminin basina load_ yazip datayi import edebiliyoruz :

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
#print(cancer.DESCR), bize data hakkinda bilgi verir :

print(cancer.DESCR)

#Unsupervised bir algoritma kullanacagimiz icin class label olan WDBC-Malignant 
#ve WDBC-Benign' i kullanmayacagiz. Sadece en son asamada basarimizi olcmek 
#icin gorsellestirmede bu labeli kullanacagiz.

#X icin cancer datasindan feature_names' leri aldik. Datadaki ilk sutunlar 
#kanser dokulari ile ilgili olcumler, sonraki sutunlar ise bu degerlerin 
#olcumlerindeki hata metrikleri :

X=pd.DataFrame(cancer.data,columns=cancer.feature_names)
X.head()
X.shape
#(569, 30)

X.describe().T

#Target label' da 357 tane iyi huylu, 212 tane ise kotu huylu tumor var. Datada 
#hedef label' in 1 olmasi gerektigi icin asagida map fonksiyonunu kullanarak 1 
#ve 0 class' larinin yerlerini degistirdik :

pd.DataFrame(cancer.target).value_counts()
#1    357
#0    212
y = pd.Series(cancer.target).map({0:1, 1:0})

#Asagida X ve y' yi birlestirerek mean degerlere bakmak icin bir groupby islemi 
#yaptik. Fakat son asamaya kadar y labeli kullanmayacagiz. 
#Target label yokmus gibi davranacagiz :

a = pd.concat([X,y], axis=1)
a
a.groupby(0).mean()
 
#######PCA with sklearn
####Scaling Data
#PCA isleminden once dataya scale islemi mutlaka uygulanmak zorunda. 
#Diger yontemlerde gerek gorulmediginde scale islemi yapilmayabilirdi 
#fakat PCA icin zorunlu ve sadece StandardScaler yapilabilir, diger scale 
#islemleri uygulanamaz. Cunku datadaki mean degerler alinmak zorunda, mean 
#degerlerine gore her bir sample' in Coveriance Matrix' leri ortaya cikar. Daha 
#sonra linear transformasyon ile Eigenvector' ler hesaplanir. Boylece birbirleri 
#ile corr olma durumlari ortadan kalkar :

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)
#Scale edilmis X'e fit_transform islemini uyguladik ve sonra DataFrame' e donusturup 
#multicollinearity sorunu var mi diye heatmap ile baktik.

#Tumorun yaricapi, cevresi, alani gibi feature' larimiz var. Bunlarin birbirleri 
#ile corr olmalari beklenen bir problem, fakat PCA multicollinearity sorununu 
#hallettigi icin bu durumla ilgilenmiyoruz :

df_scaled_X = pd.DataFrame(scaled_X, columns = X.columns)
plt.figure(figsize = (23, 7))
sns.heatmap(df_scaled_X.corr().round(2), annot = True)

######PCA
#Eger data linear bir modele uygunsa veya datada cok kuvvetli feature' lar 
#varsa PCA cok guzel sonuclar verir. Bu durumlarda 30 feature ile elde edilen 2 
#component ile, alinan skorlar cok iyi olabilir. Data linear degilse veya cok 
#kuvvetli feature' lar yoksa boyut cok fazla kucultulemez ve iyi sonuclar alinmaz. 
#Mesela 100 feature' li linear veya guclu feature' lari olan bir datadan 10 
#component ile varyansin %90' i alinabilir. Fakat bu data nonlinear ise belki 
#50 component ile varyansin %90' i alinabilir.

#!! Bir data nonlinear ise fakat kuvvetli feature' lari varsa PCA, bu datalarda da iyi sonuclar verir.

from sklearn.decomposition import PCA
######PARAMETERS
#N_COMPONENTS ----> Datadaki feature ve sample sayisindan hangisi min ise o 
#kadar sayida component alir. (Default=None)

#Bunun haricinde oynayabilecegimiz tek parametre random_state parametresi.

#Asagida component 2' yi sectik ve buna gore fit_transform islemini yaptik. fit 
#isleminde Eigenvector degerleri ile feature degerleri carpilir ve elde edilen 
#degerler ile yeni componentler olusturulmus olur.

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_X)
#component=2 dedigimiz icin asagida her feature icin 2 component olustu. Bu 
#componentleri daha okunakli hale getirebilmek icin DataFrame' e donusturduk :

principal_components

component_df = pd.DataFrame(data = principal_components, columns = ["first_component", 
                                                                    "second_component"])
component_df


########Eigenvalues & Eigenvectors
#Eigenvalues & Eigenvectors PCA' in arka planinda calisan iki vektordur. 
#Bizim donusumu yaparken kullandigimiz vektorler Eigenvector' lerdir. 
#Butun Eigenvector' leri temsilen cizilen bir vektore de Eigenvales denir.

#n_component=2 dedigimizde 30 feature' dan 2 component olustu. Bu iki component' in 
#icinde her feature' dan belli oranlarda mutlaka bilgi vardir. Genellikle ilk component' te 
#tum feature' lardan cok daha fazla katkilar vardir. Sonraki gozlemlerde bu oran gittikce duser. 
#Bilgiyi aktarma isleminde devreye Eigenvector' ler girer. Eigenvector degeri ile feature' daki 
#deger carpilip toplanir ve boylece component degerleri elde edilmis olur. Bu hesaplamanin 
#nasil yapildigini asagida gosterecegiz.

#Asagida array icinde 2 farkli liste var. Bunlardan ilki 1. eigenvector, digeri 
#2. eigenvector degerleri. Bunlar, yukaridaki principal_components' leri 
#olustururken kullanilacak olan eigenvector degerleridir. Bunlarin ne oldugunu 
#anlamak icin asagida DataFrame' e donusturduk ve 2 icerigi de tek tek inceledik :

#Eigenvectors
pca.components_  

eigenvectors_first_component = pca.components_[0]   # 0. indexteki ilk eigenvector degerleri.
eigenvectors_first_component

#Olusan ilk component' te mean concave points feature' inin katkisi 0.260854 imis, 
#en fazla katkiyi o saglamis. Yani asagidaki degerler her feature' in ilk component'e 
#yaptigi katki. Her feature' in az da olsa mutlaka bir katkisi vardir :

first_eigenvectors = pd.DataFrame(eigenvectors_first_component, index=X.columns, 
                                  columns=["first_eigenvectors"]).sort_values("first_eigenvectors", ascending=False)

first_eigenvectors      # Column kismina X' in column isimlerini vererek buyukten kucuge siraladik :

eigenvectors_second_component = pca.components_[1]
eigenvectors_second_component        # 1. indexteki ikinci eigenvector degerleri.

#Ikinci component' te mean fractal dimension feature' i 0.366575 ile en fazla katkisi olan feature :

second_eigenvectors = pd.DataFrame(eigenvectors_second_component, index=X.columns, 
                                   columns=["second_eigenvectors"]).sort_values("second_eigenvectors", ascending=False)

second_eigenvectors   

#2 component icin olusan Eigenvector degerlerini burda birlestirip birbirleriyle 
#karsilastirdik. Ilk feature' larin birinci component' e cok buyuk katkisi varken 
#ikinci component' te nerdeyse hic katkilari yok. Burda aklimiza soyle bir soru geliyor : 
#Ilk component kanser hastalarinin teshisinde kullanilan onemli feature' lari temsil 
#ederken, 2. component kanser olmayan hastalari gosteren onemli feature' larin toplandigi component olabilir mi?

#Bazi feature' lar iki component' te de birbirlerine yakin degerler almislar. 
#Bu feature' larin hem kanser olan hem de kanser olmayan hastalarda ayni derecede 
#bulunabilme ozelligi var. Bunlar kanser hastaligini tespit etmede onemli olmayan feature' lar olabilir.

#Degerlerin (+) veya (-) olmasi; o feature' in component ile olan corr iliskisini gosterir. 
#(+) ise feature' in degeri arttikca component' in degeri de artar. (-) ise feature' in degeri 
#arttikca component degeri azalir. Demek ki component' in ilkinde kanser hastalarinin 
#teshisinde kullanilan feature' lar toplanmis; ikincisinde ise kanser olmayan hastalari 
#tespit etmek icin onemli olan feature' lar toplanmis.

pd.concat([first_eigenvectors, second_eigenvectors], axis=1)

df_scaled_X.loc[0]

#df_scaled_X' in ilk sutunundaki tum degerleri aldik ve ilk Eigenvector degerleri 
#ile carpip topladik, sonra ikinci Eigenvector degerleri ile carpip topladik. 
#Bu degerlerin asagidaki component_df' in 0. indexindeki sayilarla ayni oldugunu 
#goruyoruz. (Arkada donen matematigin ne oldugunu anlamak icin manuel olarak bu islemleri yaptik.)

(df_scaled_X.loc[0] * eigenvectors_first_component).sum()
#9.192836826213236
(df_scaled_X.loc[0] * eigenvectors_second_component).sum()
#1.9485830707766398
component_df.head()

#30 feature' in her biri bir Eigenvector ile carpildiktan sonra elimizde 30 
#boyutlu yeni vektor degerleri oldu. Bunlarin 30 tanesini temsil eden tek bir 
#vektor cizilir ki bu da EigenValues' dur. Asagidaki degerler her component icin 
#bu vektorun kuvvetini gosterir. Bu vektor ne kadar buyukse o kadar cok bilgi tasiyor 
#demektir. Ilk component' in ikincisinden daha fazla bilgi tasidigini goruyoruz :

pca.explained_variance_ #Eigenvalues
#array([13.30499079,  5.7013746 ])
###Corr between components
#Component' ler arasinda corr iliskisi olmadigini soylemistik. Bunun dogrulugunu 
#asagida teyit ettik; componentler arasi herhangi bir corr iliskisi olmadigini 
#goruyoruz. Yukaridaki cikarimlarimizda ilk component' te kanser hastaliginin 
#teshisi icin kullanilan onemli feature' larin toplandigi, ikincisinde ise kanser 
#hastasi olmayanlarin teshisinde kullanilan feature' larin toplandigi cikarimini 
#yapmistik. Bu durumda da iki component arasinda herhangi bir corr iliskisinin olmasi beklenmez :

sns.pairplot(component_df)

#Kanser hastaliginda tumor buyudukce tumorun yaricapi, cevresi, alani da buyur. 
#Datamizda bu bilgilerin hepsi ayri ayri feature' larda mevcut. Yuksek corr iliskisi 
#olan feature' larin hepsi ayni component icinde toplanirlar. Modelimizde bu feature' 
#larin hepsi ilk component' te toplandilar. Diger component icinde de kanser hastaligi 
#ile negatif iliskisi olan feature' lar toplandigi icin componentler arasi multicollinearity 
#sorunu halledilmis oldu.

component_df.corr().round()

###Finding optimal number of components
#explained_varianceratio ---> Aciklanabilir varyans orani. Datanin kendi uzerinde 
#tuttugu bilgi ilk etapta %100 iken, 2 componentli bir PCA islemi uygulandiginda; 
#ilk component bu bilginin %44' unu, ikinci component ise %18' ini barindiriyor :

pca.explained_variance_ratio_
#array([0.44272026, 0.18971182])
#Bu iki component' in tuttugu bilgilerin cumulative toplamini aldik. 2 component 
#toplam datanin %63' luk bilgisini barindiriyor :

pca.explained_variance_ratio_.cumsum()
#array([0.44272026, 0.63243208])
#Asagida en iyi component sayisini aralik vererek bulmak icin Elbow metoduna 
#benzer bir sekilde plot grafigi cizdirdik. Herhangi bir n_component sayisi belirtmeden 
#PCA' ya fit islemini uyguladik. Min feature sayisi 1, max 30 oldugu icin 1-31 araligindaki 
#tum component sayilarina bakmasini istedik.

#np.cumsum(pca.explained_varianceratio ---> X araligindaki her bir deger 
#icin aciklanabilir varyans degerinin (bilgi) her asamada cumulative toplamini goster.

#n_component=5' e gore neredeyse %90' lik biur bilginin alindigi, 
#n_component=10' da neredeyse %95' lik bir bilginin alindigini goruyoruz.

#Component sayisi belirlenirken genellikle %75-%95' lik oranlar secilir. 
#(Data linear veya cok guclu feature' lar olursa cok a component ile cok yuksek 
# bilgiye ulasilabilir.)

#Kaynaklara gore data linear ise veya cok guclu feature' lar var ise, bilgi %75' 
#in altinda olsa bile (bu datada oldugu gibi), daha kucuk component degerleri 
#denenip sonuclarin kontrol edilmesi tavsiye edilir. Boyle datalarda daha kucuk 
#componentler ile alinan skorlar cok yuksek cikabilir.

pca = PCA().fit(scaled_X)
x = range(1,31)
plt.plot(x, np.cumsum(pca.explained_variance_ratio_),)
plt.xlabel("Component count")
plt.ylabel("Variance Ratio")

pca = PCA(n_components=30).fit(scaled_X)
#n_component=30 sectik ve bu deger icin Eigenvalues'i (tum componentleri temsil 
#eden tek vektor buyuklukleri), aciklanabilen varyansi, aciklanabilen varyansin 
#cumulative toplamini asagida bir degisken icine tanimlayarak 30 feature icin 
#de gorsellestirecegiz :

my_dict = {"explained_variance":pca.explained_variance_,                             # Aciklanabilen varyans (EigenValues)
        "explained_variance_ratio":pca.explained_variance_ratio_,                    # Her bilesende ne kadar bilginin toplandigi
        "cumsum_explained_variance_ratio":pca.explained_variance_ratio_.cumsum()}    # Bilgilerin cumulative toplami
#olusturdugumuz bu degiskeni DataFrame'e cevirdik.

#explained_variance (EigenValues) ne kadar buyukse o componentte o kadar fazla 
#bilgi toplanacagini soylemistik. Burada da ilk component' in EigenValues' unun 
#en buyuk oldugunu ve diger componentlerde giderek kuculdugunu goruyoruz :

#explained_variance_ratio degerinin de ilk componentte en fazla oldugunu, diger 
#componentlerde azalarak en son 0 degerine cok yaklastigini goruyoruz.

#cumsum_explained_variance_ratio degerine baktigimizda da 7. componentte %90' lik 
#bilginin elde edildigini goruyoruz. Bu data icin %75' lik bilgi icin 3 veya 4 
#component ile baslanabilir. Fakat datamiz linear oldugu icin ve guclu feature' larimiz 
#oldugu icin %75' lik sinira takilmayip component sayisi 2 veya 3 olarak da secilebilir. 
#Gorsellestirme yapabilmek icin en fazla 2 veya 3 component secilmek zorunda.

df_ev = pd.DataFrame(my_dict, index = range(1,31))
df_ev

sns.barplot(x = df_ev.index, y= df_ev.explained_variance_ratio);

#Gorsellestirme yapabilmek amaciyla component sayimizi 2 olarak sectik :

pca = PCA(n_components=2)
#Scale edilmis datamiza tekrardan fit_transfor islemini uyguladik ve degerlerimizi 
#DataFrame haline getirdik :

principal_components = pca.fit_transform(scaled_X)

component_df = pd.DataFrame(data = principal_components, columns = ["first_component", 
                                                                    "second_component"])
component_df


#Datadaki 30 feature' in bilgisini barindiran 2 component ile modelimizi kuracagiz. 
#Hangi feature' in componentlere ne kadar katki sagladigi ile ilgili bilgiyi 
#yukaridaki calismalarda gostermistik fakat artik componentler ile birlikte bu 
#bilgileri kaybettik. Artik componentler arasinda onemli onemsiz yorumu yapamiyoruz. 
#Yapabilecegimiz yorumlari notebook' un sonunda aciklayacagiz.

#####K-Means Algorithm
from pyclustertend import hopkins
#Hopkins testimiz 0.5' in altinda cikti. Yani datamiz clustering islemine meyilli :

hopkins(component_df, component_df.shape[0])
#0.13982104818492913
#inertia degeri kumelerin icindeki sample' larin center etrafinda ne kadar yogun 
#bir sekilde kumelendigini olcen bir degerdir. Bu deger ne kadar kucuk olursa 
#kume kalitesi o kadar yuksek demektir. Burada cluster sayisini 2-10 arasini 
#verdik. Bu degeri 2' den baslatmak mantiklidir. Cunku keskin dususun bittigi 
#noktayi k degerimiz olarak sececegiz ve genelde 1' den 2' ye geciste en keskin 
#dusus gorulur. Bu da sonuclarimi yaniltabilir.

#Verilen butun k degerleri icin fit islemini yapmasi icin bir for dongusu tanimladik 
#ve buradan cikan sonuclari asagidaki grafikte cizdirdik :

from sklearn.cluster import KMeans

ssd = []

K = range(2,10)

for k in K:
    model = KMeans(n_clusters = k, random_state=42)
    model.fit(component_df)
    ssd.append(model.inertia_)
plt.plot(K, ssd, "bo-")
plt.xlabel("Different k values")
plt.ylabel("inertia-error") 
plt.title("elbow method")

ssd

#diff() kullanarak tum degerleri bir sonrakinden cikararak keskin dususun durdugu 
#noktayi tespit etmek istedik. Keskin dususun durdugu noktanin n_cluster=3 oldugunu, 
#bundan sonra degerlerin yavas yavas azaldigini goruyoruz. Bu deger bize Elbow 
#metodunun verdigi deger. Asagida bir de yellowbrick sonuclarina bakacagiz.

pd.Series(ssd).diff()

df_diff =pd.DataFrame(-pd.Series(ssd).diff()).rename(index = lambda x : x+1)
df_diff

df_diff.plot(kind='bar');

#Yellowbrick keskin dususun durdugu noktayi n_cluster=4 olarak belirledi. 
#Bir de Silhouette skorlarimiza bakacagiz :

from yellowbrick.cluster import KElbowVisualizer

model_ = KMeans(random_state=42)
visualizer = KElbowVisualizer(model_, k=(2,9))

visualizer.fit(component_df)        # Fit the data to the visualizer
visualizer.show();

from sklearn.metrics import silhouette_score

range_n_clusters = range(2,9)
for num_clusters in range_n_clusters:
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(component_df)
    cluster_labels = kmeans.labels_
    # silhouette score
    silhouette_avg = silhouette_score(component_df, cluster_labels)
    print(f"For n_clusters={num_clusters}, the silhouette score is {silhouette_avg}")

#Silhouette skoruna gore en iyi kumelenme 2 degerinde, sonra 3' te. Skorlar da 
#birbirine cok yakin cikti. Asagida 1. ve 2. componenti gorsellestirerek 
#buradan da bir destek almak istedik :

sns.scatterplot(x = component_df.first_component, y= component_df.second_component, alpha=0.7);

#Gorselde sol tarafta yogun bir kumelenmenin oldugunu fakat sag tarafa dogru 
#sample' larin genis bir alana yayildigini goruyoruz. Genis alana yayilan bu 
#kisimlarda da farkli kumeler olabilir. Gorselden de net bir inside elde edemedik. 
#Elbow metodu 3 degerini vermisti fakat dususler arasinda cok buyuk ucurumlar yoktu. 
#Silhouette skorunda da 2 ve 3 degerleri arasinda cok fazla bir fark olmadigi icin 
#boyle bir datada cluster sayisi 2, 3 veya 4 secilebilir. Bu durumda musteriye öyle 
#bir bilgi verilebilir : Datada 2 cluster ile kanser veya kanser degil bilgisi olabilir 
#ya da 3 cluster ile 1. derece, 2. derece, 3. derece kanser hastalarinin oldugu bilgisi olabilir.

#Eger bir uzmandan bu datada kanser ve kanser degil olmak uzere 2 cluster oldugu 
#bilgisini alabilirsek yolumuza o sekilde devam edebiliriz.

#n_cluster=2 vererek modelimizi kurduk ve crosstab sonuclarimiza baktik :

model = KMeans(n_clusters =2, random_state=42)
clusters = model.fit_predict(component_df)
clusters

#Target labelimizi yukarida y olarak belirlemistik. Elimizdeki bu gercek bilgiler 
#ile predict verilerimiz arasinda crosstab islemimizi yaptik. Modelimizin 16+37=53 
#tane hata yaptigini goruyoruz. 569 tane sample' a gore iyi bir basari yakaladik :

ct = pd.crosstab(y, clusters)
ct

#####Interpreting PCA results
#PCA' den elde edilen componentleri yorumlayacagiz. Literaturlerde bunlarin nasil 
#yorumlanacagi ile ilgili net bilgiler yok.

#PCA' de onemli veya onemsiz feture diye bir kavram yok. Her feature' in mutlaka az da olsa katkisi var.

scaled_df = pd.DataFrame(scaled_X, columns=X.columns)
scaled_df.head()

#Yukarida scale edilmis X' i DataFrame'e cevirdik ve component edilmis df ile 
#ikisini concat ettik. Asagida bunlarin birbirleri ile olan corr iliskisine bakacagiz :

combined_df = pd.concat([component_df, scaled_df], axis =1)
correlation = combined_df.corr()
correlation.head()

#Corr iliskisine bakarken satir olarak ilk 2 satiri sectik, sutun olarak ise 2' den 
#sonraki sutunlari sectik. Cunku her iki component ile gercek feature' lar arasindaki 
#corr iliskisine bakarak PCA yorumunu yapmak istiyoruz :

fig, ax = plt.subplots(figsize=(20, 7))
sns.heatmap(correlation.iloc[:2,2:], cmap='YlGnBu', linewidths=.7, annot=True, fmt='.2f')
plt.show()

#Yukaridaki gorseli yorumlarken, 0.5 ve uzerindeki degerleri dikkate alacagiz.

#first_component' e baktigimizda su yorumu yapabiliriz : Eger 'median radius' (0.80) 
#buyurse bununla birlikte 'mean perimeter' (0.83), 'mean area' (0.81), 
#'mean concavepoints (tumorun kotu huylu olmasi)' gibi degerler buyuyecek. 
#Kanser hastaliginin teshisinde onemli olan butun feature' lar, first_component 
#ile birlikte artma egilimindeler. Demek ki basta yaptigimiz tahmin dogru. 
#first_component, kanser hastaliginin teshisinde onemli olan feature' larin 
#toplandigi bir component olmus.

#second_component ile en yuksek corr ilskisi 'meanfractal dimension', 'fractal 
#dimension error', 'worst fractal dimension' feature' lari olmus. Bir arastirma 
#yaparak fractal dimension' in kanser tespitinde onemli olmadigi, hem saglikli 
#hem de hasta olan bireylerde yuksek cikabiliecegi bilgisine ulastik. Bu yuzden 
#bu degerin yuksek cikmasini dikkate almiyoruz. Bu yuzden 0.5 uzerinde cikan 
#diger degerlere odaklanacagiz. 'mean radius' eksi (-) deger alarak kuculmus, 
#bununla birlikte cevre, alan gibi bilgiler de kuculmus. Bu feature' larin 
#second_component ile arasinda negatif yonde guclu bir corr iliskisi oldugunu 
#goruyoruz. Demek ki bu component, kanser hastasi olmayanlari tespit etmede 
#kullanilabilecek feature' larin toplandigi bir component olmus.

#'compactness error' gibi feature' lar her iki componentte de yuksek cikmis. 
#Demek ki kanser tespitinde bu feature' larin cok bir etkisi yok.

#!!! Component ile feature' lar arasinda yuksek corr iliskisinin olmasi, o 
#feature' larin kanser teshisinde en onemli feature' lar oldugu anlamina gelmiyor. 
#Bu bilgi icin mutlaka feature importance islemi yapilmali !!!

#Asagida feature' lar ile target label' i birlestirdik ve aralarindaki corr 
#iliskisine baktik. Gercekten de kanser hastaligi arttikca yukarida belirttigimiz 
#feature' larin target ile olan corr iliskisinin arttigini goruyoruz. Yukaridaki 
#tahminlerimizi bu sekilde dogrulamis olduk :

df_new = pd.concat([X, y], axis=1).rename(columns={0:"target"})
df_new
plt.figure(figsize = (20, 10))
sns.heatmap(df_new.corr().round(2), annot = True)


#Asagida X eksenine ilk componenti, y eksenine ikinci componenti verdik ve hue 
#olarak da gercek degerlerimiz olan y' yi sectik. Bir sonraki gorselde bu grafikle 
#birlikte componentleri bir de tahmin degerlerimiz olan 'cluster' ile cizdirdik :

sns.scatterplot(x = component_df.first_component, y= component_df.second_component, hue=cancer.y, alpha=0.7)

#Gercek ve tahmin degerleri kiyaslarsak, iki component ayriminin cok basarili 
#bir sekilde yapildigini soyleyebiliriz. Iki class ayrimi, gercek ve tahmin 
#grafiginde birbirine cok benzer. Sadece bilgilerin grift oldugu noktalarda az 
#da olsa yanlis tahminlerin oldugunu goruyoruz. 30 feature ile elde edemeyecegimiz 
#bir gorseli boyutu 2' ye dusurerek 2 component ile datadaki bilginin %63' unu 
#alarak elde edebilmis olduk :

plt.figure(figsize = (20,10))

plt.subplot(121)
sns.scatterplot(x = component_df.first_component, y= component_df.second_component, hue=y, alpha=0.7,
                palette=['orange','green'])
plt.title("Actual")

plt.subplot(122)
sns.scatterplot(x = component_df.first_component, y= component_df.second_component, hue=clusters, alpha=0.7,
               palette=['orange','green'])
plt.title("K_means")





 
 











