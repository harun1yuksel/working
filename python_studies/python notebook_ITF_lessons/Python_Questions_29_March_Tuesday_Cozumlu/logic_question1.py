###Endüstri Mühendisliği Final soruları###

 # cc = 40
if False:
    cc = 2

def Helmet():
    if True:
        global cc
        cc = 40
        return cc

print(Helmet())
print(cc)  # local, enclosing, global, built-in (LEGB)