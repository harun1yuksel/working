###Endüstri Mühendisliği Final soruları###
from threading import local


b = 50
def OuterFunction():
    
    b = 20

    def InnerFunction():
        nonlocal b
        b = 10
        
        c = 30
        return c

    return InnerFunction() + b


print(OuterFunction()) # LEGB
