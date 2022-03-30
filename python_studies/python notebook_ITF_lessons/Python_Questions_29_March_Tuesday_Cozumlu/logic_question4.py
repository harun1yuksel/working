###Endüstri Mühendisliği Final soruları###

z = 3 # 9
def Oracle(a, b, c):
    if a == 42:
        return b(c)
    return c


def Fonk0(x):
    return z + x ** 2


def Fonk1(y):
    global z
    z = z + y * 2
    return z + y


print(z, Oracle(42, Fonk1, 3))  # 3, 12

print(z, Oracle(42, Fonk0, 4))  # 9, 25

print(z, Oracle(42, Fonk1, 3))  # 9, 18

print(z, Oracle(22, Fonk1, 3))  # 15, 3



