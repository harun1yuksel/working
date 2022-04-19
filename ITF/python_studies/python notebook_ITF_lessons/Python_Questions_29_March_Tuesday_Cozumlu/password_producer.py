"""
Bilgisayar programcılığı vize sorusu;
Kullanıcı tarafından girilen ad, soyad ve okul numarası ile password oluşturma. Password--> ad ve soyad bilgilerinin ilk ve son harfleri ile okul numarasının sonundaki 4 rakamdaki tek sayıların toplamından oluşacaktır. Çıktıyı veren programı yazınız. 


pseudocode;
kullanıcıdan string olarak ad, soyad ve okul numarası al.
ihtiyaç olduğunda numarayı integer a çevir.  
ad ve soyad bilgilerinin ilk ve son harflerini slicela. 
okul numarasının son dört rakamını indexle

"""
ad = "Peyami" # input("ad girin : ")
soyad = "Safa" # input("soyad girin : ")
s = "01011950" # input("numara girin : ")

###AMELE STYLE####

# if int(s[-1]) % 2 == 1:
#     n1 = int(s[-1])
# else:
#     n1 = 0
# if int(s[-2]) % 2 == 1:
#     n2 = int(s[-2])
# else:
#     n2 = 0
# if int(s[-3]) % 2 == 1:
#     n3 = int(s[-3])
# else:
#     n3 = 0
# if int(s[-4]) % 2 == 1:
#     n4 = int(s[-4])
# else:
#     n4 = 0
# toplam = n1 + n2 + n3 + n4

# print(ad[0] + ad[-1] + soyad[0] + soyad[-1] + str(toplam))

###NEWBIE STYLE#####
# n1 = 0
# n2 = 0
# n3 = 0
# n4 = 0  # else lere gerek kalmadı
# if int(s[-1]) % 2 == 1:
#     n1 = int(s[-1])
# if int(s[-2]) % 2 == 1:
#     n2 = int(s[-2])
# if int(s[-3]) % 2 == 1:
#     n3 = int(s[-3])
# if int(s[-4]) % 2 == 1:
#     n4 = int(s[-4])
# toplam = n1 + n2 + n3 + n4

# print(ad[0] + ad[-1] + soyad[0] + soyad[-1] + str(toplam))

####JUNIOR STYLE####

# toplam = 0  # n leri tek tek tanımlamaya gerek kalmadı

# if int(s[-1]) % 2 == 1:
#     toplam += int(s[-1])
# if int(s[-2]) % 2 == 1:
#     toplam += int(s[-2])
# if int(s[-3]) % 2 == 1:
#     toplam += int(s[-3])
# if int(s[-4]) % 2 == 1:
#     toplam += int(s[-4])  # hepsi if olacak çünkü ayrı şartlar birini alıp diğerinden vazgeçmemesi gerekir o yüzden elif olmadı



# print(ad[0] + ad[-1] + soyad[0] + soyad[-1] + str(toplam))


####MID LEVEL DEV STYLE####

# toplam = 0  # n leri tek tek tanımlamaya gerek kalmadı
# for i in s[-4:]:
#     if int(i) % 2 == 1:
#         toplam += int(i)


# print(ad[0] + ad[-1] + soyad[0] + soyad[-1] + str(toplam))


###ADVANCED STYLE#####

# toplam = sum([int(i) for i in s[-4:] if int(i) % 2 == 1])
# print(ad[0] + ad[-1] + soyad[0] + soyad[-1] + str(toplam))


#####SENIOR STYLE#####

# def passworder():  # senior fonksiyona çevirir istediği yerde çağırır kullanır :)

#     toplam = sum([int(i) for i in s[-4:] if int(i) % 2 == 1])

#     return (ad[0] + ad[-1] + soyad[0] + soyad[-1] + str(toplam))


# print(passworder())


####SUPER SENIOR STYLE#####

(lambda ad, soyad : ad[0] + ad[-1] + soyad[0] + soyad[-1] + str(sum([int(i) for i in s[-4:] if int(i) % 2 == 1])))(ad, soyad)



