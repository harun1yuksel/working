"""
Kütüphanede çalışanların kaydını alan program. 


Pseudocode;
kullanıcıdan çalışanın ismini al ve boş bir listeye append et
kullanıcıdan çalışan telefon ve adres bilgisi al ve başka boş bir listeye append et
telefon bilgisi integer olmalı, farklı girilirse ikaz et
telefon için yeni girdi iste
girilen kayıt sayısını bir sayaç ile tut
listeleri zip fonksiyonu ile dictionary key : value pair haline getir



"""



laborer = []
information_list = []
toplam = 0
repeater = "y"
while repeater == "y" or repeater != "n":

    laborer.append(input("name : "))
    phone = input("phone number : ")
    address = input("address : ")
    
    
    if type(phone) != int:
        try:
            phone = int(phone)
        except:
            phone = int(input("please input an INTEGER value as phone number : "))
        finally:
            information_list.append(dict(phone = phone, address = address))  # dict = {}  dict(phone = 4569854,)
            print(laborer, information_list)
            toplam += 1
    else:
        information_list.append(dict(phone = phone, address = address))
        print(information_list)
        toplam += 1

    repeater = input("for new input press 'y' otherwise press 'n' : ").lower().strip()

    

result = dict(zip(laborer, information_list, ))
print("kayıt sayısı : ", toplam)
print(result)
# for i in zip(laborer, information_list, ):  alt alta yazdırmak için kullanılabilir
#     print(i)


