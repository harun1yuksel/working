divisor = 2  # smallest prime number!
empty_list = []
empty_set = set()
for number in range(1, 21):
    divisor = 2
    sub_list = []

    for j in range(1, number):  # bu döngüde j'ye atamayı sadece döngü sayısı elde etmek için kullanıyoruz!
            
        if number % divisor == 0:
            sub_list.append(divisor)  
            number /= divisor  # aynı bölen bi daha bölmesin diye
            
        else:
            divisor += 1
    empty_list.append(sub_list)

print(empty_list)