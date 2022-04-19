def amicable_nums_list(num1, num2):
    """ cheks if num1 and num2 are amicable numbers. If they are amicable returns True othervise return False"""
    amicable_list = []
    perfect_divisors_num1 = [i for i in range(1, num1) if num1 % i == 0]
    perfect_divisors_num2 = [i for i in range(1, num2) if num2 % i == 0]
    if num2 == sum(perfect_divisors_num1) and num1 == sum(perfect_divisors_num2) and num1 != num2:
        amicable_list.append(num1)
        amicable_list.append(num2)
        print(amicable_list)


for i in range(1, 10001):
     for j in range(i + 1, 10001):
        amicable_nums_list(i, j)