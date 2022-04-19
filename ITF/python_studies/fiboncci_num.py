print("Hello this is a Fibonacci number list maker")

fibonacci_list = [0, 1]
number = input("Please enter an integer to see the Fibonacci number list up to it:")

while not number.isnumeric():
    number = input("invalid entry\nPlease enter an integer to see the Fibonacci number list up to it:")


for x in fibonacci_list:
    if fibonacci_list[-1] < int(number):
        fibonacci_list.append(fibonacci_list[-1] + fibonacci_list[-2])

fibonacci_list.pop()
print(fibonacci_list)
