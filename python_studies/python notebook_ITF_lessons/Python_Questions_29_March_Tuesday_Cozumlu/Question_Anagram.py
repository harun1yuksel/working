"""
- Given a list of strings, group anagrams together.

- Example:

**Input:**
```
["eat", "tea", "tan", "ate", "nat", "bat"],
Output:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
```
Note:
All inputs will be in lowercase.
The order of your output does not matter.

Pseudocode:
listenin her elemanı üzerinde gezin
her elemanı alt elemanlarına ayır (set elemanları unique dir özelliğinden faydalanıyoruz)
bu set edilmiş elemanları listenin tüm elemanlarıyla karşılaştır
eleman bazında aynı olanları bir liste içine append et
en son hepsini bir liste içine append et

"""

given_list = ["shore", "snake", "tutor", "heart", "earth", "trout", "sneak", "horse", "coyote"]
new_list=[]   # 3 bu nedenle boş bir liste oluşturuyorum
result=[]
for i in given_list :    # 1 liste içinde her elemanı i ye atama yapıyorum
    if set(i) not in new_list :   # 2 set ile sıralanmış halini yeni bir listeye koymam gerek
        new_list.append(set(i))   # 4  yeni listeye ekliyorum

print(new_list)   #  5  aynı harflerin set içindeki durumunu görmek için
for j in range(len(new_list)) :   # 6 yeni listenin uzunluğu kadar bir döngü oluşturduk
    result.append([i.lower() for i in given_list if set(i)==new_list[j] ])  #  7 set edilmiş i kalıp olarak oluşturduğumuz 
    #  yeni liste içindeki ile eşitse bunların çıktısını liste olarak ver (list comprehension)
print(result)

