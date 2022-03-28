age = int(input("Please enter your age.  "))
if age > 75:
    age_risk = True
else:
    age_risk = False    
chronic_disease = input("Do you have a chronic disease please enter y or n.   ")
if chronic_disease == "y":
    chronic_disease_risk = True
else: 
    chronic_disease_risk = False
smoking = (input("Do you smoke please enter y or n   "))
if smoking == "y":
    smoking_risk = True
else:
    smoking_risk = False
death_risk = bool(age_risk and chronic_disease_risk and smoking_risk)
if death_risk is True:
    print("You are under death risk")
else:
    print("You are not under death risk.")
