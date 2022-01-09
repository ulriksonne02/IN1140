teller=0
liste1=["hjerner-","hjerner!","drever ","reer."]
#liste1.strip(".")
for ord in range(len(liste1)):
    print(liste1[ord].strip("."))
    if liste1[ord].strip(".").endswith("er")==True:
        teller+=1
print(teller)
print(liste1[1].endswith("er"))

streng=""
for i in range(len(liste1)):
    streng+=liste1[i]
print(streng)
print(index(liste1[0]))
