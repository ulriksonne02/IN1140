#Oppgave 1)
# Lingvistiske nivåer

#Fonetikk: Studier av menneskelige lyder
#Fonologi: klassifiseringen av lyder innenfor et språklig system
#Morfologi:
    #Hvordan ord er bygd opp
    #affikser:pre/suffiks
    #ordstruktur
#Syntaks:
    #Grammatikk
    #Hvordan setninger er bygd opp
#Semantikk:
    #Lingvistiske betydningen av ord/setninger
#pragmatikk
    #Den faktiske betydningen av ord/setninger

#Oppgave 2)
#i: Hva er forskjellen mellom bundne og frie morfemer?
    #Frie morfemer fungerer som ord, uavhengi av andre morfemer
        #Eksempel: disse, med, til
    #Bundne morfemer fungerer bare sammen med rot morfemer
#ii: De bundne morfemene kan videre deles inn i minst to undergrupper. Hva er disse og hva skiller dem fra hverandre?
    #affikser:prefikser og suffikser: bundne morfemer som kommer før etter en rot

    #Bøyingsmorfemer: -et, -ene, -ere
        #Bøyingsmorfemer er bundne morfemer som gir et rotmorfem ny form ved å bøye det. Bøyningsmorfemer kan utrykke bestemthet, kjønn, flertall, kasus, osv.
        #eksempel: barn -> barnet. bøying endrer bestemtheten.
    #Avledningsmorfemer:
        #Avledningsmorfemer er bundne morfemer som ofte endrer ordklasse
        #eksempel: barn -> barnslig. avledningssuffiks gjør et substantiv om til et adjektiv


#Oppgave 3)

#a)
setning_liste=[]
f = open("in01.txt")
min_streng=f.read()
setning_liste=[min_streng.split(" ")] #legg til


#b)
def finn_er():
    teller=0
    for i in range(len(setning_liste[0])): #for hvert ord i setning_liste
        teller+=setning_liste[0][i].count("er") #tell hvor mange ganger "er" skjer
    return teller

def finn_er_ending():
    teller=0
    for i in range(len(setning_liste[0])):
        if setning_liste[0][i].endswith("er")==True: #sjekk hvis et ord i setning_liste ender med "er"
            teller+=1
    return teller


#c
def finn_endinger():
    ending_liste=[]
    ending_streng=""
    for i in range(len(setning_liste[0])):
        ending_liste.append(setning_liste[0][i][-2:]) #går gjennom hvert ord i setning_liste for å legge for legge til de to siste karakterene i og legge dem til ending_liste
        #ending_streng=ending_liste
    for i in range(len(ending_liste)):
        ending_streng+=" "+ending_liste[i]
    return ending_liste

#Oppgave 4)

#b)

def finn_antall_linjer():
    teller=0
    for ord in min_streng: #for hvert ord i min_streng
        #print(i)
        if ord == "\n": #hvis ord er det samme som \n legg til 1 til teller
            teller+=1
    return teller

#a)
print("Total mengde ord:",len(setning_liste[0]))
print("Antall linjer i tekst:",finn_antall_linjer()) #printer antall linjer
print("forekommelser av 'er':",finn_er(),"\nforekommelser av ord som ender med 'er':",finn_er_ending()) #printer antall forekommelser av "er" og ord som ender med "er"
#Oppgave 5a)
x = open("oppgave 5.txt","a")
for ord in setning_liste[0]:
    x.write(ord+"\n")
#b) først og fremst er tegnsetting inkludert i ordene. Ikke alle linjer har ord i seg, bare tegn som (feks bindestrek).
x.close()
f.close()
