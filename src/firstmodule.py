def add(a,b):
    return a+b 

def addFixedValue(a):
    y=5
    return y +a 

print (add(2,3))
print (addFixedValue(4))

Text = 'dodo mylof,kdfuSDfhaersfaehrgnhaerngaherrenfjnerhfnDfhhgnehgdzbfgbgnhnjxfnjdgehrhnzdfnehrhydrhuyehruerrhg'
upText = Text.lower()
print (Text, upText)
count=0
vowel=0
for i in range(0, len(upText)):
       
        if upText[i]==(" "):
            count = count+1
        
        elif ((upText[i]=="a") or (upText[i]=="e") or (upText[i]=="i") or (upText[i]=="o") or (upText[i]=="u")):
           # print (upText[i])
            vowel=vowel+1
print ("The Text :", Text)
print ("Text Length :", len(Text))
print ("Total # of characters :", len(Text)-count)
print ("Total # spaces", count)
print ("Total # vowels :", vowel)
thistuple = ("apple", "banana", "cherry", "orange", "kiwi", "melon", "mango")
print(thistuple[2:5])


