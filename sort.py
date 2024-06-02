import random
import math

lista = random.sample(range(0, 100), 32)

random.shuffle(lista)

copia_lista = lista.copy()

#1
#1
for i in range(0,16,2):
    if i // 2 % 2 == 0:
        if lista[i] > lista[i+1]:
            temp = lista[i]
            lista[i] = lista[i+1]
            lista[i+1] = temp
    else:
        if lista[i] < lista[i+1]:
            temp = lista[i]
            lista[i] = lista[i+1]
            lista[i+1] = temp

#2
#2
for i in range(0,16,4):
    if i // 4 % 2 == 0:
        if lista[i] > lista[i+2]:
            temp = lista[i]
            lista[i] = lista[i+2]
            lista[i+2] = temp
        if lista[i+1] > lista[i+3]:
            temp = lista[i+1]
            lista[i+1] = lista[i+3]
            lista[i+3] = temp
    else:
        if lista[i] < lista[i+2]:
            temp = lista[i]
            lista[i] = lista[i+2]
            lista[i+2] = temp
        if lista[i+1] < lista[i+3]:
            temp = lista[i+1]
            lista[i+1] = lista[i+3]
            lista[i+3] = temp
#1
for i in range(0,16,2):
    if i // 4 % 2 == 0:
        if lista[i] > lista[i+1]:
            temp = lista[i]
            lista[i] = lista[i+1]
            lista[i+1] = temp
    else:
        if lista[i] < lista[i+1]:
            temp = lista[i]
            lista[i] = lista[i+1]
            lista[i+1] = temp

#3
#3
for i in range(0,16,8):
    if i // 8 % 2 == 0:
        if lista[i] > lista[i+4]:
            temp = lista[i]
            lista[i] = lista[i+4]
            lista[i+4] = temp
        if lista[i+1] > lista[i+5]:
            temp = lista[i+1]
            lista[i+1] = lista[i+5]
            lista[i+5] = temp
        if lista[i+2] > lista[i+6]:
            temp = lista[i+2]
            lista[i+2] = lista[i+6]
            lista[i+6] = temp
        if lista[i+3] > lista[i+7]:
            temp = lista[i+3]
            lista[i+3] = lista[i+7]
            lista[i+7] = temp
    else:
        if lista[i] < lista[i+4]:
            temp = lista[i]
            lista[i] = lista[i+4]
            lista[i+4] = temp
        if lista[i+1] < lista[i+5]:
            temp = lista[i+1]
            lista[i+1] = lista[i+5]
            lista[i+5] = temp
        if lista[i+2] < lista[i+6]:
            temp = lista[i+2]
            lista[i+2] = lista[i+6]
            lista[i+6] = temp
        if lista[i+3] < lista[i+7]:
            temp = lista[i+3]
            lista[i+3] = lista[i+7]
            lista[i+7] = temp
#2
for i in range(0,16,4):
    if i // 8 % 2 == 0:
        if lista[i] > lista[i+2]:
            temp = lista[i]
            lista[i] = lista[i+2]
            lista[i+2] = temp
        if lista[i+1] > lista[i+3]:
            temp = lista[i+1]
            lista[i+1] = lista[i+3]
            lista[i+3] = temp
    else:
        if lista[i] < lista[i+2]:
            temp = lista[i]
            lista[i] = lista[i+2]
            lista[i+2] = temp
        if lista[i+1] < lista[i+3]:
            temp = lista[i+1]
            lista[i+1] = lista[i+3]
            lista[i+3] = temp
#1
for i in range(0,16,2):
    if i // 8 % 2 == 0:
        if lista[i] > lista[i+1]:
            temp = lista[i]
            lista[i] = lista[i+1]
            lista[i+1] = temp
    else:
        if lista[i] < lista[i+1]:
            temp = lista[i]
            lista[i] = lista[i+1]
            lista[i+1] = temp


#4
#4
for i in range(0, 16, 16):
    if i // 16 % 2 == 0:
        if lista[i] > lista[i+8]:
            temp = lista[i]
            lista[i] = lista[i+8]
            lista[i+8] = temp
        if lista[i+1] > lista[i+9]:
            temp = lista[i+1]
            lista[i+1] = lista[i+9]
            lista[i+9] = temp
        if lista[i+2] > lista[i+10]:
            temp = lista[i+2]
            lista[i+2] = lista[i+10]
            lista[i+10] = temp
        if lista[i+3] > lista[i+11]:
            temp = lista[i+3]
            lista[i+3] = lista[i+11]
            lista[i+11] = temp
        if lista[i+4] > lista[i+12]:
            temp = lista[i+4]
            lista[i+4] = lista[i+12]
            lista[i+12] = temp
        if lista[i+5] > lista[i+13]:
            temp = lista[i+5]
            lista[i+5] = lista[i+13]
            lista[i+13] = temp
        if lista[i+6] > lista[i+14]:
            temp = lista[i+6]
            lista[i+6] = lista[i+14]
            lista[i+14] = temp
        if lista[i+7] > lista[i+15]:
            temp = lista[i+7]
            lista[i+7] = lista[i+15]
            lista[i+15] = temp
#3
for i in range(0, 16, 8):
    if lista[i] > lista[i+4]:
        temp = lista[i]
        lista[i] = lista[i+4]
        lista[i+4] = temp
    if lista[i+1] > lista[i+5]:
        temp = lista[i+1]
        lista[i+1] = lista[i+5]
        lista[i+5] = temp
    if lista[i+2] > lista[i+6]:
        temp = lista[i+2]
        lista[i+2] = lista[i+6]
        lista[i+6] = temp
    if lista[i+3] > lista[i+7]:
        temp = lista[i+3]
        lista[i+3] = lista[i+7]
        lista[i+7] = temp
#2
for i in range(0, 16, 4):
    if lista[i] > lista[i+2]:
        temp = lista[i]
        lista[i] = lista[i+2]
        lista[i+2] = temp
    if lista[i+1] > lista[i+3]:
        temp = lista[i+1]
        lista[i+1] = lista[i+3]
        lista[i+3] = temp
#1
for i in range(0, 16, 2):
    if lista[i] > lista[i+1]:
        temp = lista[i]
        lista[i] = lista[i+1]
        lista[i+1] = temp

for i in range(0, 15):
    if lista[i] > lista[i+1]:
        print("Error")
        break


tamanho_da_lista = len(lista)
nr_iteracoes = int(math.log2(tamanho_da_lista))
lista = copia_lista

print(lista)

for i in range(0, nr_iteracoes):
    for j in range(i+1, 0, -1):
        for k in range(0, tamanho_da_lista//2**j):
            if (k * 2**(j-1)) // 2**i % 2 == 0:
                for l in range(0, 2**(j-1)):
                    if lista[k * 2**j + l] > lista[k * 2**j + l + 2**(j-1)]:
                        temp = lista[k * 2**j + l]
                        lista[k * 2**j + l] = lista[k * 2**j + l + 2**(j-1)]
                        lista[k * 2**j + l + 2**(j-1)] = temp
            else:
                for l in range(0, 2**(j-1)):
                    if lista[k * 2**j + l] < lista[k * 2**j + l + 2**(j-1)]:
                        temp = lista[k * 2**j + l]
                        lista[k * 2**j + l] = lista[k * 2**j + l + 2**(j-1)]
                        lista[k * 2**j + l + 2**(j-1)] = temp

for i in range(0, 15):
    if lista[i] > lista[i+1]:
        print("Error")
        print(lista)
        break
print(lista)