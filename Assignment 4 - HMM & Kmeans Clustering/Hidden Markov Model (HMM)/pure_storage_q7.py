# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 18:22:19 2020

@author: Paavan Patel
"""
def digit_score(num):
    score=0
    if(num % 3==0):
        score +=2
    while(num>0):
        digit=num%10
        if digit == 7:
            score +=1
        if digit%2 == 0:
            score +=4
        num=int(num / 10)
    return score

def consecutive_fives(num):
    score=0
    current=0
    while(num>0):
        digit=num%10
        num=int(num/10)
        print(digit,",",)
        if digit!=5 and current>1:
            score+=(current-1)*3
            score+=(current-2)*3
            current=0
        if digit!=5 and current == 1:
            current=0
        if digit == 5:
            current+=1
    if current>1:
        score+=(current-1)*3
        score+=(current-2)*3
    return score

def sequence_score(num):
    count=1
    i=0
    sc_arr=[]
    snum=str(num)
    while(i<len(snum)-1):
        if int(snum[i])==int(snum[i+1])+1:
            count+=1
            i+=1
        else:
            i+=1
            sc_arr.append(count)
            count=1
    sc_arr.append(count)
    score=0
    print(sc_arr)
    for x in sc_arr:
        score += x*x
    return score
        
    
def computer_score(number):
    return digit_score(number)+consecutive_fives(number)+sequence_score(number)    

#print(consecutive_fives(55115555412))
print(sequence_score(9765320))
