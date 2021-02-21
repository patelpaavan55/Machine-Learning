# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:22:14 2020

@author: Paavan Patel
"""

def check(events):
    my_set=set()
    stck = []
    for i,event in enumerate(events):
        name=event.split(" ")[0]
        num=int(event.split(" ")[1])
        if name=="ACQUIRE":
            if num in my_set:
                return i+1
            my_set.add(num)
            stck.append(num)
        else:
            if num not in my_set or stck[-1]!=num:
                return i+1
            stck.pop()
            my_set.remove(num)
    if len(stck)==0:
        return 0
    else:
        return len(events)+1
    
test1=["ACQUIRE 364","ACQUIRE 84","RELEASE 84","RELEASE 364"]
test2=["ACQUIRE 364","ACQUIRE 84","RELEASE 364","RELEASE 84"]
test3=["ACQUIRE 123","ACQUIRE 364","ACQUIRE 84","RELEASE 84","RELEASE 364","ACQUIRE 456"]
test4=["ACQUIRE 123","ACQUIRE 364","ACQUIRE 84","RELEASE 84","RELEASE 364","ACQUIRE 789","RELEASE 456","RELEASE 123"]
test5=["ACQUIRE 123","ACQUIRE 364","ACQUIRE 84","RELEASE 84","RELEASE 364","ACQUIRE 456"]
print(check(test5))