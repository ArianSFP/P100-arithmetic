#!/usr/bin/env python3
import json

def tree(xs):
 xs=list(xs);o=len(xs)//2
 while o:
  for i in range(o):xs[i]=(xs[i],xs[i+o])
  o//=2
 return xs[0]

cases=0
for s in (8,16,32):
 for p in (2,4,8):
  partitioned=[tree(list(range(s))[i::p]) for i in range(p)]
  assert tree(partitioned)==tree(range(s))
  for groups in (1,3,8,17,32,136,160,544):
   original=[[g for g in range(stripe,groups,s)] for stripe in range(s)]
   seen=[]
   for part in range(p):
    for local in range(s//p):
     actual=list(range(local*p+part,groups,s))
     assert actual==original[local*p+part]
     seen+=actual
   assert sorted(seen)==list(range(groups))
  cases+=1
print(json.dumps(dict(status='PASS',exact_reduction_tree_cases=cases,group_coverages=cases*8)))
