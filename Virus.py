#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random


# In[2]:


g=nx.Graph()

with open("static.network") as file:
    data = file.read().split('\n')
    #print(data[0])
    for i in range(1,len(data)):
        nodes=data[i].split(' ')
        if len(nodes)==2:
            g.add_edge(int(nodes[0]),int(nodes[1]))
        
#print(nx.number_of_nodes(g))
total_node=nx.number_of_nodes(g)
adj=[[0 for _ in range(total_node)] for _ in range(total_node)]
#print (len(nx.edges(g)))
for i in nx.edges(g):
    adj[i[0]][i[1]]=1
    adj[i[1]][i[0]]=1
eigenvalue, eigenvector = np.linalg.eig(adj)
#print(eigenvalue)
#print(eigenvector)


# In[3]:


eig_set=[(eigenvalue[i],eigenvector[i]) for i in range(len(eigenvalue))]
eig_set=sorted(eig_set,key=lambda x:x[0],reverse=1)


# In[4]:


print(eig_set)


# In[5]:


beta1=0.2
beta2=0.01
delta1=0.7
delta2=0.6
Cvpm1=beta1/delta1
Cvpm2=beta2/delta2


# In[6]:


largest_eigenvalue=eig_set[0][0].real

#print(largest_eigenvalue)
s1=largest_eigenvalue*Cvpm1
print(s1)
s2=largest_eigenvalue*Cvpm2
print(s2)


# In[7]:


static_delta1=[]
flag=0
for i in range(1,1001):
    tmp_beta=float(i)/1000
    #print(tmp_beta)
    tmp_Cvpm=tmp_beta/delta1
    current=largest_eigenvalue*tmp_Cvpm
    if current>1 and flag==0:
        print (tmp_beta)
        flag=1
    static_delta1.append(current)
    #print(largest_eigenvalue,tmp_Cvpm,current)
#print(static_delta)
threshold = np.array([1 for i in range(1000)])
plt.plot(threshold, 'r--') 
plt.plot(static_delta1)

plt.show()


# In[8]:


static_beta1=[]
flag=0
for i in range(1,1001):
    tmp_delta=float(i)/1000
    tmp_Cvpm=beta1/tmp_delta
    current=largest_eigenvalue*tmp_Cvpm
    if current<1 and flag==0:
        print (tmp_delta)
        flag=1
    static_beta1.append(current)
#print(static_beta1)
threshold = np.array([1 for i in range(1000)])
plt.plot(threshold, 'r--') 
plt.plot(static_beta1)
plt.show()


# In[9]:


static_delta2=[]
flag=0
for i in range(1,1001):
    tmp_beta=float(i)/1000
    tmp_Cvpm=tmp_beta/delta2
    current=largest_eigenvalue*tmp_Cvpm
    if current>1 and flag==0:
        print (tmp_beta)
        flag=1
    static_delta2.append(current)
threshold = np.array([1 for i in range(1000)])
plt.plot(threshold, 'r--') 
plt.plot(static_delta2)

plt.show()


# In[10]:


static_beta2=[]
flag=0
for i in range(1,1001):
    tmp_delta=float(i)/1000
    
    tmp_Cvpm=beta2/tmp_delta
    current=largest_eigenvalue*tmp_Cvpm
    #print(current,tmp_delta)
    if current<1 and flag==0:
        print (tmp_delta)
        flag=1
    static_beta2.append(current)

threshold = np.array([1 for i in range(1000)])
plt.plot(threshold, 'r--') 
plt.plot(static_beta2)
plt.show()


# In[11]:


c=int(total_node/10)
t=100
def simulate(beta,delta,adj,t,c):
    infect=[False for _ in range(total_node)]
    infect_ind=[]
    while len(infect_ind)<c:
        r=random.randint(0, total_node-1)
        if infect[r]==False:
            infect[r]=True
            infect_ind.append(r)

    infect_num=[]
    infect_num.append(len(infect_ind))

    for i in range(t):
        cur_infect=set()
        cur_cure=[]
        for inf in infect_ind:
            #print(inf)
            for j in range(len(adj[inf])):
                if adj[inf][j]==1 and float(random.randint(1, 10))/10<beta:
                    cur_infect.add(j)
            if float(random.randint(1, 10))/10<delta:
                cur_cure.append(inf)
            else:
                cur_infect.add(inf)
        
        for node in cur_cure:
            infect[node]=False
        for node in cur_infect:
            infect[node]=True
        infect_num.append(len(cur_infect))
        infect_ind=cur_infect
        #print(i,len(infect_ind))
    return infect_num
                


# In[12]:



first_series=[]
for i in range(10):
    res=simulate(beta1,delta1,adj,t,c)
    first_series.append(res)
first_series=np.array(first_series)
first_series=np.mean(first_series, axis=0)
#print(first_series)
plt.plot(first_series)
plt.show()


# In[13]:


second_series=[]
for i in range(10):
    res=simulate(beta2,delta2,adj,t,c)
    second_series.append(res)
second_series=np.array(second_series)
second_series=np.mean(second_series, axis=0)
#print(second_series)
plt.plot(second_series)
plt.show()


# In[14]:


def policyA(beta,delta,adj,t,c,k):
    immun=set()
    while len(immun)<k:
        r=random.randint(0, total_node-1)
        if r not in immun:
            immun.add(r)
    
    
    infect=[False for _ in range(total_node)]
    infect_ind=[]
    while len(infect_ind)<c:
        r=random.randint(0, total_node-1)
        if infect[r]==False and r not in immun:
            infect[r]=True
            infect_ind.append(r)

    infect_num=[]
    infect_num.append(len(infect_ind))
    #print(set(infect_ind).intersection(immun))
    #print(immun)
    for i in range(t):
        cur_infect=set()
        cur_cure=[]
        #print(set(infect_ind).intersection(immun))
        for inf in infect_ind:
            #print(inf)
            for j in range(len(adj[inf])):
                if adj[inf][j]==1 and float(random.randint(1, 10))/10<beta and j not in immun:
                    cur_infect.add(j)
            if float(random.randint(1, 10))/10<delta:
                cur_cure.append(inf)
            else:
                cur_infect.add(inf)
        
        for node in cur_cure:
            infect[node]=False
        for node in cur_infect:
            infect[node]=True
        infect_num.append(len(cur_infect))
        infect_ind=cur_infect
        #print(i,len(infect_ind))
    return infect_num
    
    


# In[28]:


adjA=[[0 for _ in range(total_node)] for _ in range(total_node)]
for i in nx.edges(g):
    adjA[i[0]][i[1]]=1
    adjA[i[1]][i[0]]=1
    
immun=set()
while len(immun)<200:
    r=random.randint(0, total_node-1)
    if r not in immun:
        immun.add(r)
for node in immun:    
    for i in range(len(adjA[node])):
        adjA[node][i]=0
        adjA[i][node]=0
        
eigenvalueA, eigenvectorA = np.linalg.eig(adjA)
eig_setA=[(eigenvalueA[i],eigenvectorA[i]) for i in range(len(eigenvalueA))]
eig_setA=sorted(eig_setA,key=lambda x:x[0],reverse=1)
largest_eigenvalueA=eig_setA[0][0].real
sA=largest_eigenvalueA*Cvpm1
print(sA)


# In[ ]:



def immunA(k,adj):
    immun=set()
    while len(immun)<k:
        r=random.randint(0, total_node-1)
        if r not in immun:
            immun.add(r)
    
    for node in immun:    
        for i in range(len(adj[node])):
            adj[node][i]=0
            adj[i][node]=0
    return adj

static_beta_delta_A=[]
flag=0
for i in range(1,1001):
    new_adj=immunA(i,adj)
    val, vec = np.linalg.eig(new_adj)
    e=[(val[i],vec[i]) for i in range(len(val))]
    e=sorted(e,key=lambda x:x[0],reverse=1)
    le=e[0][0].real
    strength=le*Cvpm1
    if strength<1 and flag==0:
        print (i)
        flag=1
    static_beta_delta_A.append(strength)

threshold = np.array([1 for i in range(1000)])
plt.plot(threshold, 'r--') 
plt.plot(static_beta_delta_A)
plt.show()


# In[15]:


policy_A=[]
for i in range(10):
    res=policyA(beta1,delta1,adj,t,c,200)
    policy_A.append(res)
policy_A=np.array(policy_A)
policy_A=np.mean(policy_A, axis=0)
#print(policy_A)
plt.plot(policy_A)
plt.show()


# In[16]:


def policyB(beta,delta,adj,t,c,k,g):
    immun=set()
    degree=sorted(list(g.degree()),key=lambda x:x[1],reverse=1)
    for i in range(k):
        immun.add(degree[i][0])
        
    
    
    infect=[False for _ in range(total_node)]
    infect_ind=[]
    while len(infect_ind)<c:
        r=random.randint(0, total_node-1)
        if infect[r]==False and r not in immun:
            infect[r]=True
            infect_ind.append(r)

    infect_num=[]
    infect_num.append(len(infect_ind))
    #print(set(infect_ind).intersection(immun))
    #print(immun)
    for i in range(t):
        cur_infect=set()
        cur_cure=[]
        #print(set(infect_ind).intersection(immun))
        for inf in infect_ind:
            #print(inf)
            for j in range(len(adj[inf])):
                if adj[inf][j]==1 and float(random.randint(1, 10))/10<beta and j not in immun:
                    cur_infect.add(j)
            if float(random.randint(1, 10))/10<delta:
                cur_cure.append(inf)
            else:
                cur_infect.add(inf)
        
        for node in cur_cure:
            infect[node]=False
        for node in cur_infect:
            infect[node]=True
        infect_num.append(len(cur_infect))
        infect_ind=cur_infect
        #print(i,len(infect_ind))
    return infect_num
    
    


# In[17]:


policy_B=[]
for i in range(10):
    res=policyB(beta1,delta1,adj,t,c,200,g)
    policy_B.append(res)
policy_B=np.array(policy_B)
policy_B=np.mean(policy_B, axis=0)
#print(policy_A)
plt.plot(policy_B)
plt.show()


# In[31]:


def policyC(beta,delta,adj,t,c,k,g):
    immun=set()
    while len(immun)<k:
        degree=sorted(list(g.degree()),key=lambda x:x[1],reverse=1)
        immun.add(degree[0][0])
        g.remove_node(degree[0][0])
        
    
    
    infect=[False for _ in range(total_node)]
    infect_ind=[]
    while len(infect_ind)<c:
        r=random.randint(0, total_node-1)
        if infect[r]==False and r not in immun:
            infect[r]=True
            infect_ind.append(r)

    infect_num=[]
    infect_num.append(len(infect_ind))
    #print(set(infect_ind).intersection(immun))
    #print(immun)
    for i in range(t):
        cur_infect=set()
        cur_cure=[]
        #print(set(infect_ind).intersection(immun))
        for inf in infect_ind:
            #print(inf)
            for j in range(len(adj[inf])):
                if adj[inf][j]==1 and float(random.randint(1, 10))/10<beta and j not in immun:
                    cur_infect.add(j)
            if float(random.randint(1, 10))/10<delta:
                cur_cure.append(inf)
            else:
                cur_infect.add(inf)
        
        for node in cur_cure:
            infect[node]=False
        for node in cur_infect:
            infect[node]=True
        infect_num.append(len(cur_infect))
        infect_ind=cur_infect
        #print(i,len(infect_ind))
    return infect_num
    


# In[32]:


policy_C=[]
for i in range(10):
    res=policyC(beta1,delta1,adj,t,c,200,g)
    policy_C.append(res)
policy_C=np.array(policy_C)
policy_C=np.mean(policy_C, axis=0)
#print(policy_A)
plt.plot(policy_C)
plt.show()


# In[ ]:




