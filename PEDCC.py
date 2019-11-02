import numpy as np
import pickle
import torch
import os

class_num=10
out_dim=60
mu=np.zeros(out_dim)
sigma = np.eye(out_dim)

r_1 = np.random.multivariate_normal(mu,sigma,class_num)
v = np.zeros(r_1.shape)
r=[]
for i in r_1:
    i=i/np.linalg.norm(i)
    r.append(i)
r=np.array(r)
G=1e-2

def countnext(r,v,G):
    num = r.shape[0]
    dd = np.zeros((out_dim,num,num))
    for m in range(num):
        for n in range(num):
            dd[:,m,n] = (r[m,:]-r[n,:]).T
            dd[:,n,m] = -dd[:,m,n]
    L=np.sum(dd**2,0)**0.5
    L=L.reshape(1,L.shape[0],L.shape[1])
    L[L<1e-2] = 1e-2
    a=np.repeat(L**3,out_dim,0)
    F=np.sum(dd/a,2).T
    tmp_F=[]
    for i in range(F.shape[0]):
        tmp_F.append(np.dot(F[i],r[i]))
    d=np.array(tmp_F).T.reshape(len(tmp_F),1)
    Fr = r*np.repeat(d, out_dim, 1)
    Fv = F-Fr
    rn = r+v
    ll = np.sum(rn**2,1)**0.5
    rn=rn/np.repeat(ll.reshape(ll.shape[0],1),out_dim,1)
    vn = v+G*Fv
    return rn,vn

def generate_center(r,v,G):
    for i in range(200):
        rn,vn=countnext(r,v,G)
        r=rn
        v=vn
    # return r*(out_dim)**0.5
    return r

r1=generate_center(r,v,G)

f=open('./60d.pkl','wb')
pickle.dump(r1,f)
f.close()

ff=open("./60d.pkl",'rb')
b=pickle.load(ff)
ff.close()

os.remove("./60d.pkl")
# result=np.zeros((len(b),len(b)))
# for a in range(len(b)):
#     for aa in range(a,len(b)):
#         result[a][aa]=np.linalg.norm((b[a]-b[aa]))
#         result[aa][a] = result[a][aa]

fff=open('./'+str(class_num)+'_'+str(out_dim)+'.pkl','wb')
map={}
for i in range(len(b)):
    map[i]=torch.from_numpy(np.array([b[i]]))
pickle.dump(map,fff)
fff.close()







