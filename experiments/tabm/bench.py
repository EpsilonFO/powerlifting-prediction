import sys, time; sys.path.insert(0, "tabm")
import torch; torch.set_num_threads(2)
from tabm_model import TabMNet
cards=[5, 7, 19, 9, 95, 4, 115, 92, 312, 22, 109, 193, 4, 13, 12]
n_num=60; cont=list(range(38))
N=84000
X=torch.cat([torch.stack([torch.randint(0,c,(N,)) for c in cards],1).float(), torch.randn(N,n_num)],1)
y=torch.randn(N)
for dev in ["mps"]:
  for cat in ["onehot"]:
    torch.manual_seed(0)
    net=TabMNet(cards,n_num,cont,X[:,15:],cat=cat).to(dev)
    opt=torch.optim.AdamW(net.parameters(),2e-3)
    Xd,yd=X.to(dev),y.to(dev)
    nsteps=30; bs=512
    for j in range(nsteps+3):
        if j==3:
            if dev=="mps": torch.mps.synchronize()
            t=time.time()
        idx=torch.randint(0,N,(bs,),device=dev)
        loss=((net(Xd[idx])-yd[idx,None])**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    if dev=="mps": torch.mps.synchronize()
    dt=(time.time()-t)/nsteps
    print(dev,cat,f"{dt*1000:.1f} ms/step -> {dt*N/bs:.1f} s/epoch (bs {bs})",flush=True)
