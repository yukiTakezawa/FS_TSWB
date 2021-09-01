import torch
from torch import nn
from tqdm import tqdm


class SinkhornLayer(nn.Module):
    """
    batch version
    """
    def __init__(self, epsilon):
        super(SinkhornLayer,self).__init__()
        self.epsilon = epsilon


    def forward(self, C, a, b, L, tol=1e-9, device=0):
        K = torch.exp(-C/self.epsilon)
        u = torch.ones((b.shape[0], a.shape[0])).cuda(device)
        v = torch.ones_like(b).cuda(device)
        l = 0
        change = torch.tensor(100.0)

        a2 = a.repeat(b.shape[0], 1)
        
        for l in tqdm(range(L)):# and (change > tol):
            old_u = u
            old_v = v
            
            #u = a / torch.mv(K,v)
            u = a2 / torch.mm(K, v.T).T
            v = b / torch.mm(torch.t(K), u.T).T
            l += 1

            #print("u", u.shape)
            #print("old_u", old_u.shape)
            #print(u - old_u)
            #change = max(torch.max(torch.norm(u - old_u, dim=1) / torch.norm(u, dim=1)), torch.max(torch.norm(v - old_v, dim=1) / torch.norm(v, dim=1)))
            #print(max(torch.norm(u - old_u, dim=1) / torch.norm(u, dim=1)))
            #change = max(max(torch.norm(u - old_u, dim=1) / torch.norm(u, dim=1)), max(torch.norm(v - old_v, dim=1) / torch.norm(u, dim=1)))
        """
        K2 = K.unsqueeze(0).repeat(b.shape[0], 1, 1)
        return ((u.view(b.shape[0], -1, 1) * (K2 * v.view(b.shape[0], 1, -1)))*C).sum(1).sum(1)
        """
        distance = []
        for i in tqdm(range(b.shape[0])):
            distance.append((u[i].view(-1, 1) * (K * v[i].view(1, -1)) * C).sum().item())
        return distance

if __name__ == "__main__":
    C = torch.tensor([
        [0., 1., 2.],
        [1., 0., 2.],
        [1., 2., 0.],
        [0., 2., 1.],
#        [2., 4., 4.],
    ], requires_grad=True).cuda()
    #a = torch.ones(3)/3.
    #b = torch.ones(3)/3.
    a = torch.tensor([0.1, 0.2, 0.3, 0.4])
    b = torch.tensor([[0.0, 0.5, 0.5],
                      [0.1, 0.2, 0.7],
                      [0.0, 0.2, 0.8],
                      [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]])
    
    print(a)
    print(b)
    a = a.cuda()
    b = b.cuda()
    ot = SinkhornLayer(0.05)
    distance = ot(C, a, b, 100)
    print(distance)

    import ot
    for i in range(b.shape[0]):
        print(ot.sinkhorn2(a.cpu().detach().numpy(), b[i].cpu().detach().numpy(), C.cpu().detach().numpy(), 0.01))
