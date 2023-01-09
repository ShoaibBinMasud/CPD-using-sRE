# code adapted from https://github.com/jeanfeydy/global-divergences 
import torch

def scal(a, f):
    return torch.dot(a.view(-1), f.view(-1))


def lse(v_ij):
    """[lse(v_ij)]_i = log sum_j exp(v_ij), with numerical accuracy."""
    V_i = torch.max(v_ij, 1)[0].view(-1, 1)
    return V_i + (v_ij - V_i).exp().sum(1).log().view(-1, 1)


def ave(u, u1, tau):
    "Barycenter subroutine, used by kinetic acceleration through extrapolation."
    return tau * u + (1 - tau) * u1


def Sinkhorn_ops(p, eps, x_i, y_j):
    """
    Given:
    - an exponent p = 1 or 2
    - a regularization strength ε > 0
    - point clouds x_i and y_j, encoded as N-by-D and M-by-D torch arrays,
    Returns a pair of routines S_x, S_y such that
      [S_x(f_i)]_j = -log sum_i exp( f_i - |x_i-y_j|^p / ε )
      [S_y(f_j)]_i = -log sum_j exp( f_j - |x_i-y_j|^p / ε )
    """
    # We precompute the |x_i-y_j|^p matrix once and for all...

    # This needs to be modified according to the problem: Perhaps give the precomputed matrix as an input?
    # for example when we also use the feature maps or define in a feature domain

    x_y = x_i.unsqueeze(1) - y_j.unsqueeze(0)

    if len(x_y.shape) == 2:
        if   p == 1: C_e = torch.abs(x_y) / eps
        elif p == 2: C_e = (x_y ** 2) / eps
        else: C_e = torch.abs(x_y)**(p/2) / eps
    elif len(x_y.shape) > 2:
        if   p == 1: C_e = x_y.norm(dim=2) / eps
        elif p == 2: C_e = (x_y ** 2).sum(2) / eps
        else: C_e = x_y.norm(dim=2)**(p/2) / eps

    CT_e = C_e.t()

    def S_x(f_i): return -lse(f_i.view(1, -1) - CT_e)  # don't use lambda functions: SA

    def S_y(f_j): return -lse(f_j.view(1, -1) - C_e)   # don't use lambda functions: SA

    return S_x, S_y

# Sinkhorn iterations

def sink(a_i, x_i, b_j, y_j, p=1, eps=.1, nits=5000, tol=1e-9, assume_convergence=True): 

    # ε = eps # Python supports Unicode. So fancy!
    if type(nits) in [list, tuple]:
        nits = nits[0]  # The user may give different limits for Sink and SymSink

    # Sinkhorn loop with A = a/eps , B = b/eps ....................................................
    
    a_i_log, b_j_log = a_i.log(), b_j.log()  # Precompute the logs of the measures' weights
    B_i, A_j = torch.zeros_like(a_i), torch.zeros_like(b_j)  # Sampled influence fields
    # if we assume convergence, we can skip all the "save computational history" stuff
    torch.set_grad_enabled(not assume_convergence)

    S_x, S_y = Sinkhorn_ops(p, eps, x_i, y_j)  # Softmin operators (divided by ε, as it's slightly cheaper...)
    for i in range(nits-1):
        # print(i)
        B_i_prev = B_i
        A_j = S_x(B_i.view(1, -1) + a_i_log.view(1, -1))   # a(y)/ε = Smin_ε,x~α [ C(x,y) - b(x) ]  / ε
        B_i = S_y(A_j.view(1, -1) + b_j_log.view(1, -1))   # b(x)/ε = Smin_ε,y~β [ C(x,y) - a(y) ]  / ε

        err = eps * (B_i - B_i_prev).abs().mean()  # Stopping criterion: L1 norm of the updates
        if err.item() < tol:
            break

    torch.set_grad_enabled(True)
    # One last step, which allows us to bypass PyTorch's backprop engine if required (as explained in the paper)
    if not assume_convergence:
        A_j = S_x(B_i.view(1, -1) + a_i_log.view(1, -1))
        B_i = S_y(A_j.view(1, -1) + b_j_log.view(1, -1))
    else : # Assume that we have converged, and can thus use the "exact" (and cheap!) gradient's formula
        S_x, _ = Sinkhorn_ops(p, eps, x_i.detach(), y_j)
        _, S_y = Sinkhorn_ops(p, eps, x_i, y_j.detach())
        A_j = S_x((B_i.view(1, -1) + a_i_log).detach())
        B_i = S_y((A_j.view(1, -1) + b_j_log).detach())

    a_y, b_x = eps*A_j.view(-1), eps*B_i.view(-1)
    return a_y, b_x


def plan(a_i, x_i, b_j, y_j, p=1, eps=.1):
    x_y = x_i.unsqueeze(1) - y_j.unsqueeze(0)

    if len(x_y.shape) == 2:
        if p == 1:
            C_e = torch.abs(x_y) / eps
        elif p == 2:
            C_e = (torch.abs(x_y) ** 2) / eps
        else:
            C_e = torch.abs(x_y) ** (p / 2) / eps
    elif len(x_y.shape) > 2:
        if p == 1:
            C_e = x_y.norm(dim=2) / eps
        elif p == 2:
            C_e = (x_y ** 2).sum(2) / eps
        else:
            C_e = x_y.norm(dim=2) ** (p / 2) / eps

    a_y, b_x = sink(a_i, x_i, b_j, y_j, p, eps)  # a_y = g; b_x = f -- the dual vectors for the OT_eps problem only
    a_ib_j = torch.mm(a_i.unsqueeze(1), (b_j.unsqueeze(0)))
    plan = torch.mul((torch.exp((b_x.unsqueeze(1) + a_y.unsqueeze(0) - C_e*eps)/eps)), a_ib_j)
    return plan
