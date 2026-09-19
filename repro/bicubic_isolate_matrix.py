import torch, torch.nn as nn, itertools

def build(mode, antialias):
    def f(x, size):
        return nn.functional.interpolate(x, size=(size, size), mode=mode, align_corners=False, antialias=antialias)
    return f

print(f"{'mode':9s} {'antialias':9s} {'grad':5s} {'dynamic':7s}  result")
for mode, antialias, grad, dynamic in itertools.product(("bicubic","bilinear"), (True,False), (True,False), (True,False)):
    x = torch.randn(1, 8, 16, 16, device="cuda", requires_grad=grad)
    f = torch.compile(build(mode, antialias), dynamic=dynamic)
    try:
        out = f(x, 21)
        if grad: out.sum().backward()
        status = "ok"
    except Exception as e:
        status = f"RAISED {type(e).__name__}"
    print(f"{mode:9s} {str(antialias):9s} {str(grad):5s} {str(dynamic):7s}  {status}")
