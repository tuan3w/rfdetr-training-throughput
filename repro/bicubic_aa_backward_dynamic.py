import torch, torch.nn as nn

def pos_embed_resize(embed, height, width):
    # dinov2_with_windowed_attn.interpolate_pos_encoding, lines 393-401
    return nn.functional.interpolate(
        embed, size=(int(height) // 16, int(width) // 16),
        mode="bicubic", align_corners=False, antialias=True,
    )

embed = torch.randn(1, 384, 36, 36, device="cuda", requires_grad=True)

def step(fn, res):
    out = fn(embed, res, res)
    out.sum().backward()
    return out.shape

print("eager           :", step(pos_embed_resize, 672))
compiled = torch.compile(pos_embed_resize, dynamic=True)
try:
    print("compile(dynamic):", step(compiled, 672))
except Exception as e:
    print("compile(dynamic): RAISED", type(e).__name__, str(e).split(chr(10))[0][:120])
compiled_static = torch.compile(pos_embed_resize, dynamic=False)
try:
    print("compile(static) :", step(compiled_static, 672))
except Exception as e:
    print("compile(static) : RAISED", type(e).__name__, str(e).split(chr(10))[0][:120])
