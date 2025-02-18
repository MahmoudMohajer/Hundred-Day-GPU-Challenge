import triton 
import triton.language as tl 
import torch 


@triton.jit 
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask)
    y = tl.load(y_ptr + offsets, mask)

    output = x + y 

    tl.store(output_ptr + offsets, output, mask=mask)

def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.numel() == y.numel()
    n_elements = x.numel()

    output = torch.empty_like(x)

    BLOCK_SIZE = 1024

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output

if __name__ == "__main__":
    x = torch.randn(10000, device='cuda', dtype=torch.float32)
    y = torch.randn(10000, device='cuda', dtype=torch.float32)

    z = vector_add(x, y)

    print("Result close to torch.add?", torch.allclose(z, x+y))