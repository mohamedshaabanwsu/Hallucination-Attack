
import torch


example1 = torch.tensor([1.268554688, 1.188476563, 1.311523438, 1.103515625, 0.989746094])
example2 = torch.tensor([2.19343e-05, 0.000997999, 3.43317e-05, 3.09939e-05, 2.61065e-05])

'''example1 = torch.tensor([10, 15, 20, 100, 200, 300, 500, 800, 1100, 1500])
example2 = torch.tensor([0.00008, 0.00009, 0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.0008, 0.00088, 0.0009])'''
# For lists/arrays/tensors:

def min_max_normalize(tensor):
    # tensor: PyTorch 1D tensor
    min_val = tensor.min()
    max_val = tensor.max()
    print("Min:", min_val)
    print("Max:", max_val)
    if max_val == min_val:
        return torch.zeros_like(tensor)  # handle edge case
    return (tensor - min_val) / (max_val - min_val)

llm_norm = min_max_normalize(example1)
mind_norm = min_max_normalize(example2)

print("LLM Norm:", llm_norm)
print("MIND Norm:", mind_norm)

#add and print values and min
combined = llm_norm + mind_norm
print("Combined:", combined)

print("Combined Min:", combined.min())
print("Combined Max:", combined.max())
 