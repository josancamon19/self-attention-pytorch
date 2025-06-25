import torch

torch.manual_seed(42)

# - x.view(-1) vs x.flatten(), .view() needs contiguous, .reshape(), stride
# - broadcasting scenarios, multidimensional slicing `[::, :1,:]`
# - element wise operations, in place operations
# - torch.Parameter, basics of existing blocks, dropout, Linear

# x = torch.randn((3, 2))
# y = torch.randn((3, 2))
# print(x.shape)
# print(y.shape)

# print(x[0][0] * y[0][0])
# print(x * y)
# print(x @ y.T)

# x = torch.randn((2, 3, 4))
# print(x.stride())  # 4*3,4,1
# stride: # elements you need to skip in memory to move to the next in each dimension

# x = torch.randn((2, 3))
# print(x)
# print(torch.sum(x))
# print(torch.sum(x, dim=0))

# confusing part about dim=0, if rows, cols, is that when summing through rows, you are not saying between a row, but from 0 to n rows together
# operations at row level, dim=1, would mean column, got it

# print(torch.sum(x, dim=1))
# print("x[:, -1]:", x[:, 2])


# ========== PYTORCH TENSOR MASTERY TASKS ==========

# Task 1: Memory sharing and views
# a = torch.randn(3, 4)
# Task: Create a view of 'a' with shape (2, 6), modify one element in the view, verify original changed
# b = a.view(2, 6)
# b[0,0] = -1.0909
# print(a)


# Task 2: Contiguous vs non-contiguous
# b = torch.randn(3, 4)
# print(b.is_contiguous(), b.stride())
# b = b.transpose(0, 1)
# Task: Check if 'b' is contiguous, try .view(-1) vs .flatten(), explain the difference
# print(b.is_contiguous(), b.stride())
# print(b.flatten())
# when .contiguous() is called, a copy is made, if initial copy not used anymore, garbage collected
# but also takes time as it has to copy to a new memory space, so for a second it occupies more memory
# this is a physical copy, not too optimizable, .contiguous is safe to call if it already
# print(b.contiguous().view(-1))

# Task 3: Advanced slicing and indexing
# c = torch.randn(4, 5, 6)
# Task: Extract every 2nd element along dim 0, first 3 elements along dim 1, last 2 along dim 2
# print(c)
# print(c[::2, :3, -2:])

# Task 4: Broadcasting mastery
# d = torch.randn(3, 1, 4)
# e = torch.randn(1, 5, 1)
# Task: Add d + e, predict the output shape before running, verify your prediction
# dimensions are compatible if they're equal or one of them is 1. The 1s get "stretched" to match the larger dimension.
# print((d + e).shape)  # output shape: 3,5,4? takes max at every dim

# Task 5: In-place operations and memory
# f = torch.randn(3, 3)
# g = f.view(9)
# Task: Use .add_() on g, observe what happens to f, explain why
# g.add_(3)
# print(g)
# print(f)
# f get's modified as well, as g is a view of f

# Task 6: Dimension manipulation
# h = torch.randn(2, 1, 3, 1, 4)
# Task: Remove all dimensions of size 1, then add them back in different positions
# print(h.squeeze().shape)
# print(h.squeeze().unsqueeze(0).shape)

# Task 7: Advanced aggregations
# i = torch.randn(3, 4, 5)
# Task: Compute mean along multiple dimensions [0, 2], keep dimensions vs remove them
# print(torch.mean(i, dim=0, keepdim=True).shape)
# print(torch.mean(i, dim=0, keepdim=False).shape)
# print(torch.mean(i, dim=2, keepdim=False).shape)
# TODO: visualize this in more detail, at each dimension how does it look like
# initialize with ints that are trackable

# Task 8: Tensor concatenation and stacking
# j = torch.randn(3, 3)
# k = torch.randn(3, 3)
# Task: Concatenate along dim 0, dim 1, then stack to create a new dimension
# print(j)
# print(k)
# dim to stack at
# TODO: create flashcards for this
# print(torch.stack([j, k], dim=0).shape)
# print(torch.stack([j, k], dim=1).shape)
# print(torch.stack([j, k], dim=2).shape)
# print(torch.vstack([j, k]).shape)

# print("---")

# j = torch.randn(2, 3)
# k = torch.randn(2, 3)
# print(torch.vstack([j, k]).shape)
# print(torch.hstack((j, k)).shape)
# print(torch.cat((j, k), dim=0).shape)
# print(torch.cat((j, k), dim=1).shape)


#   torch.cat() - Most flexible, concatenates along existing dimension:
#   - dim=0: vertical stacking (more rows)
#   - dim=1: horizontal stacking (more columns)
#   - Use when you want to join tensors along an existing axis

#   torch.stack() - Creates a new dimension:
#   - Always increases number of dimensions by 1
#   - dim=0: new batch dimension at front
#   - dim=1: new dimension in middle
#   - dim=2: new dimension at end
#   - Use when you want to batch tensors together

#   torch.vstack()/hstack() - Convenience functions:
#   - vstack() = cat(dim=0) (vertical)
#   - hstack() = cat(dim=1) (horizontal)
#   - Use for simple 2D matrix operations when meaning is clear

#   Rule of thumb:
#   - Concatenating: Use cat()
#   - Batching: Use stack()
#   - Simple 2D: Use vstack()/hstack()


# Task 9: Reshaping with -1 inference
# l = torch.randn(2, 3, 4, 5)
# Task: Reshape to (6, -1), then (-1, 10), predict shapes before running
# print(l.reshape(6, -1).shape)
# print(l.reshape(-1, 10).shape)
# print(l.reshape(-1, 15).shape)
# TODO: how are 10 items chosen for each, in what order?

# Task 10: Memory layout and performance
# m = torch.randn(50000, 30000)
# n = m.transpose(0, 1)
# Task: Compare performance of summing m vs n, explain why there's a difference
# import time  # noqa: E402

# start = time.time()
# print(torch.sum(m))
# print(time.time() - start)
# start = time.time()
# print(torch.sum(n))
# print(time.time() - start)
