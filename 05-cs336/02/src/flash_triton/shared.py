import triton


def get_cuda_autotune_config():
    def config_item(q, k, ns, nw, mr=None):
        config = triton.Config({"BLOCK_SIZE_Q": q, "BLOCK_SIZE_K": k}, num_stages=ns, num_warps=nw)
        if mr is not None:
            config.maxnreg = mr
        return config

    return [
        config_item(128, 256, 3, 8),
        config_item(64, 256, 4, 4),
        config_item(64, 64, 3, 4),
        config_item(128, 128, 4, 4),
        config_item(128, 64, 4, 4),
        config_item(64, 128, 4, 4),
        config_item(128, 32, 4, 4),
        config_item(64, 32, 5, 2),
        config_item(32, 64, 5, 2),
        config_item(128, 64, 5, 8),
        config_item(256, 64, 3, 8),
        config_item(256, 64, 5, 8),
        config_item(256, 64, 6, 8),
    ]
