
@timeit
def get_train_datasets(args, paraphrase_dataset:bool = True, is_distributed: bool = False, rank: int = None):
    para_train_data = load_paraphrase_data(args.para_train)
    para_dev_data = load_paraphrase_data(args.para_dev)

    if paraphrase_dataset:
        para_train_data = ParaphraseDetectionDataset(para_train_data, args)
        para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
    else:
        sonnet_dataset = SonnetsDataset(args.sonnet_path)
        
    
    if is_distributed:
        if rank == 0: # print only once
            world_size = torch.cuda.device_count()
            print(f"Distributed: {len(para_train_data)} total samples")
            print(f"Distributed: ~{len(para_train_data)//world_size} samples per GPU")
            print(f"Distributed: ~{(len(para_train_data)//world_size)//args.batch_size} batches per GPU")
            # print(f"Distributed: Global effective batch size: {args.batch_size * world_size}")
    else:
        print(f"Single GPU: {len(para_train_data)} samples")
        print(f"Single GPU: {len(para_train_data)//args.batch_size} batches total")


    # handle multiple processes
    dev_sampler = DistributedSampler(para_dev_data, shuffle=False) if is_distributed else None
    train_sampler = DistributedSampler(para_train_data, shuffle=True) if is_distributed else None
    # dataloader, grain python.

    para_train_dataloader = DataLoader(
        para_train_data,
        shuffle=(train_sampler is None), # TODO: why?
        batch_size=args.batch_size,
        collate_fn=para_train_data.collate_fn,
        sampler=train_sampler # TODO: what's this for
    )
    para_dev_dataloader = DataLoader(
        para_dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=para_dev_data.collate_fn,
        sampler=dev_sampler,
    )
    return para_train_dataloader, para_dev_dataloader



def get_model_and_optimizer(args, device):
    model = ParaphraseGPT(args)
    model = model.to(device)
    if args.peft:
        model = get_peft_model(model, _get_lora_config(False))
        model.print_trainable_parameters()

    # if args.use_bf16: # no autocast
    #     model = model.to(torch.bfloat16)
    #     print("get_model_and_optimizer: model moved to bf16")
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    return model, optimizer


def train_epoch(
    model,
    epoch,
    device,
    optimizer,
    para_train_dataloader,
    para_dev_dataloader,
    best_dev_acc,
    rank = None,
    train_sampler = None,
    gradient_accumulation_steps = 1,
    use_bf16 = False,
):  
    # is necessary to make shuffling work properly across multiple epochs. 
    # # Otherwise, the same ordering will be used in each epoch.
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    
    model.train()
    train_loss = 0
    num_batches = 0
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(tqdm(
        para_train_dataloader, desc=f"train-{epoch}", disable=TQDM_DISABLE
    )):
        # Get the input and move it to the gpu (I do not recommend training this model on CPU).
        b_ids, b_mask, labels = (
            batch["token_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["labels"].flatten().to(device),
        )
        
        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(b_ids, b_mask)
            loss = F.cross_entropy(logits, labels, reduction="mean")
            loss = loss / gradient_accumulation_steps
        
        loss.backward()
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

    if num_batches % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()   
    
    train_loss = train_loss / num_batches
    
    # TODO: docs
    if rank is not None:
        train_loss_tensor = torch.tensor(train_loss).cuda()
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        train_loss = train_loss_tensor.item() / dist.get_world_size()

    if rank is None or rank == 0:
        dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            model_to_save = model.module if hasattr(model, 'module') else model
            save_model(model_to_save, optimizer, args)
        print(f"Epoch {epoch}: train loss :: {train_loss:.3f}, dev acc :: {dev_acc:.3f}")
        
    if rank is not None:
        best_dev_acc_tensor = torch.tensor(best_dev_acc).cuda()
        dist.broadcast(best_dev_acc_tensor, src=0)
        best_dev_acc = best_dev_acc_tensor.item()
        
    return best_dev_acc


def train(args):
    """Train GPT-2 for paraphrase detection on the Quora dataset."""

    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    para_train_dataloader, para_dev_dataloader = get_train_datasets(args)
    model, optimizer = get_model_and_optimizer(args, device)
    best_dev_acc = 0

    for epoch in range(args.epochs):
        best_dev_acc = train_epoch(
            model,
            epoch,
            device,
            optimizer,
            para_train_dataloader,
            para_dev_dataloader,
            best_dev_acc,
            None,
            None,
            1,
            args.use_bf16
        )


def train_dist(rank, args):
    try:
        world_size = torch.cuda.device_count()
        dist.init_process_group(
            "nccl", # gloo, never used, test things on cpu
            init_method="tcp://localhost:12355",
            # rendevouz, server python spins up, distribute works to other processes
            rank=rank,
            world_size=world_size
        )
        torch.cuda.set_device(rank)
        para_train_dataloader, para_dev_dataloader = get_train_datasets(args, True, rank)
        device = torch.device(f'cuda:{rank}')
        
        model, optimizer = get_model_and_optimizer(args, device)
        # tries to schedule things ahead of time, mark operations ready to go, but it some vars never receive gradients
        # it will keep waiting for that. (find_unused_parameters = True)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

        best_dev_acc = 0

        for epoch in range(args.epochs):
            best_dev_acc = train_epoch(
                model,
                epoch,
                device,
                optimizer,
                para_train_dataloader,
                para_dev_dataloader,
                best_dev_acc,
                rank,
                para_train_dataloader.sampler,
                3,
                args.use_bf16
            )
            dist.barrier()
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
    parser.add_argument(
        "--para_dev_out", type=str, default="predictions/para-dev-output.csv"
    )
    parser.add_argument(
        "--para_test_out", type=str, default="predictions/para-test-output.csv"
    )

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--peft", action="store_true")
    parser.add_argument("--distributed", action="store_true")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument(
        "--model_size",
        type=str,
        help="The model size as specified on hugging face. DO NOT use the xl model.",
        choices=["gpt2", "gpt2-medium", "gpt2-large"],
        default="gpt2",
    )

    args = parser.parse_args()
    return args


def add_arguments(args):
    """Add arguments that are deterministic on model size."""
    if args.model_size == "gpt2":
        args.d = 768
        args.l = 12
        args.num_heads = 12
    elif args.model_size == "gpt2-medium":
        args.d = 1024
        args.l = 24
        args.num_heads = 16
    elif args.model_size == "gpt2-large":
        args.d = 1280
        args.l = 36
        args.num_heads = 20
    else:
        raise Exception(f"{args.model_size} is not supported.")
    return args


def check_bf16_support():
    if torch.cuda.is_available():
        # Check if GPU supports BF16
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:  # Ampere (RTX 30xx) and newer
            print("✅ BF16 supported on this GPU")
            return True
        else:
            print("❌ BF16 not supported on this GPU (need Ampere or newer)")
            return False