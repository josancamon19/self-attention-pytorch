import torch
from _0_shared import *  # noqa: F403, F401


if __name__ == "__main__":
    config, train_dataloader, val_dataloader = get_model_config_and_data_loaders(  # noqa: F405
        train_path="04-practice/_1_data/train.csv",
        test_path="04-practice/_1_data/test.csv",
        max_tokens=100,
        embedding_dimensions=128,
        num_attention_heads=8,
        hidden_dropout_prob=0.3,
        num_encoder_layers=2,
        batch_size=64,
        smaller_dataset=False,
    )

    model = Transformer(config).to(config.device)  # noqa: F405
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(  # noqa: F405
        config,
        model,
        loss_function,
        optimizer,
        train_dataloader,
        val_dataloader,
        n_epochs=8,
        name_model="best_model_1",
    )
