import torch
from _0_tokenization import tokenize_input
from _9_transformer import Transformer


def load_model(config, model_path):
    """Load the trained model."""
    checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
    model = Transformer(config).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict(text, config, model=None, debug=False):
    if model is None:
        model = load_model()

    # Tokenize input
    tokens = tokenize_input(text, max_length=config.max_tokens, return_tensors="pt")
    tokens = tokens.to(config.device)

    if debug:
        print(f"Input text: {text}")
        print(f"Tokens shape: {tokens.shape}")
        print(f"First few tokens: {tokens[0][:10]}")

    # Get prediction
    with torch.no_grad():
        output = model(tokens)
        if debug:
            print(f"Raw model output: {output}")
            print(f"Output shape: {output.shape}")
            print(f"Output value: {output.squeeze().item()}")

        prediction = output.squeeze().item() > 0.5

    return "negative" if prediction else "positive"


if __name__ == "__main__":
    from _1_config import config

    model = load_model(config, "03-tf-from-scratch-kaggle-guide/best_model.pt")
    texts = [
        "Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all",  # 1
        "Just got sent this photo from Ruby #Alaska as smoke from #wildfires pours into a school",  # 1
        "I love fruits",  # 0
        "London is cool ;)",  # 0
    ]

    for text in texts:
        result = predict(text, config, model)
        print(f"'{text}' -> {result}")
