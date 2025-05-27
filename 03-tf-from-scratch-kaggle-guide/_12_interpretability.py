# Interactive Attention Visualization Tool

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import inspectus
import ipywidgets as widgets
from IPython.display import display, clear_output
import warnings

warnings.filterwarnings("ignore")

from _0_tokenization import tokenize_input, tokenizer
from _1_config import config
from _9_transformer import Transformer


class AttentionVisualizerNoPadding:
    """Extract and visualize attention patterns without PAD tokens"""

    def __init__(self, model_path: str):
        # Load the saved model
        checkpoint = torch.load(
            model_path, map_location=config.device, weights_only=False
        )

        # Create model instance and load weights
        self.model = Transformer(config).to(config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def extract_attention_without_padding(self, text: str):
        """Extract attention patterns and filter out PAD tokens"""

        # Tokenize input
        input_ids = tokenize_input(
            text, max_length=config.max_tokens, return_tensors="pt"
        )
        input_ids = input_ids.to(config.device)

        # Get all tokens (including PAD)
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

        # Find indices of non-PAD tokens
        non_pad_indices = [i for i, token in enumerate(all_tokens) if token != "[PAD]"]
        non_pad_tokens = [all_tokens[i] for i in non_pad_indices]

        # Extract attention weights
        all_attentions = []

        def extract_attn_from_multihead(module, input, output):
            hidden_state = input[0]
            head_attentions = []

            for head in module.heads:
                q = head.q(hidden_state)
                k = head.k(hidden_state)

                # Compute attention
                scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(q.size(-1))
                weights = torch.nn.functional.softmax(scores, dim=-1)

                # Filter out PAD tokens from attention matrix
                weights_np = weights.squeeze(0).cpu().numpy()

                # Extract only non-PAD rows and columns
                filtered_weights = weights_np[non_pad_indices, :][:, non_pad_indices]

                head_attentions.append(torch.tensor(filtered_weights))

            # Stack heads
            layer_attn = torch.stack(head_attentions)
            all_attentions.append(layer_attn)

        # Register hooks
        hooks = []
        for encoder in self.model.encoders:
            hook = encoder.attention.register_forward_hook(extract_attn_from_multihead)
            hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            _ = self.model(input_ids)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Stack all layers: [num_layers, num_heads, seq_len, seq_len]
        attention_tensor = torch.stack(all_attentions)

        return attention_tensor, non_pad_tokens, non_pad_indices

    def visualize_attention_head_clean(
        self, text: str, layer_idx: int, head_idx: int, save_path: str = None
    ):
        """Visualize attention pattern for a specific head without PAD tokens"""

        # Get filtered attention
        attention_tensor, tokens, _ = self.extract_attention_without_padding(text)

        # Get attention for specific layer and head
        attn = attention_tensor[layer_idx, head_idx].numpy()

        # Create figure
        plt.figure(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(
            attn,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="Blues",
            cbar_kws={"label": "Attention Weight"},
            square=True,
            vmin=0,
            vmax=1,
        )

        plt.title(
            f"Layer {layer_idx} - Head {head_idx} Attention Pattern (PAD tokens removed)"
        )
        plt.xlabel("Keys")
        plt.ylabel("Queries")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

        return attn, tokens, plt.gcf()

    def visualize_with_inspectus(self, text: str):
        """Visualize attention with inspectus, filtering PAD tokens"""

        # Get filtered attention
        attention_tensor, tokens, _ = self.extract_attention_without_padding(text)

        # Visualize with inspectus
        inspectus.attention(attention_tensor, tokens)

        return attention_tensor, tokens

    def visualize_all_heads_grid(self, text: str, save_path: str = None):
        """Create a grid visualization of all attention heads without PAD tokens"""

        # Get filtered attention
        attention_tensor, tokens, _ = self.extract_attention_without_padding(text)

        num_layers = config.num_encoder_layers
        num_heads = config.num_attention_heads

        fig, axes = plt.subplots(
            num_layers, num_heads, figsize=(4 * num_heads, 4 * num_layers)
        )

        if num_layers == 1:
            axes = axes.reshape(1, -1)
        if num_heads == 1:
            axes = axes.reshape(-1, 1)

        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                ax = axes[layer_idx, head_idx]
                attn = attention_tensor[layer_idx, head_idx].numpy()

                # Create heatmap
                im = ax.imshow(attn, cmap="Blues", aspect="auto", vmin=0, vmax=1)

                # Set title
                ax.set_title(f"L{layer_idx} H{head_idx}", fontsize=10)

                # Set tick labels for corner plots
                if layer_idx == num_layers - 1:
                    ax.set_xticks(range(len(tokens)))
                    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
                else:
                    ax.set_xticks([])

                if head_idx == 0:
                    ax.set_yticks(range(len(tokens)))
                    ax.set_yticklabels(tokens, fontsize=8)
                else:
                    ax.set_yticks([])

        plt.suptitle(
            f'All Attention Heads - "{text}" (PAD tokens removed)', fontsize=14
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

        return fig


def analyze_attention_patterns_clean(model_path: str, sample_texts: List[str]):
    """Analyze attention patterns with PAD filtering"""

    visualizer = AttentionVisualizerNoPadding(model_path)

    for i, text in enumerate(sample_texts):
        print(f"\nAnalyzing: '{text}'")

        # Visualize with inspectus (interactive)
        print("Opening inspectus visualization...")
        attention_tensor, tokens = visualizer.visualize_with_inspectus(text)

        # Create grid visualization
        visualizer.visualize_all_heads_grid(
            text, save_path=f"attention_grid_example_{i}.png"
        )

        # Visualize specific interesting heads
        for layer_idx in range(config.num_encoder_layers):
            for head_idx in range(min(3, config.num_attention_heads)):
                visualizer.visualize_attention_head_clean(
                    text,
                    layer_idx,
                    head_idx,
                    save_path=f"attention_clean_L{layer_idx}_H{head_idx}_ex{i}.png",
                )


class InteractiveAttentionVisualizer:
    """Interactive widget-based attention visualizer"""

    def __init__(self, model_path: str):
        self.visualizer = AttentionVisualizerNoPadding(model_path)
        self.current_attention = None
        self.current_tokens = None
        self.current_text = ""

        # Create widgets
        self.text_input = widgets.Textarea(
            value="The earthquake destroyed buildings",
            description="Input Text:",
            layout=widgets.Layout(width="600px", height="80px"),
        )

        self.layer_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=config.num_encoder_layers - 1,
            description="Layer:",
            continuous_update=False,
        )

        self.head_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=config.num_attention_heads - 1,
            description="Head:",
            continuous_update=False,
        )

        self.viz_type = widgets.Dropdown(
            options=[
                ("Single Head Heatmap", "single"),
                ("All Heads Grid", "grid"),
                ("Inspectus Interactive", "inspectus"),
            ],
            value="single",
            description="Visualization:",
        )

        self.analyze_button = widgets.Button(
            description="Analyze Text", button_style="primary"
        )

        self.output = widgets.Output()

        # Set up event handlers
        self.analyze_button.on_click(self._on_analyze_click)
        self.layer_slider.observe(self._on_param_change, names="value")
        self.head_slider.observe(self._on_param_change, names="value")
        self.viz_type.observe(self._on_param_change, names="value")

    def _on_analyze_click(self, b):
        """Handle analyze button click"""
        with self.output:
            clear_output()
            self.current_text = self.text_input.value.strip()
            if not self.current_text:
                print("Please enter some text to analyze")
                return

            print(f"Analyzing: '{self.current_text}'")
            print("Extracting attention patterns...")

            # Extract attention for current text
            self.current_attention, self.current_tokens, _ = (
                self.visualizer.extract_attention_without_padding(self.current_text)
            )

            print(f"Found {len(self.current_tokens)} tokens: {self.current_tokens}")
            print("Use the sliders above to explore different layers and heads")

            # Show initial visualization
            self._update_visualization()

    def _on_param_change(self, change):
        """Handle parameter changes"""
        if self.current_attention is not None:
            with self.output:
                self._update_visualization()

    def _update_visualization(self):
        """Update the visualization based on current parameters"""
        if self.current_attention is None:
            return

        viz_type = self.viz_type.value
        layer_idx = self.layer_slider.value
        head_idx = self.head_slider.value

        plt.close("all")  # Close previous plots

        if viz_type == "single":
            self._show_single_head(layer_idx, head_idx)
        elif viz_type == "grid":
            self._show_grid()
        elif viz_type == "inspectus":
            self._show_inspectus()

    def _show_single_head(self, layer_idx, head_idx):
        """Show single head attention heatmap"""
        attn = self.current_attention[layer_idx, head_idx].numpy()

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(attn, cmap="Blues", aspect="auto", vmin=0, vmax=1)

        # Set labels
        ax.set_xticks(range(len(self.current_tokens)))
        ax.set_yticks(range(len(self.current_tokens)))
        ax.set_xticklabels(self.current_tokens, rotation=45, ha="right")
        ax.set_yticklabels(self.current_tokens)

        ax.set_title(f"Layer {layer_idx} - Head {head_idx} Attention Pattern")
        ax.set_xlabel("Keys")
        ax.set_ylabel("Queries")

        # Add colorbar
        plt.colorbar(im, label="Attention Weight")
        plt.tight_layout()
        plt.show()

        # Show attention statistics
        print(f"\\nAttention Statistics for Layer {layer_idx}, Head {head_idx}:")
        print(f"Max attention: {attn.max():.3f}")
        print(f"Min attention: {attn.min():.3f}")
        print(f"Mean attention: {attn.mean():.3f}")

        # Find most attended tokens
        max_indices = np.unravel_index(np.argmax(attn), attn.shape)
        print(
            f"Strongest attention: '{self.current_tokens[max_indices[0]]}' -> '{self.current_tokens[max_indices[1]]}' ({attn[max_indices]:.3f})"
        )

    def _show_grid(self):
        """Show grid of all heads"""
        fig = self.visualizer.visualize_all_heads_grid(self.current_text)

    def _show_inspectus(self):
        """Show inspectus visualization"""
        print("Opening Inspectus interactive visualization...")
        inspectus.attention(self.current_attention, self.current_tokens)

    def display(self):
        """Display the interactive interface"""
        # Layout the widgets
        controls = widgets.VBox(
            [
                self.text_input,
                self.analyze_button,
                widgets.HBox([self.layer_slider, self.head_slider]),
                self.viz_type,
            ]
        )

        interface = widgets.VBox(
            [
                widgets.HTML("<h2>Interactive Attention Visualizer</h2>"),
                controls,
                self.output,
            ]
        )

        display(interface)


# Usage functions
def launch_interactive_visualizer(model_path: str = "best_model.pt"):
    """Launch the interactive attention visualizer"""
    visualizer = InteractiveAttentionVisualizer(model_path)
    visualizer.display()
    return visualizer


def quick_attention_analysis(model_path: str, text: str, layer: int = 0, head: int = 0):
    """Quick function to analyze attention for a specific text"""
    visualizer = AttentionVisualizerNoPadding(model_path)
    attn, tokens, fig = visualizer.visualize_attention_head_clean(text, layer, head)
    return attn, tokens, fig


class CommandLineInteractiveVisualizer:
    """Command-line interactive attention visualizer"""
    
    def __init__(self, model_path: str):
        self.visualizer = AttentionVisualizerNoPadding(model_path)
        self.current_attention = None
        self.current_tokens = None
        self.current_text = ""
        
    def run(self):
        """Run the interactive command-line interface"""
        print("🔍 Interactive Attention Visualizer")
        print("=" * 50)
        print(f"Model config: {config.num_encoder_layers} layers, {config.num_attention_heads} heads")
        print()
        
        while True:
            try:
                self._show_menu()
                choice = input("Enter your choice: ").strip()
                
                if choice == '1':
                    self._analyze_text()
                elif choice == '2':
                    self._show_single_head()
                elif choice == '3':
                    self._show_all_heads()
                elif choice == '4':
                    self._show_inspectus()
                elif choice == '5':
                    self._show_statistics()
                elif choice == 'q':
                    print("Goodbye!")
                    break
                else:
                    print("Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                
    def _show_menu(self):
        """Display the main menu"""
        print("\n📋 Main Menu:")
        print("1. Analyze new text")
        if self.current_attention is not None:
            print("2. Show single head attention")
            print("3. Show all heads grid")
            print("4. Show Inspectus visualization") 
            print("5. Show attention statistics")
        print("q. Quit")
        print()
        
    def _analyze_text(self):
        """Analyze new text input"""
        print("\n📝 Text Analysis")
        print("-" * 20)
        
        # Get text input
        text = input("Enter text to analyze: ").strip()
        if not text:
            print("No text provided.")
            return
            
        print(f"Analyzing: '{text}'")
        print("Extracting attention patterns...")
        
        try:
            # Extract attention
            self.current_attention, self.current_tokens, _ = self.visualizer.extract_attention_without_padding(text)
            self.current_text = text
            
            print(f"✅ Found {len(self.current_tokens)} tokens: {self.current_tokens}")
            print("Use menu options 2-5 to explore the attention patterns.")
            
        except Exception as e:
            print(f"❌ Error analyzing text: {e}")
            
    def _show_single_head(self):
        """Show single head attention"""
        if self.current_attention is None:
            print("Please analyze text first (option 1)")
            return
            
        print(f"\n🎯 Single Head Attention")
        print("-" * 25)
        
        # Get layer and head
        try:
            layer = int(input(f"Enter layer (0-{config.num_encoder_layers-1}): "))
            if not 0 <= layer < config.num_encoder_layers:
                print(f"Layer must be between 0 and {config.num_encoder_layers-1}")
                return
                
            head = int(input(f"Enter head (0-{config.num_attention_heads-1}): "))
            if not 0 <= head < config.num_attention_heads:
                print(f"Head must be between 0 and {config.num_attention_heads-1}")
                return
                
        except ValueError:
            print("Please enter valid numbers")
            return
            
        # Visualize
        print(f"Showing Layer {layer}, Head {head}...")
        attn, tokens, fig = self.visualizer.visualize_attention_head_clean(
            self.current_text, layer, head
        )
        
        # Show statistics
        print(f"\nAttention Statistics:")
        print(f"Max: {attn.max():.3f}, Min: {attn.min():.3f}, Mean: {attn.mean():.3f}")
        
        max_indices = np.unravel_index(np.argmax(attn), attn.shape)
        print(f"Strongest: '{tokens[max_indices[0]]}' → '{tokens[max_indices[1]]}' ({attn[max_indices]:.3f})")
        
    def _show_all_heads(self):
        """Show all heads grid"""
        if self.current_attention is None:
            print("Please analyze text first (option 1)")
            return
            
        print(f"\n🔄 All Heads Grid")
        print("-" * 20)
        print("Generating grid visualization...")
        
        fig = self.visualizer.visualize_all_heads_grid(self.current_text)
        print("Grid visualization displayed!")
        
    def _show_inspectus(self):
        """Show Inspectus visualization"""
        if self.current_attention is None:
            print("Please analyze text first (option 1)")
            return
            
        print(f"\n🎮 Inspectus Interactive")
        print("-" * 25)
        print("Opening Inspectus visualization...")
        
        inspectus.attention(self.current_attention, self.current_tokens)
        
    def _show_statistics(self):
        """Show detailed attention statistics"""
        if self.current_attention is None:
            print("Please analyze text first (option 1)")
            return
            
        print(f"\n📊 Attention Statistics")
        print("-" * 25)
        print(f"Text: '{self.current_text}'")
        print(f"Tokens: {self.current_tokens}")
        print(f"Shape: {self.current_attention.shape}")
        print()
        
        # Per-layer statistics
        for layer_idx in range(config.num_encoder_layers):
            layer_attn = self.current_attention[layer_idx]
            print(f"Layer {layer_idx}:")
            
            for head_idx in range(config.num_attention_heads):
                head_attn = layer_attn[head_idx].numpy()
                max_val = head_attn.max()
                mean_val = head_attn.mean()
                max_indices = np.unravel_index(np.argmax(head_attn), head_attn.shape)
                strongest_pair = f"'{self.current_tokens[max_indices[0]]}' → '{self.current_tokens[max_indices[1]]}'"
                
                print(f"  Head {head_idx}: max={max_val:.3f}, mean={mean_val:.3f}, strongest={strongest_pair}")
            print()


def launch_command_line_visualizer(model_path: str = "best_model.pt"):
    """Launch command-line interactive visualizer"""
    try:
        visualizer = CommandLineInteractiveVisualizer(model_path)
        visualizer.run()
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        print("Make sure you have trained a model first or provide the correct path.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")


# Usage example
if __name__ == "__main__":
    # For Jupyter notebook usage
    try:
        get_ipython()
        print("Jupyter environment detected. Use: launch_interactive_visualizer()")
        print("Example: visualizer = launch_interactive_visualizer('best_model.pt')")
    except NameError:
        # Command line usage - run interactive visualizer
        model_path = "best_model.pt"
        launch_command_line_visualizer(model_path)
