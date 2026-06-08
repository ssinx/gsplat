#!/usr/bin/env python
"""Visualize semantic palette colors for each instance ID."""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def generate_semantic_palette(num_classes):
    """Generate semantic palette (same as _semantic_palette in simple_trainer.py)."""
    # Deterministic, well-separated colors via golden-ratio hue spacing.
    hues = (torch.arange(num_classes, dtype=torch.float32) * 0.61803398875) % 1.0
    # HSV (s=0.85, v=0.95) -> RGB without external deps.
    s, v = 0.85, 0.95
    h6 = hues * 6.0
    c = v * s
    x = c * (1.0 - (h6 % 2.0 - 1.0).abs())
    m = v - c
    z = torch.zeros_like(hues)
    i = h6.long() % 6
    rgb = torch.stack(
        [
            torch.where(
                i == 0, c, torch.where(i == 1, x, torch.where(i == 2, z, torch.where(i == 3, z, torch.where(i == 4, x, c))))
            ),
            torch.where(
                i == 0, x, torch.where(i == 1, c, torch.where(i == 2, c, torch.where(i == 3, x, torch.where(i == 4, z, z))))
            ),
            torch.where(
                i == 0, z, torch.where(i == 1, z, torch.where(i == 2, x, torch.where(i == 3, c, torch.where(i == 4, c, x))))
            ),
        ],
        dim=-1,
    ) + m
    return rgb.numpy()  # [num_classes, 3]

def visualize_palette(num_classes=16, output_path="semantic_palette.png"):
    """Create visualization of semantic palette."""

    palette = generate_semantic_palette(num_classes)

    # Create figure with color swatches
    fig = plt.figure(figsize=(12, max(8, num_classes * 0.4)))

    # Create grid layout
    rows = (num_classes + 3) // 4  # 4 colors per row
    cols = 4

    for idx in range(num_classes):
        ax = plt.subplot(rows, cols, idx + 1)

        # Create color swatch
        color = palette[idx]
        swatch = np.ones((100, 100, 3)) * color

        ax.imshow(swatch)
        ax.set_title(f"ID: {idx}", fontsize=12, fontweight='bold')
        ax.axis('off')

        # Add RGB values as text
        rgb_text = f"RGB: ({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})"
        ax.text(50, 110, rgb_text, ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Semantic palette visualization saved to: {output_path}")

    # Also create a compact horizontal strip version
    fig2, ax2 = plt.subplots(1, 1, figsize=(num_classes * 0.8, 2))

    # Create horizontal color strip
    strip = palette[np.newaxis, :, :]  # [1, num_classes, 3]
    strip = np.repeat(strip, 100, axis=0)  # [100, num_classes, 3]

    ax2.imshow(strip, aspect='auto')
    ax2.set_yticks([])
    ax2.set_xticks(range(num_classes))
    ax2.set_xticklabels([f"ID {i}" for i in range(num_classes)], rotation=45, ha='right')
    ax2.set_title(f"Semantic Palette ({num_classes} classes)", fontsize=14, fontweight='bold')

    strip_path = output_path.replace('.png', '_strip.png')
    plt.tight_layout()
    plt.savefig(strip_path, dpi=150, bbox_inches='tight')
    print(f"✓ Compact strip version saved to: {strip_path}")

    # Print color table
    print(f"\nSemantic Palette Color Table ({num_classes} classes):")
    print("=" * 60)
    print(f"{'ID':<4} {'RGB (0-1)':<30} {'RGB (0-255)':<20} {'Hex':<10}")
    print("-" * 60)
    for idx in range(num_classes):
        color = palette[idx]
        rgb_01 = f"({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})"
        rgb_255 = f"({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})"
        hex_color = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
        print(f"{idx:<4} {rgb_01:<30} {rgb_255:<20} {hex_color:<10}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize semantic palette")
    parser.add_argument("--num_classes", type=int, default=16,
                        help="Number of semantic classes")
    parser.add_argument("--output", type=str, default="semantic_palette.png",
                        help="Output image path")
    args = parser.parse_args()

    visualize_palette(args.num_classes, args.output)
