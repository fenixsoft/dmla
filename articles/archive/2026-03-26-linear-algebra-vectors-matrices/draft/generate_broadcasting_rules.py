"""
Broadcasting Rules Visualization
Using English labels to avoid font issues
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def draw_array(ax, data, origin, color, label, text_color='white', highlight=None, alpha=0.85):
    """Draw array blocks"""
    rows, cols = data.shape
    x0, y0 = origin

    for i in range(rows):
        for j in range(cols):
            x = x0 + j
            y = y0 - i
            face_color = color
            edge_color = 'darkgray'
            linewidth = 1.5

            # Highlight specific elements
            if highlight and (i, j) == highlight:
                face_color = '#FFD93D'
                edge_color = '#FF6B35'
                linewidth = 3

            rect = FancyBboxPatch((x + 0.05, y + 0.05), 0.9, 0.9,
                                    boxstyle="round,pad=0.02",
                                    facecolor=face_color, edgecolor=edge_color,
                                    linewidth=linewidth, alpha=alpha)
            ax.add_patch(rect)

            # Show value
            value = data[i, j]
            if isinstance(value, float):
                text = f'{value:.0f}'
            else:
                text = str(int(value))
            ax.text(x + 0.5, y + 0.5, text, ha='center', va='center',
                   fontsize=10, color=text_color, fontweight='bold')

    # Label
    ax.text(x0 + cols/2, y0 + 1.2, label, ha='center', va='bottom',
           fontsize=12, fontweight='bold')

def draw_rule1(ax):
    """Rule 1: Pad with 1s"""
    ax.set_xlim(0, 14)
    ax.set_ylim(-1, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Rule 1: Pad dimensions with 1s', fontsize=14, fontweight='bold', pad=20, color='#1976D2')

    # Array A: (3, 4)
    A = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]], dtype=float)
    draw_array(ax, A, (0.5, 4), '#4A90E2', 'Array A\nshape (3, 4)')
    ax.text(2.5, 3.3, 'ndim=2', ha='center', fontsize=10, color='#666', style='italic')

    # Plus sign
    ax.text(5, 2.5, '+', fontsize=24, fontweight='bold', ha='center', va='center', color='#333')

    # Array B: (4,) shown as 1D
    B_1d = np.array([10, 20, 30, 40], dtype=float)
    # Draw as horizontal row
    for j, val in enumerate(B_1d):
        rect = FancyBboxPatch((6.5 + j + 0.05, 1.55), 0.9, 0.9,
                                boxstyle="round,pad=0.02",
                                facecolor='#81C784', edgecolor='darkgray',
                                linewidth=1.5, alpha=0.85)
        ax.add_patch(rect)
        ax.text(6.5 + j + 0.5, 2, f'{val:.0f}', ha='center', va='center',
               fontsize=10, color='white', fontweight='bold')
    ax.text(8.5, 3.2, 'Array B\nshape (4,)', ha='center', fontsize=12, fontweight='bold')
    ax.text(8.5, 0.8, 'ndim=1', ha='center', fontsize=10, color='#666', style='italic')

    # Arrow pointing down
    ax.annotate('', xy=(8.5, 0.5), xytext=(8.5, 1.3),
               arrowprops=dict(arrowstyle='->', color='#E53935', lw=2))
    ax.text(9.5, 0.9, 'Pad', fontsize=10, color='#E53935', fontweight='bold')

    # B after padding: (1, 4)
    B_padded = B_1d.reshape(1, 4)
    draw_array(ax, B_padded, (10.5, 1.5), '#66BB6A', 'After padding\nshape (1, 4)')

    # Note box
    note = FancyBboxPatch((5.5, -0.8), 7.5, 1.2, boxstyle="round,pad=0.1",
                         facecolor='#FFF9C4', edgecolor='#FBC02D', linewidth=2)
    ax.add_patch(note)
    ax.text(9.25, -0.2, 'Original: (4,)  ->  After padding: (1, 4)', fontsize=10,
           ha='center', va='center', fontweight='bold', color='#F57F17')

def draw_rule2(ax):
    """Rule 2: Check compatibility"""
    ax.set_xlim(0, 13)
    ax.set_ylim(-1, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Rule 2: Check compatibility', fontsize=14, fontweight='bold', pad=20, color='#388E3C')

    # Array A: (3, 4)
    A = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]], dtype=float)
    draw_array(ax, A, (0.5, 4), '#4A90E2', 'A: (3, 4)')

    # Compatibility check labels
    # dim 0: 3 vs 1 -> compatible (one is 1)
    ax.annotate('', xy=(5.2, 3.8), xytext=(4.8, 3.8),
               arrowprops=dict(arrowstyle='<->', color='#7B1FA2', lw=2.5))
    ax.text(5, 4.4, 'dim 0: 3 vs 1', ha='center', fontsize=10, color='white',
           fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#7B1FA2', edgecolor='#4A148C', pad=0.3))
    ax.text(5, 4.8, 'compatible (one is 1)', ha='center', fontsize=8, color='#7B1FA2')

    # Array B: (1, 4)
    B = np.array([[10, 20, 30, 40]], dtype=float)
    draw_array(ax, B, (6, 2), '#66BB6A', 'B: (1, 4)')

    # dim 1: 4 vs 4 -> compatible (same)
    ax.annotate('', xy=(3.5, 0.5), xytext=(7.5, 0.5),
               arrowprops=dict(arrowstyle='<->', color='#2E7D32', lw=2.5))
    ax.text(5.5, -0.2, 'dim 1: 4 = 4', ha='center', fontsize=10, color='white',
           fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#2E7D32', edgecolor='#1B5E20', pad=0.3))
    ax.text(5.5, 0.2, 'compatible (same)', ha='center', fontsize=8, color='#2E7D32')

    # Compatibility rules box
    rules_box = FancyBboxPatch((9, 2), 3.5, 3, boxstyle="round,pad=0.1",
                              facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2)
    ax.add_patch(rules_box)
    ax.text(10.75, 4.5, 'Rules:', fontsize=11, ha='center', fontweight='bold', color='#1565C0')
    ax.text(10.75, 3.8, 'same size  OK', fontsize=10, ha='center', color='#2E7D32')
    ax.text(10.75, 3.2, 'one is 1  OK', fontsize=10, ha='center', color='#2E7D32')
    ax.text(10.75, 2.6, 'otherwise ERROR', fontsize=10, ha='center', color='#C62828')

def draw_rule3(ax):
    """Rule 3: Stretch to match"""
    ax.set_xlim(0, 15)
    ax.set_ylim(-2, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Rule 3: Stretch to match', fontsize=14, fontweight='bold', pad=20, color='#F57C00')

    # Array A: (3, 4)
    A = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]], dtype=float)
    draw_array(ax, A, (0.5, 4), '#4A90E2', 'A: (3, 4)')

    # B original (1, 4)
    B = np.array([[10, 20, 30, 40]], dtype=float)
    draw_array(ax, B, (6, 2), '#66BB6A', 'B: (1, 4)\nstretch ->')

    # Draw stretched B (3, 4) with dashed outline
    stretch_color = '#FFD54F'
    for i in range(3):
        for j in range(4):
            x = 10.5 + j + 0.05
            y = 4 - i + 0.05
            rect = FancyBboxPatch((x, y), 0.9, 0.9,
                                boxstyle="round,pad=0.02",
                                facecolor=stretch_color, edgecolor='#FF8F00',
                                linewidth=2, alpha=0.7, linestyle='--')
            ax.add_patch(rect)
            # Show that values are replicated
            val = B[0, j]
            ax.text(x + 0.45, y + 0.45, f'{val:.0f}',
                   ha='center', va='center', fontsize=9, color='#E65100', fontweight='bold')

    ax.text(12.5, 5.2, 'B stretched: (3, 4)', ha='center', fontsize=11,
           fontweight='bold', color='#E65100')

    # Draw stretch arrows from original B to stretched positions
    for i in range(3):
        ax.annotate('', xy=(10.3, 4-i+0.5), xytext=(9.7, 2),
                   arrowprops=dict(arrowstyle='->', color='#E53935', lw=1.5,
                                  connectionstyle="arc3,rad=0.1"))

    # Note
    ax.text(12.5, -0.5, 'Size-1 dimension\nstretched/replicated', ha='center', fontsize=10,
           style='italic', color='#E53935', fontweight='bold')

    # Result
    result_box = FancyBboxPatch((10.5, -1.5), 4, 0.8, boxstyle="round,pad=0.05",
                               facecolor='#C8E6C9', edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(result_box)
    ax.text(12.5, -1.1, 'A + B (element-wise)', ha='center', fontsize=11,
           fontweight='bold', color='#1B5E20')

def draw_summary(ax):
    """Summary flowchart"""
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Broadcasting Process Summary', fontsize=14, fontweight='bold', pad=20, color='#424242')

    # Step 1
    rect1 = FancyBboxPatch((0.5, 5), 3.5, 1.3, boxstyle="round,pad=0.05",
                          facecolor='#BBDEFB', edgecolor='#1976D2', linewidth=2.5)
    ax.add_patch(rect1)
    ax.text(2.25, 5.9, 'Step 1', fontsize=11, fontweight='bold', color='#0D47A1')
    ax.text(2.25, 5.4, 'Pad with 1s', fontsize=12, ha='center', fontweight='bold')
    ax.text(2.25, 4.7, 'Align dimensions', fontsize=9, ha='center', color='#666', style='italic')

    # Arrow
    ax.annotate('', xy=(4.5, 5.65), xytext=(4.1, 5.65),
               arrowprops=dict(arrowstyle='->', color='#666', lw=2.5))

    # Step 2
    rect2 = FancyBboxPatch((4.7, 5), 3.5, 1.3, boxstyle="round,pad=0.05",
                          facecolor='#C8E6C9', edgecolor='#388E3C', linewidth=2.5)
    ax.add_patch(rect2)
    ax.text(6.45, 5.9, 'Step 2', fontsize=11, fontweight='bold', color='#1B5E20')
    ax.text(6.45, 5.4, 'Check', fontsize=12, ha='center', fontweight='bold')
    ax.text(6.45, 4.7, 'same or one is 1', fontsize=9, ha='center', color='#666', style='italic')

    # Arrow
    ax.annotate('', xy=(8.7, 5.65), xytext=(8.3, 5.65),
               arrowprops=dict(arrowstyle='->', color='#666', lw=2.5))

    # Step 3
    rect3 = FancyBboxPatch((9, 5), 3.5, 1.3, boxstyle="round,pad=0.05",
                          facecolor='#FFE0B2', edgecolor='#F57C00', linewidth=2.5)
    ax.add_patch(rect3)
    ax.text(10.75, 5.9, 'Step 3', fontsize=11, fontweight='bold', color='#E65100')
    ax.text(10.75, 5.4, 'Stretch', fontsize=12, ha='center', fontweight='bold')
    ax.text(10.75, 4.7, 'Replicate values', fontsize=9, ha='center', color='#666', style='italic')

    # Example box
    example_box = FancyBboxPatch((0.5, 0.5), 13, 3.5, boxstyle="round,pad=0.05",
                                facecolor='#FAFAFA', edgecolor='#757575',
                                linewidth=2, linestyle='--')
    ax.add_patch(example_box)

    ax.text(7, 3.5, 'Example: (3, 4) + (4,)  Result: (3, 4)', fontsize=13,
           ha='center', fontweight='bold', color='#212121')

    # Detailed steps
    ax.text(1.2, 2.7, '1. (4,) -> (1, 4)     [Pad with 1]', fontsize=11, color='#424242')
    ax.text(1.2, 2.1, '2. (3,4) vs (1,4)    dim0: 3vs1 OK  dim1: 4vs4 OK', fontsize=11, color='#424242')
    ax.text(1.2, 1.5, '3. (1,4) -> (3,4)     [Stretch]', fontsize=11, color='#424242')

    # Final result box
    result_box = FancyBboxPatch((9.5, 1.2), 3.5, 2, boxstyle="round,pad=0.05",
                               facecolor='#A5D6A7', edgecolor='#2E7D32', linewidth=2.5)
    ax.add_patch(result_box)
    ax.text(11.25, 2.7, 'Result', fontsize=11, ha='center', fontweight='bold', color='#1B5E20')
    ax.text(11.25, 2.1, 'shape (3, 4)', fontsize=12, ha='center', fontweight='bold', color='#1B5E20')
    ax.text(11.25, 1.6, 'Element-wise', fontsize=9, ha='center', color='#2E7D32')

def main():
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('NumPy Broadcasting Rules Explained', fontsize=18, fontweight='bold', y=0.98, color='#212121')

    # Rule 1
    ax1 = fig.add_subplot(2, 2, 1)
    draw_rule1(ax1)

    # Rule 2
    ax2 = fig.add_subplot(2, 2, 2)
    draw_rule2(ax2)

    # Rule 3
    ax3 = fig.add_subplot(2, 2, 3)
    draw_rule3(ax3)

    # Summary
    ax4 = fig.add_subplot(2, 2, 4)
    draw_summary(ax4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = '/root/devspaces/ideaspaces/docs/linear-algebra-vectors-matrices/images/broadcasting_rules.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.3)
    print(f"Broadcasting rules visualization saved to: {output_path}")

    plt.show()

if __name__ == '__main__':
    main()
