import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_workflow_fig(output_dir: str):
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # Hide axes
    ax.axis('off')
    
    steps = [
        "Simulation Setup",
        "Device Behaviour Generation",
        "Telemetry Collection",
        "Behaviour Detection",
        "Graph Analysis",
        "Risk Fusion",
        "Dual Trigger",
        "Metric Computation"
    ]
    
    num_steps = len(steps)
    box_width = 0.6
    box_height = 0.08
    vertical_spacing = 0.11
    
    x_center = 0.5
    start_y = 0.95
    
    for i, step in enumerate(steps):
        y = start_y - i * vertical_spacing
        
        # Draw box
        rect = patches.Rectangle(
            (x_center - box_width/2, y - box_height/2),
            box_width, box_height,
            linewidth=1.5,
            edgecolor='#1976D2',
            facecolor='#E3F2FD',
            zorder=2
        )
        ax.add_patch(rect)
        
        # Add text
        ax.text(
            x_center, y,
            step,
            ha='center',
            va='center',
            fontsize=10,
            fontweight='bold',
            fontfamily='serif',
            zorder=3
        )
        
        # Draw arrow to next step
        if i < num_steps - 1:
            next_y = start_y - (i + 1) * vertical_spacing
            ax.annotate(
                '',
                xy=(x_center, next_y + box_height/2),
                xytext=(x_center, y - box_height/2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#333333'),
                zorder=1
            )
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "experimental_workflow.pdf"), dpi=300)
    plt.savefig(os.path.join(output_dir, "experimental_workflow.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_workflow_fig(".")
