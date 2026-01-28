import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. Load Data
try:
    df = pd.read_csv("memory_log.csv")
except FileNotFoundError:
    print("Error: memory_log.csv not found. Run the C++ executable first!")
    exit()

# 2. Setup Plot
fig, (ax_table, ax_map) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1]})
plt.subplots_adjust(hspace=0.4)

# --- Top: Allocation Report (Table) ---
ax_table.axis('off')
ax_table.set_title("Standard Allocator (cudaMalloc) Layout - Allocation Report", fontsize=14, pad=20, color='white')
fig.patch.set_facecolor('#1e1e1e') # Dark mode like PDF

# Prepare table data
table_data = df[['Name', 'Size_KB', 'Offset_KB']].copy()
table_data['Size_KB'] = table_data['Size_KB'].apply(lambda x: f"{x:.3f} KB")
table_data['Offset_KB'] = table_data['Offset_KB'].apply(lambda x: f"+{x:.0f} KB")

# Draw Table
table = ax_table.table(cellText=table_data.values,
                       colLabels=["Name", "Size", "Address Offset"],
                       loc='center', cellLoc='left', colColours=['#333']*3)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# Style Table cells
for key, cell in table.get_celld().items():
    cell.set_text_props(color='white')
    cell.set_facecolor('#2d2d2d')
    cell.set_edgecolor('#444')

# --- Bottom: Virtual Memory Map (Diagram) ---
ax_map.set_title("Virtual Memory Map (Visualizing Fragmentation)", fontsize=12, color='white')
ax_map.set_facecolor('#1e1e1e')
ax_map.set_xlabel("Memory Address Space (KB)", color='white')
ax_map.tick_params(axis='x', colors='white')
ax_map.get_yaxis().set_visible(False)

# Calculate total span
max_offset = df.iloc[-1]['Offset_KB'] + df.iloc[-1]['Size_KB']
ax_map.set_xlim(-100, max_offset + 100)
ax_map.set_ylim(0, 2)

# Colors for blocks
colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7']

# Draw Blocks
for i, row in df.iterrows():
    # Draw Allocation Block
    rect = patches.Rectangle((row['Offset_KB'], 0.5), row['Size_KB'], 1, 
                             linewidth=1, edgecolor='black', facecolor=colors[i % len(colors)])
    ax_map.add_patch(rect)
    
    # Label inside block if big enough
    if row['Size_KB'] > (max_offset * 0.05):
        ax_map.text(row['Offset_KB'] + row['Size_KB']/2, 1.0, row['Name'], 
                    ha='center', va='center', color='white', fontsize=8, rotation=90)

    # Draw Gap (Red) if exists
    if row['Gap_Before_KB'] > 0:
        gap_start = row['Offset_KB'] - row['Gap_Before_KB']
        rect_gap = patches.Rectangle((gap_start, 0.5), row['Gap_Before_KB'], 1, 
                                     linewidth=0, facecolor='#ff0000', alpha=0.3)
        ax_map.add_patch(rect_gap)
        # Label Gap
        if row['Gap_Before_KB'] > (max_offset * 0.05):
            ax_map.text(gap_start + row['Gap_Before_KB']/2, 0.7, "GAP", 
                        ha='center', va='center', color='red', fontsize=8, fontweight='bold')

# Legend for Gap
red_patch = patches.Patch(color='#ff0000', alpha=0.3, label='External Fragmentation (Gaps)')
ax_map.legend(handles=[red_patch], loc='upper right', facecolor='#333', labelcolor='white')

plt.savefig("memory_diagram.png", dpi=150, bbox_inches='tight')
print("Diagram saved to memory_diagram.png")