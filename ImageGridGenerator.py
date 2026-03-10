import os
from PIL import Image

def generate_movement_grids(source_dir='./signal_plots', save_dir='./movement_grids', scale_factor=0.5):
    """
    Reads individual channel PNGs and stitches them into a massive grid for each movement.
    Columns = Participants (1 to 8)
    Rows = Channels (0 to 7)
    """
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Starting grid generation in '{save_dir}'...")

    # Grab one image to determine the cell dimensions
    sample_img_path = os.path.join(source_dir, 'p1_m1_channel0.png')
    if not os.path.exists(sample_img_path):
        print(f"Error: Could not find '{sample_img_path}'. Did you run the plot generator first?")
        return

    with Image.open(sample_img_path) as img:
        orig_width, orig_height = img.size
        
    # Apply scaling to keep the final file size manageable
    cell_width = int(orig_width * scale_factor)
    cell_height = int(orig_height * scale_factor)

    num_participants = 8
    num_channels = 8

    # Calculate the total canvas size for the 8x8 grid
    grid_width = num_participants * cell_width
    grid_height = num_channels * cell_height

    print(f"Final grid resolution will be {grid_width}x{grid_height} pixels per movement.")

    # Generate 1 grid per movement (Movements 1 through 9)
    for m in range(1, 10):
        print(f"Stitching Movement {m}...")
        
        # Create a blank white canvas
        canvas = Image.new('RGB', (grid_width, grid_height), 'white')
        
        # Populate the grid
        for c in range(num_channels):          # Rows
            for p in range(1, num_participants + 1):  # Columns
                
                img_name = f"p{p}_m{m}_channel{c}.png"
                img_path = os.path.join(source_dir, img_name)
                
                if os.path.exists(img_path):
                    with Image.open(img_path) as cell_img:
                        # Resize the image if a scale factor is applied
                        if scale_factor != 1.0:
                            cell_img = cell_img.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
                        
                        # Calculate where to paste this specific image
                        x_offset = (p - 1) * cell_width
                        y_offset = c * cell_height
                        
                        # Paste it into the massive canvas
                        canvas.paste(cell_img, (x_offset, y_offset))
                else:
                    print(f"  -> Warning: Missing image {img_name}, leaving a blank space.")
        
        # Save the completed grid
        save_path = os.path.join(save_dir, f'Movement_{m}_Grid.png')
        canvas.save(save_path)
        print(f"Saved: {save_path}")

    print("All movement grids generated successfully!")

# ====================================================================================
# ============================== EXECUTION ===========================================
# ====================================================================================

if __name__ == "__main__":
    # Note: Make sure to pip install Pillow if you haven't already
    generate_movement_grids(source_dir='./signal_plots', save_dir='./movement_grids', scale_factor=0.5)