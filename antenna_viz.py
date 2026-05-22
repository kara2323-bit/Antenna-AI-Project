import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_antenna(antenna_id, lp, wp):
    # Set high resolution
    fig, ax = plt.subplots(figsize=(8, 10), dpi=100)
    
    # --- TECHNICAL DESIGN PALETTE ---
    substrate_color = '#F5DEB3' # FR-4 Wheat
    copper_color = '#CD7F32'    # Metallic Copper
    shadow_color = '#A9A9A9'    # Simple Gray Shadow
    
    lf, wf, g, ws, ls = 15.0, 3.0, 0.5, 50.0, 60.0
    g_ymax = 12.0 # Grounds end at 12mm (lf-3)

    # 1. DRAW SUBSTRATE
    ax.add_patch(patches.Rectangle((-ws/2, 0), ws, ls, color=substrate_color, ec='#8B7355', lw=1, zorder=1))
    
    # 2. HELPER FUNCTION (Manual Shadow + Metal)
    def add_metal(patch_obj):
        # Create a manual shadow by shifting the shape
        if isinstance(patch_obj, patches.Rectangle):
            shadow = patches.Rectangle((patch_obj.get_x()+0.3, patch_obj.get_y()-0.3), 
                                      patch_obj.get_width(), patch_obj.get_height(), color=shadow_color, zorder=2)
        elif isinstance(patch_obj, patches.Polygon):
            shadow = patches.Polygon(patch_obj.get_xy() + [0.3, -0.3], color=shadow_color, zorder=2)
        elif isinstance(patch_obj, patches.Ellipse):
            shadow = patches.Ellipse((patch_obj.center[0]+0.3, patch_obj.center[1]-0.3), 
                                    patch_obj.width, patch_obj.height, color=shadow_color, zorder=2)
        
        ax.add_patch(shadow)
        patch_obj.set_facecolor(copper_color)
        patch_obj.set_edgecolor('#5D2906')
        patch_obj.set_linewidth(1)
        patch_obj.set_zorder(3)
        ax.add_patch(patch_obj)

    # 3. DRAW GROUNDS AND FEED
    gw = (ws/2) - (wf/2) - g
    add_metal(patches.Rectangle((-ws/2, 0), gw, g_ymax)) # Left
    add_metal(patches.Rectangle((wf/2 + g, 0), gw, g_ymax)) # Right
    add_metal(patches.Rectangle((-wf/2, 0), wf, lf)) # Feed

    # 4. THE 12 ANTENNA SHAPES (EXACT CST REPLICAS)
    if antenna_id == 1: # Rectangle
        add_metal(patches.Rectangle((-wp/2, lf), wp, lp))
        
    elif antenna_id == 2: # Stepped
        add_metal(patches.Rectangle((-7.5, lf), 15, lp/2))
        add_metal(patches.Rectangle((-wp/2, lf + lp/2), wp, lp/2))
        
    elif antenna_id == 3: # T-Shape
        add_metal(patches.Rectangle((-1.5, lf), 3, lp/2))
        add_metal(patches.Rectangle((-wp/2, lf + lp/2), wp, lp/2))
        
    elif antenna_id == 4: # Ellipse
        add_metal(patches.Ellipse((0, lf + lp/2), wp, lp))
        
    elif antenna_id == 5: # Semi-Ellipse
        theta = np.linspace(0, np.pi, 50)
        x = (wp/2) * np.cos(theta)
        y = lf + lp * np.sin(theta)
        pts = np.column_stack([x, y])
        pts = np.vstack([pts, [wp/2, lf], [-wp/2, lf]])
        add_metal(patches.Polygon(pts))

    elif antenna_id == 6: # Pie-Sector (Curved Fan)
        # 1. Create the arc for the top of the "fan"
        # theta goes from 0 (Right) to pi (Left)
        theta = np.linspace(0, np.pi, 50)
        
        # Width controlled by wp
        x_arc = (wp/2) * np.cos(theta)
        
        # Height controlled by lp (Top arc starts at 60% of lp height)
        y_arc = (lf + lp * 0.6) + (lp * 0.4) * np.sin(theta)
        pts_arc = np.column_stack([x_arc, y_arc])
        
        # 2. COMBINE POINTS IN CORRECT CIRCULAR ORDER
        # Order: Bottom-Right -> Arc (Right to Left) -> Bottom-Left
        # This prevents the lines from crossing (the hourglass error)
        pts = np.vstack([
            [1.5, lf],      # Bottom-Right: Connects to feed line top-right corner
            pts_arc,        # The entire curved top (50 points)
            [-1.5, lf]      # Bottom-Left: Connects to feed line top-left corner
        ])
        
        add_metal(patches.Polygon(pts))

    elif antenna_id == 7: # Triangle
        add_metal(patches.Polygon([[-wp/2, lf], [wp/2, lf], [0, lf+lp]]))

    elif antenna_id == 8: # Trapezoid
        add_metal(patches.Polygon([[-1.5, lf], [1.5, lf], [wp/2, lf+lp], [-wp/2, lf+lp]]))

    elif antenna_id == 9: # Diamond
        add_metal(patches.Polygon([[-1.5, lf], [1.5, lf], [wp/2, lf+lp/2], [0, lf+lp], [-wp/2, lf+lp/2]]))

    elif antenna_id == 10: # Hexagon
        y1, y2 = lf + lp/3, lf + 2*lp/3
        pts = [[-1.5, lf], [1.5, lf], [wp/2, y1], [wp/2, y2], [1.5, lf+lp], [-1.5, lf+lp], [-wp/2, y2], [-wp/2, y1]]
        add_metal(patches.Polygon(pts))

    elif antenna_id == 11: # Pentagon
        add_metal(patches.Polygon([[-wp/2, lf], [wp/2, lf], [wp/2, lf+lp/2], [0, lf+lp], [-wp/2, lf+lp/2]]))

    elif antenna_id == 12: # Cross
        add_metal(patches.Rectangle((-1.5, lf), 3, lp))
        add_metal(patches.Rectangle((-wp/2, lf + lp/2 - 1.5), wp, 3))

    # 5. PROFESSIONAL DIMENSIONS (Technical Boxes)
    def add_label(start, end, text, is_vert=True):
        col = '#2C3E50'
        if is_vert:
            ax.annotate('', xy=(start[0], start[1]), xytext=(end[0], end[1]), arrowprops=dict(arrowstyle='<->', color=col))
            ax.text(start[0]+2, (start[1]+end[1])/2, text, rotation=90, va='center', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=1), fontsize=8)
        else:
            ax.annotate('', xy=(start[0], start[1]), xytext=(end[0], end[1]), arrowprops=dict(arrowstyle='<->', color='#C0392B'))
            ax.text((start[0]+end[0])/2, start[1]-4, text, ha='center', fontweight='bold', color='#C0392B',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec='#C0392B', lw=1), fontsize=8)

    add_label((wp/2 + 8, lf), (wp/2 + 8, lf + lp), f"Lp: {lp}mm")
    add_label((-wp/2, lf - 8), (wp/2, lf - 8), f"Wp: {wp}mm", is_vert=False)

    # Final settings
    ax.set_xlim(-30, 30); ax.set_ylim(-5, 65)
    ax.set_aspect('equal'); ax.axis('off')
    plt.title(f"AI PREDICTED DESIGN: ANTENNA {antenna_id}", fontsize=12, fontweight='bold', pad=20)
    return fig

if __name__ == "__main__":
    # Test the Stepped Rectangle (ID 2)
    draw_antenna(12, 15, 30)
    plt.show()