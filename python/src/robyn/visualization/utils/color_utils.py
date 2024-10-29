from typing import List, Tuple, Dict, Union
import plotly.graph_objects as go
import numpy as np

class ColorPalette:
    """Handles color schemes and gradients for consistent plot styling."""
    
    @staticmethod
    def get_brand_colors() -> Dict[str, str]:
        """Get standard brand color hex codes.
        
        Returns:
            Dict mapping color names to hex codes
        """
        return {
            'primary': '#59B3D2',
            'secondary': '#E5586E',
            'tertiary': '#39638b',
            'neutral': '#808080',
            'positive': '#228B22',
            'negative': '#8B4513',
            'highlight': '#FFD700'
        }
    
    @staticmethod
    def generate_color_scale(
        start_color: str,
        end_color: str,
        n_colors: int
    ) -> List[str]:
        """Generate a smooth color gradient between two colors.
        
        Args:
            start_color: Starting hex color code
            end_color: Ending hex color code
            n_colors: Number of colors to generate
            
        Returns:
            List of hex color codes forming a gradient
        """
        pass
    
    @staticmethod
    def get_channel_colors(channel_names: List[str]) -> Dict[str, str]:
        """Get consistent colors for marketing channels.
        
        Args:
            channel_names: List of channel names
            
        Returns:
            Dict mapping channel names to hex colors
        """
        pass