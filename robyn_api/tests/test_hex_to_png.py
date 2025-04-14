import unittest
import os
import binascii
from PIL import Image
import io
import sys
sys.path.append('..')  # Add parent directory to path
from python_helper import hexToPng

class TestPythonHelper(unittest.TestCase):
    def setUp(self):
        # Create a small test image in memory
        self.test_image = Image.new('RGB', (100, 100), color='red')
        # Save it to a bytes buffer
        img_buffer = io.BytesIO()
        self.test_image.save(img_buffer, format='PNG')
        # Convert to hex
        self.test_hex = binascii.hexlify(img_buffer.getvalue()).decode()
        
    def test_hexToPng(self):
        # Test file name
        test_file = 'test_output.png'
        
        # Run the function
        hexToPng(test_file, self.test_hex)
        
        # Check if file exists
        self.assertTrue(os.path.exists(test_file))
        
        # Check if it's a valid image
        try:
            img = Image.open(test_file)
            self.assertEqual(img.size, (100, 100))
        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)
    
    def test_invalid_hex(self):
        # Test with invalid hex data
        with self.assertRaises(binascii.Error):
            hexToPng('invalid.png', 'invalid_hex')

if __name__ == '__main__':
    unittest.main()
