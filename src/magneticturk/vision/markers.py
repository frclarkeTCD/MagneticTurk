"""Find ArUco markers in an image and estimate where the view sits on the board.

Input: image. Output: homography / pose mapping image pixels to the board frame,
plus which squares are (fully or partly) in view.
"""
