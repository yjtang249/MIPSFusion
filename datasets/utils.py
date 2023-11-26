import torch


def get_camera_rays(H, W, fx, fy=None, cx=None, cy=None, type="OpenGL"):
    """Get ray origins, directions from a pinhole camera."""
    #  ----> i
    # |
    # |
    # X
    # j
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(H, dtype=torch.float32), indexing="xy")  # row_Id/col_Id of each pixel
    
    # View direction (X, Y, Lambda) / lambda
    # Move to the center of the screen
    #  -------------
    # |      y      |
    # |      |      |
    # |      .-- x  |
    # |             |
    # |             |
    #  -------------

    if cx is None:
        cx, cy = 0.5 * W, 0.5 * H

    if fy is None:
        fy = fx
    if type is "OpenGL":
        dirs = torch.stack( [ (i - cx)/fx, -(j - cy)/fy, -torch.ones_like(i) ], -1 )
    elif type is "OpenCV":
        dirs = torch.stack( [ (i - cx)/fx, (j - cy)/fy, torch.ones_like(i) ], -1 )
    else:
        raise NotImplementedError()

    rays_d = dirs  # Tensor(H, W, 3)
    return rays_d

