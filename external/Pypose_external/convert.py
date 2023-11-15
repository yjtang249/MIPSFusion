import torch, warnings
import pypose as pp


def mat2SO3(mat, check=True, rtol=1e-5, atol=1e-5):
    r"""Convert batched rotation or transformation matrices to SO3Type LieTensor.

    Args:
        mat (Tensor): the batched matrices to convert. If input is of shape :obj:`(*, 3, 4)`
            or :obj:`(*, 4, 4)`, only the top left 3x3 submatrix is used.
        check (bool, optional): flag to check if the input is valid rotation matrices (orthogonal
            and with a determinant of one). Set to ``False`` if less computation is needed.
            Default: ``True``.
        rtol (float, optional): relative tolerance when check is enabled. Default: 1e-05
        atol (float, optional): absolute tolerance when check is enabled. Default: 1e-05

    Return:
        LieTensor: the converted SO3Type LieTensor.

    Shape:
        Input: :obj:`(*, 3, 3)` or :obj:`(*, 3, 4)` or :obj:`(*, 4, 4)`

        Output: :obj:`(*, 4)`

    Let the input be matrix :math:`\mathbf{R}`, :math:`\mathbf{R}_i` represents each individual
    matrix in the batch. :math:`\mathbf{R}^{m,n}_i` represents the :math:`m^{\mathrm{th}}` row
    and :math:`n^{\mathrm{th}}` column of :math:`\mathbf{R}_i`, :math:`m,n\geq 1`, then the 
    quaternion can be computed by:

    .. math::
        \left\{\begin{aligned}
        q^x_i &= \mathrm{sign}(\mathbf{R}^{2,3}_i - \mathbf{R}^{3,2}_i) \frac{1}{2}
            \sqrt{1 + \mathbf{R}^{1,1}_i - \mathbf{R}^{2,2}_i - \mathbf{R}^{3,3}_i}\\
        q^y_i &= \mathrm{sign}(\mathbf{R}^{3,1}_i - \mathbf{R}^{1,3}_i) \frac{1}{2}
            \sqrt{1 - \mathbf{R}^{1,1}_i + \mathbf{R}^{2,2}_i - \mathbf{R}^{3,3}_i}\\
        q^z_i &= \mathrm{sign}(\mathbf{R}^{1,2}_i - \mathbf{R}^{2,1}_i) \frac{1}{2}
            \sqrt{1 - \mathbf{R}^{1,1}_i - \mathbf{R}^{2,2}_i + \mathbf{R}^{3,3}_i}\\
        q^w_i &= \frac{1}{2} \sqrt{1 + \mathbf{R}^{1,1}_i + \mathbf{R}^{2,2}_i + \mathbf{R}^{3,3}_i}
        \end{aligned}\right.,
    
    In summary, the output LieTensor should be of format:

    .. math::
        \textbf{y}_i = [q^x_i, q^y_i, q^z_i, q^w_i]

    Warning:
        Numerically, a transformation matrix is considered legal if:

        .. math::
            |{\rm det}(\mathbf{R}) - 1| \leq \texttt{atol} + \texttt{rtol}\times 1\\
            |\mathbf{RR}^{T} - \mathbf{I}| \leq \texttt{atol} + \texttt{rtol}\times \mathbf{I}
        
        where :math:`|\cdot |` is element-wise absolute function. When ``check`` is set to ``True``,
        illegal input will raise a ``ValueError``. Otherwise, no data validation is performed. 
        Illegal input will output an irrelevant result, which likely contains ``nan``.
        
    Examples:

        >>> input = torch.tensor([[0., -1.,  0.],
        ...                       [1.,  0.,  0.],
        ...                       [0.,  0.,  1.]])
        >>> pp.mat2SO3(input)
        SO3Type LieTensor:
        tensor([0.0000, 0.0000, 0.7071, 0.7071])

    See :meth:`pypose.SO3` for more details of the output LieTensor format.
    """

    if not torch.is_tensor(mat):
        mat = torch.tensor(mat)

    if len(mat.shape) < 2:
        raise ValueError("Input size must be at least 2 dimensions. Got {}".format(mat.shape))

    if not (mat.shape[-2:] == (3, 3) or mat.shape[-2:] == (3, 4) or mat.shape[-2:] == (4, 4)):
        raise ValueError("Input size must be a * x 3 x 3 or * x 3 x 4 or * x 4 x 4 tensor. \
                Got {}".format(mat.shape))

    mat = mat[..., :3, :3]
    shape = mat.shape

    with torch.no_grad():
        if check:
            e0 = mat @ mat.mT
            e1 = torch.eye(3, dtype=mat.dtype, device=mat.device)
            if not torch.allclose(e0, e1.expand_as(e0), rtol=rtol, atol=atol):
                raise ValueError("Input rotation matrices are not all orthogonal matrix")

            ones = torch.ones(shape[:-2], dtype=mat.dtype, device=mat.device)
            if not torch.allclose(torch.det(mat), ones, rtol=rtol, atol=atol):
                raise ValueError("Input rotation matrices' determinant are not all equal to 1")

    rmat_t = mat.mT

    mask_d2 = rmat_t[..., 2, 2] < atol

    mask_d0_d1 = rmat_t[..., 0, 0] > rmat_t[..., 1, 1]
    mask_d0_nd1 = rmat_t[..., 0, 0] < -rmat_t[..., 1, 1]

    t0 = 1 + rmat_t[..., 0, 0] - rmat_t[..., 1, 1] - rmat_t[..., 2, 2]
    q0 = torch.stack([rmat_t[..., 1, 2] - rmat_t[..., 2, 1],
                      t0, rmat_t[..., 0, 1] + rmat_t[..., 1, 0],
                      rmat_t[..., 2, 0] + rmat_t[..., 0, 2]], -1)
    t0_rep = t0.unsqueeze(-1).repeat((len(list(shape))-2)*(1,)+(4,))

    t1 = 1 - rmat_t[..., 0, 0] + rmat_t[..., 1, 1] - rmat_t[..., 2, 2]
    q1 = torch.stack([rmat_t[..., 2, 0] - rmat_t[..., 0, 2],
                      rmat_t[..., 0, 1] + rmat_t[..., 1, 0],
                      t1, rmat_t[..., 1, 2] + rmat_t[..., 2, 1]], -1)
    t1_rep = t1.unsqueeze(-1).repeat((len(list(shape))-2)*(1,)+(4,))

    t2 = 1 - rmat_t[..., 0, 0] - rmat_t[..., 1, 1] + rmat_t[..., 2, 2]
    q2 = torch.stack([rmat_t[..., 0, 1] - rmat_t[..., 1, 0],
                      rmat_t[..., 2, 0] + rmat_t[..., 0, 2],
                      rmat_t[..., 1, 2] + rmat_t[..., 2, 1], t2], -1)
    t2_rep = t2.unsqueeze(-1).repeat((len(list(shape))-2)*(1,)+(4,))

    t3 = 1 + rmat_t[..., 0, 0] + rmat_t[..., 1, 1] + rmat_t[..., 2, 2]
    q3 = torch.stack([t3, rmat_t[..., 1, 2] - rmat_t[..., 2, 1],
                      rmat_t[..., 2, 0] - rmat_t[..., 0, 2],
                      rmat_t[..., 0, 1] - rmat_t[..., 1, 0]], -1)
    t3_rep = t3.unsqueeze(-1).repeat((len(list(shape))-2)*(1,)+(4,))

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.unsqueeze(-1).type_as(q0)
    mask_c1 = mask_c1.unsqueeze(-1).type_as(q1)
    mask_c2 = mask_c2.unsqueeze(-1).type_as(q2)
    mask_c3 = mask_c3.unsqueeze(-1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= 2*torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa

    q = q.view(shape[:-2]+(4,))
    # wxyz -> xyzw
    q = q.index_select(-1, torch.tensor([1, 2, 3, 0], device=q.device))

    return pp.SO3(q)


def mat2SE3(mat, check=True, rtol=1e-5, atol=1e-5):
    r"""Convert batched rotation or transformation matrices to SE3Type LieTensor.

    Args:
        mat (Tensor): the batched matrices to convert. If input is of shape :obj:`(*, 3, 3)`, then
            translation will be filled with zero. For input with shape :obj:`(*, 3, 4)`, the last
            row will be treated as ``[0, 0, 0, 1]``.
        check (bool, optional): flag to check if the input is valid rotation matrices (orthogonal
            and with a determinant of one). Set to ``False`` if less computation is needed.
            Default: ``True``.
        rtol (float, optional): relative tolerance when check is enabled. Default: 1e-05
        atol (float, optional): absolute tolerance when check is enabled. Default: 1e-05

    Return:
        LieTensor: the converted SE3Type LieTensor.

    Shape:
        Input: :obj:`(*, 3, 3)` or :obj:`(*, 3, 4)` or :obj:`(*, 4, 4)`

        Output: :obj:`(*, 7)`

    Let the input be matrix :math:`\mathbf{T}`,  :math:`\mathbf{T}_i` represents each individual
    matrix in the batch. :math:`\mathbf{R}_i\in\mathbb{R}^{3\times 3}` be the top left 3x3 block
    matrix of :math:`\mathbf{T}_i`. :math:`\mathbf{T}^{m,n}_i` represents the :math:`m^{\mathrm{th}}`
    row and :math:`n^{\mathrm{th}}` column of :math:`\mathbf{T}_i`, :math:`m,n\geq 1`, then the
    translation and quaternion can be computed by:

    .. math::
        \left\{\begin{aligned}
        t^x_i &= \mathbf{T}^{1,4}_i\\
        t^y_i &= \mathbf{T}^{2,4}_i\\
        t^z_i &= \mathbf{T}^{3,4}_i\\
        q^x_i &= \mathrm{sign}(\mathbf{R}^{2,3}_i - \mathbf{R}^{3,2}_i) \frac{1}{2}
            \sqrt{1 + \mathbf{R}^{1,1}_i - \mathbf{R}^{2,2}_i - \mathbf{R}^{3,3}_i}\\
        q^y_i &= \mathrm{sign}(\mathbf{R}^{3,1}_i - \mathbf{R}^{1,3}_i) \frac{1}{2}
            \sqrt{1 - \mathbf{R}^{1,1}_i + \mathbf{R}^{2,2}_i - \mathbf{R}^{3,3}_i}\\
        q^z_i &= \mathrm{sign}(\mathbf{R}^{1,2}_i - \mathbf{R}^{2,1}_i) \frac{1}{2}
            \sqrt{1 - \mathbf{R}^{1,1}_i - \mathbf{R}^{2,2}_i + \mathbf{R}^{3,3}_i}\\
        q^w_i &= \frac{1}{2} \sqrt{1 + \mathbf{R}^{1,1}_i + \mathbf{R}^{2,2}_i + \mathbf{R}^{3,3}_i}
        \end{aligned}\right.,

    In summary, the output LieTensor should be of format:

    .. math::
        \textbf{y}_i = [t^x_i, t^y_i, t^z_i, q^x_i, q^y_i, q^z_i, q^w_i]

    Warning:
        Numerically, a transformation matrix is considered legal if:

        .. math::
            |{\rm det}(\mathbf{R}) - 1| \leq \texttt{atol} + \texttt{rtol}\times 1\\
            |\mathbf{RR}^{T} - \mathbf{I}| \leq \texttt{atol} + \texttt{rtol}\times \mathbf{I}

        where :math:`|\cdot |` is element-wise absolute function. When ``check`` is set to ``True``,
        illegal input will raise a ``ValueError``. Otherwise, no data validation is performed.
        Illegal input will output an irrelevant result, which likely contains ``nan``.

        For input with shape :obj:`(*, 4, 4)`, when ``check`` is set to ``True`` and the last row
        of the each individual matrix is not ``[0, 0, 0, 1]``, a warning will be triggered.
        Even though the last row is not used in the computation, it is worth noting that a matrix not
        satisfying this condition is not a valid transformation matrix.

    Examples:

        >>> input = torch.tensor([[0., -1., 0., 0.1],
        ...                       [1.,  0., 0., 0.2],
        ...                       [0.,  0., 1., 0.3],
        ...                       [0.,  0., 0.,  1.]])
        >>> pp.mat2SE3(input)
        SE3Type LieTensor:
        tensor([0.1000, 0.2000, 0.3000, 0.0000, 0.0000, 0.7071, 0.7071])

    Note:
        The individual matrix in a batch can be written as:

        .. math::
            \begin{bmatrix}
                    \mathbf{R}_{3\times3} & \mathbf{t}_{3\times1}\\
                    \textbf{0} & 1
            \end{bmatrix},

        where :math:`\mathbf{R}` is the rotation matrix. The translation vector :math:`\mathbf{t}` defines the
        displacement between the original position and the transformed position.


    See :meth:`pypose.SE3` for more details of the output LieTensor format.
    """
    if not torch.is_tensor(mat):
        mat = torch.tensor(mat)

    if len(mat.shape) < 2:
        raise ValueError("Input size must be at least 2 dimensions. Got {}".format(mat.shape))

    if not (mat.shape[-2:] == (3, 3) or mat.shape[-2:] == (3, 4) or mat.shape[-2:] == (4, 4)):
        raise ValueError("Input size must be a * x 3 x 3 or * x 3 x 4 or * x 4 x 4  tensor. \
                Got {}".format(mat.shape))

    shape = mat.shape
    if shape[-2:] == (4, 4) and check == True:
        zerosone = torch.tensor([0, 0, 0, 1], dtype=mat.dtype, device=mat.device)
        if not torch.allclose(mat[..., 3, :], zerosone.expand_as(mat[..., 3, :]), rtol=rtol, atol=atol):
            warnings.warn("input of shape 4x4 last rows are not all equal [0, 0, 0, 1]")

    q = mat2SO3(mat[..., :3, :3], check=check, rtol=rtol, atol=atol).tensor()
    if shape[-1] == 3:
        t = torch.zeros(shape[:-2] + (3,), dtype=mat.dtype, device=mat.device, requires_grad=mat.requires_grad)
    else:
        t = mat[..., :3, 3]
    vec = torch.cat([t, q], dim=-1)

    return pp.SE3(vec)
