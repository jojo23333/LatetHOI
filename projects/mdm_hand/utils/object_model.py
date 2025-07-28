import numpy as np

import torch
import torch.nn as nn
from smplx.lbs import batch_rodrigues
from collections import namedtuple

model_output = namedtuple('output', ['vertices', 'global_orient', 'transl'])

class ObjectModel(nn.Module):

    def __init__(self,
                 v_template,
                 faces=None,
                 dtype=torch.float32):
        ''' 3D rigid object model

                Parameters
                ----------
                v_template: np.array Vx3, dtype = np.float32
                    The vertices of the object
                batch_size: int, N, optional
                    The batch size used for creating the model variables

                dtype: torch.dtype
                    The data type for the created variables
            '''

        super(ObjectModel, self).__init__()


        self.dtype = dtype

        # Mean template vertices
        self.register_buffer('v_template', torch.from_numpy(v_template[np.newaxis]).to(dtype))
        self.faces = faces.astype(np.int32)


    def forward(self, global_orient, transl, v_template=None, **kwargs):

        ''' Forward pass for the object model

        Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)

            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            v_template: torch.tensor, optional, shape BxVx3
                The new object vertices to overwrite the default vertices

        Returns
            -------
                output: ModelOutput
                A named tuple of type `ModelOutput`
        '''
        if v_template is None:
            v_template = self.v_template
        bsz = global_orient.shape[0]

        rot_mats = batch_rodrigues(global_orient.view(-1, 3)).view([bsz, 3, 3])

        vertices = torch.matmul(v_template, rot_mats) + transl.unsqueeze(dim=1)

        output = model_output(vertices=vertices,
                              global_orient=global_orient,
                              transl=transl)

        return output
