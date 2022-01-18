
from shapes_utils import *
from meshes_utils import *


n_pts          = 5
n_sampling_pts = 10
radius         = 0.5*np.ones([n_pts])
edgy           = 0.5*np.ones([n_pts])
plot_pts       = True
filename       = '3359.csv'
cylinder       = False


shape = Shape(filename,None,n_pts,n_sampling_pts,radius,edgy)

shape.read_csv(filename)
shape.generate()



shape.mesh(mesh_domain = True,
           shape_h     = 1.0,
           domain_h    = 1.0,
           xmin        =-10.0,
           xmax        = 20.0,
           ymin        =-10.0,
           ymax        = 10.0,
           mesh_format = 'mesh')
shape.generate_image(plot_pts=plot_pts)
shape.write_csv()
