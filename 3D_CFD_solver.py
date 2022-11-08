# Generic imports
import os
import sys
import math
import numpy               as np
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
# Custom imports
from dolfin    import *
from mshr      import *

def solve_flow(*args, **kwargs):
    # Handle optional arguments   
    final_time = 1.0      # Final time
    cfl        = 0.5      # Courant number
    xmin       = -0.04   # min x-Domain
    xmax       = 0.04    # max x-Domain
    ymin       = -0.03   # min y-Domain
    ymax       = 0.03    # max y-Domain
    zmin       = 0.0      # min z-Domain
    zmax       = 0.01     # max z-Domain
    min_x      = -0.03   
    max_x      = 0.015
    min_y      = -0.015
    max_y      = 0.015
    
    # Parameters
    v_in           = 0.1 # average velocity at inlet
    inflow_profile = ('4.0*1.5*'+str(v_in)+'*x[2]*('+str(zmax)+' - x[2]) / pow('+str(zmax)+', 2)', '0', '0') 
    
    Dh        = 4*(ymax-ymin)*(zmax-zmin)/(2*((ymax-ymin)+(zmax-zmin))) # hydralic diameter
    mu        = 1.8e-5 # dynamic viscosity
    rho       = 1.225  # density
    nu        = mu/rho # kinematic viscosity
    Re        = rho*Dh*v_in/mu # Reynolds number
    print('Re = ',Re)
    cp        = 1006.0 # specific heat
    k         = 0.024 # thermal conductivity
    D         = k/(rho*cp) # thermal diffusivity
    Pr        = nu/D # Prandtl number
    print('Pr = ',Pr)
    t_in      = 300.0 # inlet temperature
    t_init    = 300.0 # initial temperature
    t_obs     = 450.0 # cylinder wall temperature

    # Create subdomain containing fin boundary
    class Obstacle_fin(SubDomain):
        def inside(self, x, on_boundary):
            return (on_boundary and
                    (min_x < x[0] < max_x) and
                    (min_y < x[1] < max_y) and
                    (zmin < x[2] < zmax))

    # Sub domain for Periodic boundary condition
    class PeriodicBoundary(SubDomain):
        # bottom boundary is "target domain" G
        def inside(self, x, on_boundary):
            return bool((x[1]-ymin) < DOLFIN_EPS and (x[1]-ymin) > -DOLFIN_EPS and on_boundary)

        # map top boundary (H) to bottom boundary (G)
        # map coordinates x in H to coordinates y in G
        def map(self, x, y):
            y[1] = x[1] - (ymax-ymin) # the dimension along y axis
            y[0] = x[0]
            y[2] = x[2]

    # Create periodic boundary condition
    pbc = PeriodicBoundary()
   
    # Import mesh
    mesh_file = "shape.xml"
    mesh = Mesh(mesh_file)
    h    = mesh.hmin()
    print('h = ',h)

    # Compute timestep and max nb of steps
    dt        = cfl*h/v_in
    timestep  = dt
    T         = final_time
    num_steps = math.floor(T/dt)
    print('num_steps = ',num_steps)

    # Define function spaces
    V = VectorFunctionSpace(mesh, 'P', 2, constrained_domain=pbc)
    Q = FunctionSpace      (mesh, 'P', 1, constrained_domain=pbc)
    C = FunctionSpace(mesh, 'P', 1, constrained_domain=pbc) # for the temperature

    # Define boundaries
    inflow  = 'near(x[0], '+str(xmin)+')'
    outflow = 'near(x[0], '+str(xmax)+')'
    walls   = 'near(x[2], '+str(zmin)+') || near(x[2], '+str(zmax)+')'
    shape_fins   = 'on_boundary && x[0]>('+str(min_x)+') && x[0]<'+str(max_x)+' && x[1]>('+str(min_y)+') && x[1]<('+str(max_y)+') && x[2]>('+str(zmin)+') && x[2]<('+str(zmax)+')'
    
    # Define boundary conditions
    bcu_inflow  = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
    bcu_fins     = DirichletBC(V, Constant((0.0, 0.0, 0.0)),  shape_fins)
    bcu_walls   = DirichletBC(V, Constant((0.0 , 0.0, 0.0)), walls)
    bcp_outflow = DirichletBC(Q, Constant(0.0),         outflow)
    bcu         = [bcu_inflow, bcu_fins, bcu_walls]
    bcp         = [bcp_outflow]

    bct_fins = DirichletBC(C, Constant(t_obs), shape_fins) # constant temperate on obstacle
    bct_in = DirichletBC(C, Constant(t_in), inflow) # constant temperate on inlet
    bct_walls = DirichletBC(C, Constant(t_obs), walls) # constant temperature on walls
    bc = [bct_in, bct_fins, bct_walls]

    # Define trial and test functions
    u, v  = TrialFunction(V), TestFunction(V)
    p, q  = TrialFunction(Q), TestFunction(Q)
    c, cv = TrialFunction(C), TestFunction(C)

    # Define functions for solutions at previous and current time steps
    u_n, u_, u_m = Function(V), Function(V), Function(V)
    p_n, p_      = Function(Q), Function(Q)
    c_n = interpolate(Expression(f"{t_init}", degree=2), C) # initial conditions for temperature

    # Define expressions and constants used in variational forms
    n   = FacetNormal(mesh)
    f   = Constant((0, 0, 0))

    # Set BDF2 coefficients for 1st timestep
    bdf2_a = Constant( 1.0)
    bdf2_b = Constant(-1.0)
    bdf2_c = Constant( 0.0)
    
    # Define symmetric gradient
    def epsilon(u):
        return sym(nabla_grad(u))

    # Define stress tensor
    def sigma(u, p):
        return 2.0*mu*epsilon(u) - p*Identity(len(u))

    # Define variational problem for step 1 Using BDF2 scheme
    F1 = rho*dot((bdf2_a*u + bdf2_b*u_n + bdf2_c*u_m)/dt, v)*dx + rho*dot(dot(u_n, nabla_grad(u)), v)*dx + inner(sigma(u, p_n), epsilon(v))*dx + dot(p_n*n, v)*ds - dot(mu*nabla_grad(u)*n, v)*ds - dot(f, v)*dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    # Define variational problem for step 2
    a2 = dot(nabla_grad(p),   nabla_grad(q))*dx
    L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (bdf2_a/dt)*div(u_)*q*dx

    # Define variational problem for step 3
    a3 = dot(u,  v)*dx
    L3 = dot(u_, v)*dx - (dt/bdf2_a)*dot(nabla_grad(p_ - p_n), v)*dx

    # Define variational problem for diffusion-convection equation
    F = ((c - c_n) / dt)*cv*dx + dot(u_, grad(c))*cv*dx + D*dot(grad(c), grad(cv))*dx
    a4 = lhs(F)
    L4 = rhs(F)

    # Assemble A3 matrix since it will not need re-assembly
    A3 = assemble(a3)
    
    ppp = []
    htt = []

    vtkfile_u = File('output_u/output_u.pvd')
    vtkfile_p = File('output_p/output_p.pvd')
    vtkfile_t = File('output_t/output_t.pvd')

    ########################################
    # Time-stepping loop
    ########################################
    try:
        t     = 0.0
        c = Function(C)
        set_log_active(False)

        for m in tqdm(range(num_steps)):
            # Update current time
            t += timestep

            # Step 1: Tentative velocity step
            A1 = assemble(a1)
            b1 = assemble(L1)
            [bc.apply(A1) for bc in bcu]
            [bc.apply(b1) for bc in bcu]
            solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg') #gmres

            # Step 2: Pressure correction step
            A2 = assemble(a2)
            b2 = assemble(L2)
            [bc.apply(A2) for bc in bcp]
            [bc.apply(b2) for bc in bcp]
            solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')

            # Step 3: Velocity correction step
            b3 = assemble(L3)
            solve(A3, u_.vector(), b3, 'cg',       'sor')

            solve(a4 == L4, c, bc)

            # Update previous solution
            u_m.assign(u_n)
            u_n.assign(u_)
            p_n.assign(p_)
            c_n.assign(c)

            # Set BDF2 coefficients for m>1
            bdf2_a.assign(Constant( 3.0/2.0))
            bdf2_b.assign(Constant(-2.0))
            bdf2_c.assign(Constant( 1.0/2.0))

            if m%3 ==0:
                vtkfile_t  << (c, t)
                vtkfile_p  << (p_,t)
                vtkfile_u  << (u_,t)


            if (m > 0.9*num_steps):

                tol = 0.00001  # avoid hitting points outside the domain
                y = np.linspace(ymin + tol, ymax - tol, 100) # y value of points 
                z = np.linspace(zmin + tol, zmax - tol, 100) # z value of points 
                points_inlet = [(xmin, y_, z_) for y_ in y for z_ in z] # points in the inlet
                points_outlet = [(xmax, y_, z_) for y_ in y for z_ in z] # points in the outlet

                p_inlet = np.array([p_(point) for point in points_inlet]) 
                p_outlet = np.array([p_(point) for point in points_outlet]) 
                
                u_inlet = np.array([u_(point) for point in points_inlet])   
                u_outlet = np.array([u_(point) for point in points_outlet])
                ux_inlet = np.delete(u_inlet, [1,2],1)
                ux_outlet = np.delete(u_outlet, [1,2],1)
                ux_in_average = np.average(ux_inlet) 
                ux_out_average = np.average(ux_outlet)
                print('ux_in_average =',ux_in_average)
                print('ux_out_average =',ux_out_average)
                
                t_inlet = np.array([c(point) for point in points_inlet]) 
                t_outlet = np.array([c(point) for point in points_outlet])
                t_in_average = np.average(t_inlet)
                t_out_average = np.average(t_outlet)
                print('t_in_average =',t_in_average)
                print('t_out_average =',t_out_average)
                
                p_average = np.average(p_inlet)
                
                c_average_inlet = np.average(np.multiply(t_in,ux_inlet))
                c_average_outlet = np.average(np.multiply(t_outlet,ux_outlet))

                ppp.append (0.0 - p_average)
                htt.append(rho*cp*(ymax-ymin)*(zmax-zmin)*(c_average_outlet - c_average_inlet))

        def ave(name):
            return sum(name)/len(name)

        pressure_drp =ave(ppp)
        heat = ave(htt)
        print('pressure_drop = ',pressure_drp)
        print('heat_transfer = ',heat)

    except Exception as exc:
        print(exc)
        return 0.0, 0.0, False

    # return reward components
    return pressure_drp, heat, True
    
    
if __name__ == "__main__":
    solve_flow()
    
    
