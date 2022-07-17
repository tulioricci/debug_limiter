"""mirgecom driver for the Y0 demonstration.

Note: this example requires a *scaled* version of the Y0
grid. A working grid example is located here:
github.com:/illinois-ceesd/data@y0scaled
"""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import yaml
import logging
import sys
import numpy as np
import pyopencl as cl
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
from functools import partial

from arraycontext import thaw, freeze

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.dof_desc import DTAG_BOUNDARY
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer

from mirgecom.profiling import PyOpenCLProfilingArrayContext
from mirgecom.navierstokes import ns_operator
from mirgecom.simutil import (
    check_step,
    get_sim_timestep,
    generate_and_distribute_mesh,
    write_visfile,
    check_naninf_local,
    check_range_local,
    global_reduce,
    force_evaluation
)
from mirgecom.restart import (
    write_restart_file
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools

from grudge.shortcuts import compiled_lsrk45_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedFluidBoundary,
    OutflowBoundary,
    SymmetryBoundary
)
from mirgecom.fluid import make_conserved, species_mass_fraction_gradient
from mirgecom.initializers import PlanarDiscontinuity
from mirgecom.transport import SimpleTransport, MixtureAveragedTransport
from mirgecom.viscous import get_viscous_timestep, get_viscous_cfl#, diffusive_flux
from mirgecom.eos import PyrometheusMixture
from mirgecom.gas_model import GasModel, make_fluid_state
import cantera

from logpyle import IntervalTimer, set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info, logmgr_set_time, LogUserQuantity,
    set_sim_state
)
from pytools.obj_array import make_obj_array

from mirgecom.limiter import (
    limiter_liu_osher,
#    drop_order,
    cell_volume,
    neighbor_list,
    positivity_preserving_limiter
)

#######################################################################################

class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


h1 = logging.StreamHandler(sys.stdout)
f1 = SingleLevelFilter(logging.INFO, False)
h1.addFilter(f1)
root_logger = logging.getLogger()
root_logger.addHandler(h1)
h2 = logging.StreamHandler(sys.stderr)
f2 = SingleLevelFilter(logging.INFO, True)
h2.addFilter(f2)
root_logger.addHandler(h2)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


def sponge_func(cv, cv_ref, sigma):
    cv_ref = cv_ref.replace(species_mass=cv.species_mass)
    return sigma*(cv_ref - cv)


class InitSponge:
    r"""
    .. automethod:: __init__
    .. automethod:: __call__
    """
    def __init__(self, *, x_min=None, x_max=None, y_min=None, y_max=None, x_thickness=None, y_thickness=None, amplitude):
        r"""Initialize the sponge parameters.
        Parameters
        ----------
        x0: float
            sponge starting x location
        thickness: float
            sponge extent
        amplitude: float
            sponge strength modifier
        """

        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._x_thickness = x_thickness
        self._y_thickness = y_thickness
        self._amplitude = amplitude

    def __call__(self, x_vec, *, time=0.0):
        """Create the sponge intensity at locations *x_vec*.
        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        time: float
            Time at which solution is desired. The strength is (optionally)
            dependent on time
        """
        xpos = x_vec[0]
        ypos = x_vec[1]
        actx = xpos.array_context
        zeros = 0*xpos

        sponge = xpos*0.0

        if (self._x_max is not None):
          x0 = (self._x_max - self._x_thickness)
          dx = +((xpos - x0)/self._x_thickness)
          sponge = sponge + self._amplitude * actx.np.where(
              actx.np.greater(xpos, x0),
              #(zeros + 3.0*dx**2 - 2.0*dx**3),
              dx,
              zeros + 0.0
          )

        if (self._x_min is not None):
          x0 = (self._x_min + self._x_thickness)
          dx = -((xpos - x0)/self._x_thickness)
          sponge = sponge + self._amplitude * actx.np.where(
              actx.np.less(xpos, x0),
              #(zeros + 3.0*dx**2 - 2.0*dx**3),
              dx,
              zeros + 0.0
          )

        return sponge


@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, casename=None, lazy=False,
         rst_filename=None):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce

    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
                               mode="wo", mpi_comm=comm)

    cl_ctx = ctx_factory()

    if use_profiling:
        queue = cl.CommandQueue(
            cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))
    else:
        actx = actx_class(comm, queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
                force_device_scalars=True)

    # ~~~~~~~~~~~~~~~~~~

    rst_path = "restart_data_thetaVar_avg/"
    viz_path = "viz_data_thetaVar_avg/"
    vizname = viz_path+casename
    rst_pattern = rst_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    # default i/o frequencies
    nviz = 50
    nrestart = 1000 
    nhealth = 1
    nstatus = 100

    # default timestepping control
    integrator = "compiled_lsrk45"
    current_dt = 4.0e-9
    t_final = 4.0e-6

    niter = int(t_final/current_dt)
    
    # discretization and model control
    order = 1
    fuel = "C2H4"

######################################################

    local_dt = False
    constant_cfl = False
    current_cfl = 0.2

    dim = 2
    current_t = 0
    current_step = 0

##########################################################################################3

    def _compiled_stepper_wrapper(state, t, dt, rhs):
        return compiled_lsrk45_step(actx, state, t, dt, rhs)
        
    force_eval = True
    if integrator == "compiled_lsrk45":
        timestepper = _compiled_stepper_wrapper
        force_eval = False

    if rank == 0:
        print("\n#### Simulation control data: ####")
        print(f"\tnviz = {nviz}")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        if (constant_cfl == False):
          print(f"\tcurrent_dt = {current_dt}")
          print(f"\tt_final = {t_final}")
        else:
          print(f"\tconstant_cfl = {constant_cfl}")
          print(f"\tcurrent_cfl = {current_cfl}")
        print(f"\tniter = {niter}")
        print(f"\torder = {order}")
        print(f"\tTime integration = {integrator}")

##########################################################################

    # {{{  Set up initial state using Cantera

    # Use Cantera for initialization
    # -- Pick up a CTI for the thermochemistry config
    # --- Note: Users may add their own CTI file by dropping it into
    # ---       mirgecom/mechanisms alongside the other CTI files.
    from mirgecom.mechanisms import get_mechanism_cti
    mech_cti = get_mechanism_cti("uiuc")

    cantera_soln = cantera.Solution(phase_id="gas", source=mech_cti)
    nspecies = cantera_soln.n_species

    # Initial temperature, pressure, and mixture mole fractions are needed to
    # set up the initial state in Cantera.
    temp_unburned = 300.0
    temp_ignition = 2397.2584300840062
    # Parameters for calculating the amounts of fuel, oxidizer, and inert species
    equiv_ratio = 1.0
    ox_di_ratio = 0.21
    stoich_ratio = 3.0
    # Grab the array indices for the specific species, ethylene, oxygen, and nitrogen
    i_fu = cantera_soln.species_index("C2H4")
    i_ox = cantera_soln.species_index("O2")
    i_di = cantera_soln.species_index("N2")
    x = np.zeros(nspecies)
    # Set the species mole fractions according to our desired fuel/air mixture
    x[i_fu] = (ox_di_ratio*equiv_ratio)/(stoich_ratio+ox_di_ratio*equiv_ratio)
    x[i_ox] = stoich_ratio*x[i_fu]/equiv_ratio
    x[i_di] = (1.0-ox_di_ratio)*x[i_ox]/ox_di_ratio
    # Uncomment next line to make pylint fail when it can't find cantera.one_atm
    one_atm = cantera.one_atm  # pylint: disable=no-member
    # one_atm = 101325.0
    pres_unburned = one_atm

    # Let the user know about how Cantera is being initilized
    print(f"Input state (T,P,X) = ({temp_unburned}, {pres_unburned}, {x}")
    # Set Cantera internal gas temperature, pressure, and mole fractios
    cantera_soln.TPX = temp_unburned, one_atm, x
    # Pull temperature, total density, mass fractions, and pressure from Cantera
    # We need total density, and mass fractions to initialize the fluid/gas state.
    y_unburned = np.zeros(nspecies)
    can_t, rho_unburned, y_unburned = cantera_soln.TDY
    # *can_t*, *can_p* should not differ (significantly) from user's initial data,
    # but we want to ensure that we use exactly the same starting point as Cantera,
    # so we use Cantera's version of these data.

    # now find the conditions for the burned gas
    cantera_soln.TPX = temp_ignition, pres_unburned, x
    cantera_soln.equilibrate("TP")
    temp_burned, rho_burned, y_burned = cantera_soln.TDY
    pres_burned = cantera_soln.P

    # }}}

    # {{{ Create Pyrometheus thermochemistry object & EOS

    # Create a Pyrometheus EOS with the Cantera soln. Pyrometheus uses Cantera and
    # generates a set of methods to calculate chemothermomechanical properties and
    # states for this particular mechanism.
    from mirgecom.thermochemistry import get_thermochemistry_class_by_mechanism_name
    pyrometheus_mechanism = \
        get_thermochemistry_class_by_mechanism_name("uiuc_Sharp",
                                                        temperature_niter=3)(actx.np)

    eos = PyrometheusMixture(pyrometheus_mechanism, temperature_guess=temp_unburned)
    species_names = pyrometheus_mechanism.species_names
    
    # {{{ Initialize transport model
#    mu = 2.0e-5
#    kappa = mu*1000/0.71
#    spec_diffusivity = 1.0e-4 * np.ones(nspecies)
#    transport_model = SimpleTransport(viscosity=mu,
#                                      thermal_conductivity=kappa,
#                                      species_diffusivity=spec_diffusivity)
    transport_model = MixtureAveragedTransport(pyrometheus_mechanism)
    # }}}    
    
    gas_model = GasModel(eos=eos, transport=transport_model)

    # }}}
    
    tseed = temp_unburned

    print(f"Pyrometheus mechanism species names {species_names}")
    print(f"Unburned (T,P,Y) = ({temp_unburned}, {pres_unburned}, {y_unburned}")
    print(f"Burned (T,P,Y) = ({temp_burned}, {pres_burned}, {y_burned}")

    def _get_temperature_update(cv, temperature):
        y = cv.species_mass_fractions
        e = gas_model.eos.internal_energy(cv) / cv.mass
        return actx.np.abs(
            pyrometheus_mechanism.get_temperature_update_energy(e, temperature, y))

    def _get_fluid_state(cv, temp_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model,
                                temperature_seed=temp_seed)

    get_temperature_update = actx.compile(_get_temperature_update)
    get_fluid_state = actx.compile(_get_fluid_state)

#####################################################################################

    restart_step = None
    if restart_file is None:  
        sys.exit()      
        #local_mesh, global_nelements = generate_and_distribute_mesh(
        #    comm, get_mesh(dim=dim))
        #local_nelements = local_mesh.nelements

    else:  # Restart
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_file)
        restart_step = restart_data["step"]
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])

        assert comm.Get_size() == restart_data["num_parts"]

    from grudge.dof_desc import DISCR_TAG_QUAD
    from mirgecom.discretization import create_discretization_collection
    discr = create_discretization_collection(actx, local_mesh, order, comm)
    nodes = actx.thaw(discr.nodes())

    quadrature_tag = None

    def vol_min_loc(x):
        from grudge.op import nodal_min_loc
        return actx.to_numpy(nodal_min_loc(discr, "vol", x))[()]

    def vol_max_loc(x):
        from grudge.op import nodal_max_loc
        return actx.to_numpy(nodal_max_loc(discr, "vol", x))[()]

    def vol_min(x):
        from grudge.op import nodal_min
        return actx.to_numpy(nodal_min(discr, "vol", x))[()]

    def vol_max(x):
        from grudge.op import nodal_max
        return actx.to_numpy(nodal_max(discr, "vol", x))[()]

    from grudge.dt_utils import characteristic_lengthscales
    length_scales = characteristic_lengthscales(actx, discr)
    h_min = vol_min(length_scales)
    h_max = vol_max(length_scales)

    if rank == 0:
        print("----- Discretization info ----")
        print(f"Discr: {nodes.shape=}, {order=}, {h_min=}, {h_max=}")
    for i in range(nparts):
        if rank == i:
            print(f"{rank=},{local_nelements=},{global_nelements=}")
        comm.Barrier()
    
############################################################################

#    casename = f"{casename}-d{dim}p{order}e{global_nelements}n{nparts}"
    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
                               mode="wo", mpi_comm=comm)
                               
    vis_timer = None
    log_cfl = LogUserQuantity(name="cfl", value=current_cfl)

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_set_time(logmgr, current_step, current_t)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            #("dt.max", "dt: {value:1.6e} s, "),
            ("t_sim.max", "sim time: {value:1.6e} s, "),
            ("t_step.max", "------- step walltime: {value:6g} s\n")
            ])

        #logmgr_add_device_memory_usage(logmgr, queue)
        try:
            logmgr.add_watches(["memory_usage_python.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

#################################################################

    vel_bnd = np.zeros(shape=(dim,))

    def _flow_bnd(nodes, eos, side):
        pressure = one_atm + 0*nodes[0]
        if side == 'burned':
            temperature = temp_burned + 0*nodes[0]
            y = make_obj_array([y_burned[i] + 0*nodes[0]
                            for i in range(nspecies)])
        else:
            temperature = temp_unburned + 0*nodes[0]
            y = make_obj_array([y_unburned[i] + 0*nodes[0]
                            for i in range(nspecies)])
        velocity = vel_bnd + 0*nodes[0]

        mass = eos.get_density(pressure, temperature, y)
        specmass = mass * y
        mom = mass * velocity
        internal_energy = eos.get_internal_energy(temperature=temperature,
                                                   species_mass_fractions=y)
        kinetic_energy = 0.5 * np.dot(velocity, velocity)
        energy = mass * (internal_energy + kinetic_energy)

        return make_conserved(dim=2, mass=mass, energy=energy,
                              momentum=mom, species_mass=specmass)


    inflow_btag = DTAG_BOUNDARY("inlet")
    inflow_bnd_discr = discr.discr_from_dd(inflow_btag)
    inflow_nodes = actx.thaw(inflow_bnd_discr.nodes())
    inflow_bnd_cond = force_evaluation(
                       actx, _flow_bnd(inflow_nodes, eos, 'unburned'))
    inflow_state = get_fluid_state(cv=inflow_bnd_cond, temp_seed=temp_unburned)
    inflow_state = force_evaluation(actx, inflow_state)

    def _inflow_bnd_state_func(**kwargs):
        return inflow_state

    outflow_btag = DTAG_BOUNDARY("outlet")
    outflow_bnd_discr = discr.discr_from_dd(outflow_btag)
    outflow_nodes = actx.thaw(outflow_bnd_discr.nodes())
    outflow_bnd_cond = force_evaluation(
                       actx, _flow_bnd(outflow_nodes, eos, 'burned'))
    outflow_state = get_fluid_state(cv=outflow_bnd_cond, temp_seed=temp_burned)
    outflow_state = force_evaluation(actx, outflow_state)

    def _outflow_bnd_state_func(**kwargs):
        return outflow_state


    inflow_boundary = PrescribedFluidBoundary(boundary_state_func=_inflow_bnd_state_func)
    outflow_boundary = PrescribedFluidBoundary(boundary_state_func=_outflow_bnd_state_func)

    boundaries = {DTAG_BOUNDARY("inlet"): inflow_boundary,
                  DTAG_BOUNDARY("outlet"): outflow_boundary               
                  }

#################################################################

    if restart_file is None:
        sys.exit()
#        if rank == 0:
#            logging.info("Initializing soln.")
#        current_cv = force_evaluation(actx, flow_init(x_vec=nodes, eos=eos))
#        tseed = nodes[0]*0.0 + tseed
    else:
        if local_dt:
            current_t = restart_data["step"]
        else:
            current_t = restart_data["t"]
        current_step = restart_step

        if restart_order != order:
            sys.exit()
#            restart_discr = EagerDGDiscretization(
#                actx,
#                local_mesh,
#                order=restart_order,
#                mpi_communicator=comm)
#            from meshmode.discretization.connection import make_same_mesh_connection
#            connection = make_same_mesh_connection(
#                actx,
#                discr.discr_from_dd("vol"),
#                restart_discr.discr_from_dd("vol"))

#            current_cv = connection(restart_data["cv"])
#            tseed = connection(restart_data["temperature_seed"])
        else:
            current_cv = restart_data["cv"]
            tseed = restart_data["temperature_seed"]

        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)

    current_state = get_fluid_state(current_cv, tseed)
    current_state = force_evaluation(actx, current_state)

#####################################################################################

    # initialize the sponge field
    sponge_x_thickness = 0.015
    sponge_amp = 1000.0

    xMaxLoc = vol_min(nodes[0])
    xMinLoc = vol_max(nodes[0])
        
    sponge_init = InitSponge(x_max=xMaxLoc,
                             x_min=xMinLoc,
                             x_thickness=sponge_x_thickness,
                             amplitude=sponge_amp)

    sponge_sigma = sponge_init(x_vec=nodes)
    
#    ref_cv = ref_state(x_vec=nodes, eos=eos, time=0.)
    ref_cv = 1.0*current_cv

####################################################################################

    visualizer = make_visualizer(discr)

    initname = "flame1D"
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final,
                                     nstatus=nstatus, nviz=nviz,
                                     cfl=current_cfl,
                                     constant_cfl=constant_cfl,
                                     initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

#########################################################################

#    def get_fluid_state(cv, temperature_seed):
#        return make_fluid_state(cv=cv, gas_model=gas_model,
#                                temperature_seed=temperature_seed)

#    create_fluid_state = actx.compile(get_fluid_state)

#    def get_temperature_update(cv, temperature):
#        y = cv.species_mass_fractions
#        e = eos.internal_energy(cv) / cv.mass
#        return make_obj_array(
#            [pyrometheus_mechanism.get_temperature_update_energy(e, temperature, y)]
#        )

#    compute_temperature_update = actx.compile(get_temperature_update)

##################################################################

    def get_production_rates(cv, temperature):
        return make_obj_array([eos.get_production_rates(cv, temperature)])
    compute_production_rates = actx.compile(get_production_rates)

    def my_write_viz(step, t, state, dt=None,
                     ns_rhs=None, chem_sources=None,
                     grad_cv=None, grad_t=None, grad_y=None,
                     ref_cv=None, sources=None):

        grad_P = gas_model.eos.gas_const(state.cv)*(
            state.dv.temperature*grad_cv.mass + state.cv.mass*grad_t )
        grad_v = velocity_gradient(state.cv, grad_cv)

        reaction_rates, = compute_production_rates(state.cv, state.temperature)
        viz_fields = [("CV_rho", state.cv.mass),
                      ("CV_rhoU", state.cv.momentum),
                      ("CV_rhoE", state.cv.energy),
                      ("DV_P", state.pressure),
                      ("DV_T", state.temperature),
                      ("DV_U", state.velocity[0]),
                      ("DV_V", state.velocity[1]),
                      ("grad_U", grad_v[0]),
                      ("grad_V", grad_v[1]),
                      ("grad_mass", grad_cv.mass),
                      ("grad_T", grad_t),
                      ("grad_P", grad_P),
                      ("sponge", sponge_sigma),
                      ("reaction_rates", reaction_rates),
#                      ("dt" if constant_cfl else "cfl", ts_field)
#                      ("dt", dt)
                      ]
                      
        # species mass fractions
        viz_fields.extend(
            ("Y_"+species_names[i], state.cv.species_mass_fractions[i])
                for i in range(nspecies))
        viz_fields.extend(
            ("grad_Y_"+species_names[i], grad_y[i]) for i in range(nspecies))                
        viz_fields.extend([
            ("TV_viscosity", state.tv.viscosity),
            ("TV_thermal_conductivity", state.tv.thermal_conductivity)
            ])        
        viz_fields.extend(
            ("TV_"+species_names[i], state.tv.species_diffusivity[i])
                for i in range(nspecies)
            )
          
        print('Writing solution file...')
        write_visfile(discr, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True)

    def my_write_restart(step, t, cv, tseed):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != rst_filename:
            rst_data = {
                "local_mesh": local_mesh,
                "cv": cv,
                "temperature_seed": tseed,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

##################################################################

    def my_health_check(cv, dv):
        health_error = False
        pressure = force_evaluation(actx, dv.pressure)
        temperature = force_evaluation(actx, dv.temperature)

        if global_reduce(check_naninf_local(discr, "vol", pressure), op="lor"):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if global_reduce(check_naninf_local(discr, "vol", temperature), op="lor"):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in temperature data.")

        return health_error

###############################################################################

#    neighbors = neighbor_list(dim, local_mesh)
#    cell_size = cell_volume(discr, nodes[0])
#    #radial_dist = actx.np.sqrt(nodes[0]**2 + nodes[1]**2)
#    #apply_limiter = actx.np.greater(radial_dist, 0.0)
#    apply_limiter = actx.np.greater(nodes[0], 0.0)
#    limiting_function = limiter_liu_osher #drop_order
#    def limiter(cv,dv=None):
#    
#        #if use_limiter == False: return cv
#    
#        mass_lim = actx.np.where( apply_limiter,
#                    limiting_function(discr, neighbors, cell_size, cv.mass),
#                    cv.mass )
#        velc_lim = make_obj_array([
#                    actx.np.where( apply_limiter,
#                    limiting_function(discr, neighbors, cell_size, cv.velocity[i]),
#                    cv.velocity[i] )
#                                                       for i in range(dim)])

#        spec_lim = make_obj_array([
#                   actx.np.where(apply_limiter,
#                       limiting_function(discr, neighbors, cell_size,
#                                         cv.species_mass_fractions[i]),
#                       cv.species_mass_fractions[i])
#                   for i in range(nspecies)])
#               
##        pressure = (cv.energy - gas_model.eos.kinetic_energy(cv))*(gas_model.eos.gamma() - 1.0)
##        pres_lim = actx.np.where( apply_limiter,
##                     drop_order(discr, neighbors, cell_size, pressure), pressure)

##        ener_lim = pres_lim/(gas_model.eos.gamma() - 1.0) + \
##                       0.5 * mass_lim * np.dot(velc_lim, velc_lim)

##        temp_limited = actx.np.where( apply_limiter,
##                     drop_order(discr, neighbors, cell_size, dv.temperature),
##                     dv.temperature)
##        energy_lim = mass_lim * ( gas_model.eos.get_internal_energy(temp_limited,
##                                                    species_mass_fractions=spec_lim)
##                     + 0.5 * np.dot(velc_lim, velc_lim) )

##        return make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
##                       momentum=velc_lim*mass_lim, species_mass=mass_lim*spec_lim )

#        int_energy = cv.energy - 0.5*cv.mass*np.dot(cv.velocity,cv.velocity)
#        
#        energy_lim = (
#                         limiting_function(discr, neighbors, cell_size, int_energy)
#                         + 0.5*mass_lim*np.dot(velc_lim,velc_lim)
#                     )

##        energy_lim = actx.np.where( apply_limiter,
##                     limiting_function(discr, neighbors, cell_size, cv.energy),
##                     cv.energy)

#        return make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
#                       momentum=velc_lim*mass_lim, species_mass=mass_lim*spec_lim )


    neighbors = neighbor_list(dim, local_mesh)
    cell_size = cell_volume(discr, nodes[0])
    apply_limiter = actx.np.greater(nodes[0], 0.0)
    limiting_function = positivity_preserving_limiter
    def limiter(cv,dv=None,temp=None):

        spec_lim = make_obj_array([
                   actx.np.where(apply_limiter,
                       limiting_function(discr, cell_size,
                                         cv.species_mass_fractions[i]),
                       cv.species_mass_fractions[i])
                   for i in range(nspecies)])

        kin_energy = 0.5*np.dot(cv.velocity,cv.velocity)
        int_energy = cv.energy - cv.mass*kin_energy
        
        energy_lim = cv.mass*(
            gas_model.eos.get_internal_energy(temp, species_mass_fractions=spec_lim)
            + kin_energy
        )

        return make_conserved(dim=dim, mass=cv.mass, energy=energy_lim,
                       momentum=cv.momentum, species_mass=cv.mass*spec_lim )

################################################################################################

    import os
    #from mirgecom.axisymmetric_sources import axisymmetry_source_terms
    from mirgecom.fluid import velocity_gradient, species_mass_fraction_gradient    
    def my_pre_step(step, t, dt, state):
        
        if logmgr:
            logmgr.tick_before()

        cv, tseed = state
        cv = force_evaluation(actx, cv)
        tseed = force_evaluation(actx, tseed)

#        fluid_state = get_fluid_state(cv, tseed)
        fluid_state = get_fluid_state(limiter(cv=cv,temp=tseed), tseed)
        fluid_state = force_evaluation(actx, fluid_state)
        
        cv = fluid_state.cv
        dv = fluid_state.dv

        if constant_cfl:
            dt = get_sim_timestep(discr, fluid_state, t, dt, current_cfl,
                                           t_final, constant_cfl, local_dt)     
        if local_dt:
            t = force_evaluation(actx, t)
            dt = force_evaluation(actx, get_sim_timestep(discr, fluid_state,
                 cfl=current_cfl, constant_cfl=constant_cfl, local_dt=local_dt))
             
        try:
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)
            do_status = check_step(step=step, interval=nstatus)
        
            file_exists = os.path.exists('write_restart')
            if file_exists:
              os.system('rm write_restart')
              do_restart = True

            if do_health:
                health_errors = global_reduce(my_health_check(cv, dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart and step > 0:
                my_write_restart(step=step, t=t, cv=cv, tseed=tseed)

            if do_viz:            
                ns_rhs, grad_cv, grad_t = \
                    ns_operator(discr, state=fluid_state, time=t,
                                boundaries=boundaries,
                                gas_model=gas_model,
                                return_gradients=True,
                                quadrature_tag=quadrature_tag)
                                
                grad_y = species_mass_fraction_gradient(cv, grad_cv)
                chem_rhs = eos.get_species_source_terms(cv,
                                                  fluid_state.temperature)
                                                                      
                sponge = sponge_func(cv=cv, cv_ref=ref_cv, sigma=sponge_sigma)

                ns_rhs = ns_rhs + chem_rhs + sponge
                
                my_write_viz(step=step, t=t, state=fluid_state,
                             ref_cv=None, dt=dt,
                             ns_rhs=None,
                             grad_cv=grad_cv, grad_t=grad_t, grad_y=grad_y,
                             sources=None)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, state=fluid_state)
            raise

        return state, dt

    def my_post_step(step, t, dt, state):
        cv, tseed = state
        fluid_state = get_fluid_state(cv, tseed)

        if logmgr:
            if local_dt:    
                set_dt(logmgr, 1.0)
            else:
                set_dt(logmgr, dt)            
            logmgr.tick_after()

        return make_obj_array([cv, fluid_state.temperature]), dt

    def my_rhs(t, state):
        cv, tseed = state
        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
                                   temperature_seed=tseed)

        ns_rhs, grad_cv, grad_t = (
            ns_operator(discr, state=fluid_state,
                        time=t, boundaries=boundaries,
                        gas_model=gas_model,
                        return_gradients=True,
                        quadrature_tag=quadrature_tag)
        )

        chem_rhs = eos.get_species_source_terms(cv, fluid_state.temperature)
                                                              
        sponge = sponge_func(cv=cv, cv_ref=ref_cv, sigma=sponge_sigma)
        
        #grad_y = species_mass_fraction_gradient(cv, grad_cv)
        #sources = axisymmetry_source_terms(
        #        actx, discr, fluid_state, grad_cv, grad_t, grad_y)
                
        cv_rhs = ns_rhs + chem_rhs + sponge #+ sources
        
        return make_obj_array([cv_rhs, 0*tseed])

    current_dt = get_sim_timestep(discr, current_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)

    if rank == 0:
        logging.info("Stepping.")

    (current_step, current_t, stepper_state) = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      state=make_obj_array([current_state.cv, tseed]),
                      dt=current_dt, t_final=t_final, t=current_t,
                      istep=current_step)
    current_cv, tseed = stepper_state
    current_state = make_fluid_state(current_cv, gas_model, tseed)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    my_write_viz(step=current_step, t=current_t, #dt=current_dt,
                 state=current_state)
    my_write_restart(step=current_step, t=current_t, cv=current_state.cv,
                     temperature_seed=tseed)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    exit()


if __name__ == "__main__":
    import sys
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(description="MIRGE-Com 1D Flame Driver")
    parser.add_argument("-r", "--restart_file",  type=ascii,
                        dest="restart_file", nargs="?", action="store",
                        help="simulation restart file")
    parser.add_argument("-i", "--input_file",  type=ascii,
                        dest="input_file", nargs="?", action="store",
                        help="simulation config file")
    parser.add_argument("-c", "--casename",  type=ascii,
                        dest="casename", nargs="?", action="store",
                        help="simulation case name")
    parser.add_argument("--profile", action="store_true", default=False,
        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=True,
        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
        help="enable lazy evaluation [OFF]")

    args = parser.parse_args()

    # for writing output
    casename = "flame1D"
    if(args.casename):
        print(f"Custom casename {args.casename}")
        casename = (args.casename).replace("'", "")
    else:
        print(f"Default casename {casename}")

    restart_file = None
    if args.restart_file:
        restart_file = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_file}")

    input_file = None
    if(args.input_file):
        input_file = (args.input_file).replace("'", "")
        print(f"Reading user input from {args.input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=args.lazy, distributed=True)

    main(actx_class, use_logmgr=args.log, 
         use_profiling=args.profile, casename=casename,
         lazy=args.lazy, rst_filename=restart_file)
