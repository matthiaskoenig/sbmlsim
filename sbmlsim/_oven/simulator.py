#!/usr/bin/python
"""
Module for running/starting simulations.
Starts processes on the n_cores which listen for available simulations.
The odesim settings and parameters determine the actual odesim.
The simulator supports parallalization by putting different simulations
on different CPUs. 

-------------------------------------------------------------------------------------
How multiprocessing works, in a nutshell:

    Process() spawns (fork or similar on Unix-like systems) a copy of the 
    original program.
    The copy communicates with the original to figure out that 
        (a) it's a copy and 
        (b) it should go off and invoke the target= function (see below).
    At this point, the original and copy are now different and independent, 
    and can run simultaneously.

Since these are independent processes, they now have independent Global Interpreter Locks 
(in CPython) so both can use up to 100% of a CPU on a multi-cpu box, as long as they don't 
contend for other lower-level (OS) resources. That's the "multiprocessing" part.
-------------------------------------------------------------------------------------
"""

from __future__ import print_function, division

import fcntl
import logging
import multiprocessing
import os
import socket
import struct
import time

from django.db import transaction
from django.utils import timezone
from simapp.models import Task, Core, Simulation, SimulationStatus
from multiscale.simulate import solve


# TODO: provide the multicore functionality for all simulations
# TODO: use the roadrunner r.getInstanceID() & getInstanceCount() if multiple instances are running

def worker(cpu, lock, Nsim):
    """ Creates a worker for the cpu which listens for available simulations. """
    ip = get_ip_address()
    core, _ = Core.objects.get_or_create(ip=ip, cpu=cpu)
    
    while True:
        # update core time
        time_now = timezone.now()
        core.time = time_now 
        core.save()
    
        # Assign simulations within a multiprocessing lock
        lock.acquire()
        task, sims = assign_simulations(core, Nsim)
        lock.release()
        
        # Perform ODE integration
        if sims:
            print ('{:<20} <{}> {}'.format(core, task, sims))
            solve.run_simulations(sims, task)
        else:
            print ('{:<20} <No Simulations>'.format(core))
            time.sleep(10)


def assign_simulations(core, n_sim=1):
    """
    Assigns simulations(s) to core.
    Returns None if no simulation(s) could be assigned.
    The assignment has to be synchronized between the different cores.
    Use lock to handle the different cores on one cpu.
    """
    # get distinct tasks sorted by priority which have unassigned simulations 
    task_query = Task.objects.filter(simulation__status=SimulationStatus.UNASSIGNED).distinct('pk', 'priority').order_by('-priority')
    if task_query.exists():
        task = task_query[0]
        
        @transaction.atomic()
        def update_simulations():
            """ Updates the simulation status and times.
            Select the simulations for update with locked rows
            -> database lock for atomic transaction and to ensure that evey
            simulation is assigned to exactely on core.
            """
            sims = Simulation.objects.select_for_update().filter(task=task,
                                                                 status=SimulationStatus.UNASSIGNED)[0:n_sim]
            for sim in sims:
                sim.time_assign = timezone.now()
                sim.core = core
                sim.status = SimulationStatus.ASSIGNED
                sim.save()
            
            # update all at once
            # inner_q = Simulation.objects.select_for_update().filter(task=task,
            #               status=SimulationStatus.UNASSIGNED).values('pk')[0:n_sim]
            # sims = Simulation.objects.filter(pk__in=inner_q) 
            # sims.update(time_assign=timezone.now(),
            #             core=core, 
            #            status=ASSIGNED)
            return sims
        
        sims = update_simulations()    
        return task, sims
    else:
        return None, None


def get_ip_address(interface='eth0'):
    """ Returns the IP adress for the given computer. """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ip = socket.inet_ntoa(fcntl.ioctl(
                                          s.fileno(),
                                          0x8915,  # SIOCGIFADDR
                                          struct.pack('256s', interface[:15])
                                          )[20:24])
    except IOError:
        ip = "127.0.0.1"
        print("eth0 not found, using 127.0.0.1")
    return ip

    
def info(title):
    print(title)
    print('module name:', __name__)
    if hasattr(os, 'getppid'):  # only available on Unix
        print('parent process:', os.getppid())
    print('process id:', os.getpid())


def _sync_sbml_in_network():
    """
    Copies all SBML files to the server.
        run an operating system command
        call(["ls", "-l"])
    """
    # TODO: get the environment variables from the settings file
    # TODO: do direct synchronization to this computer, not to all computers
    # TODO: only make synchornizaton
    from subprocess import call
    call_command = [os.path.join(os.environ['MULTISCALE_GALACTOSE'], "syncDjangoSBML.sh")]
    logging.debug(str(call_command))
    call(call_command)

#####################################################################################

if __name__ == "__main__":     
    """
    Starting the simulations on the local computer.
    Call with --cpu option if not using 100% resources    
    
    TODO: implement communication via MIP. This becomes important
    in the case parallel integration within the cluster.
    """
    import django
    django.setup()
    from optparse import OptionParser
    import math
    parser = OptionParser()
    parser.add_option("-c", "--cpu", dest="cpu_load",
                      help="CPU load between 0 and 1, i.e. 0.5 uses half the n_cores")
    parser.add_option("-s", "--sync", dest="do_sync",
                      help="Sync models from DB Server (1 yes, 0 no)")
    (options, args) = parser.parse_args()

    # Syncronize SBML from server to computer
    do_sync = True
    if options.do_sync:
        do_sync = bool(options.do_sync)
    if do_sync:
        _sync_sbml_in_network()

    # Handle the number of cores
    print('#'*60)
    print('# Simulator')
    print('#'*60)
    n_cores = multiprocessing.cpu_count()
    print('CPUs: ', n_cores)
    if options.cpu_load:
        n_cores = int(math.floor(float(options.cpu_load)*n_cores))
    print('Used CPUs: ', n_cores)
    print('#'*60)
    
    n_sim = 40
    
    # Lock for syncronization between processes (but locks)
    lock = multiprocessing.Lock()
    # start processes on every cpu
    processes = []
    for cpu in range(n_cores):
        p = multiprocessing.Process(target=worker, args=(cpu, lock, n_sim))
        processes.append(p)
        p.start()
