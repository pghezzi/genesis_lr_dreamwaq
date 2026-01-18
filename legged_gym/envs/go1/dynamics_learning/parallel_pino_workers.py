# Helper functions used to parallelize calls to the dynamics functions in Pinocchio with shared input/output memory buffers and reused-workers.
import numpy as np
import torch
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import pinocchio as pn
import time
import os
import queue 

class SharedTensors:
    """
    Allocates shared memory buffers used by both the main process and worker processes.
    """

    def __init__(self, num_envs, num_dof, foot_dim=(4,3)):
        self.num_envs = num_envs
        self.DOF = num_dof
        self.FOOT_DIM = foot_dim

        self.shm = {}

        # Inputs
        #      the plus (1) accounts for the quat. representation of orientation used by pinocchio
        self.q       = self.make("uni_2_q",       (num_envs, self.DOF+1))
        self.qd      = self.make("uni_2_qd",      (num_envs, self.DOF))
        self.qd_prev = self.make("uni_2_qd_prev", (num_envs, self.DOF))
        self.grf     = self.make("uni_2_grf",     (num_envs, *self.FOOT_DIM))
        self.dt      = self.make("uni_2_dt",      (1,))

        # Outputs
        self.wb_dynamics = self.make("uni_2_wb_dynamics", (num_envs, self.DOF))
        self.wb_contacts = self.make("uni_2_wb_contacts", (num_envs, self.DOF))
        self.mass_mat    = self.make("uni_2_mass_mat",    (num_envs, self.DOF, self.DOF))
        self.bias        = self.make("uni_2_bias",        (num_envs, self.DOF))
        self.acc6d       = self.make("uni_2_acc6d",       (num_envs, 6))

    def make(self, name, shape, dtype=np.float32):
        nbytes = np.prod(shape) * np.dtype(dtype).itemsize
        shm = SharedMemory(create=True, size=nbytes, name=name)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        self.shm[name] = (shm, arr)
        return arr

    def close(self):
        for shm, _ in self.shm.values():
            shm.close()
            shm.unlink()


def compute_single_env(i, shared, pino_model, pino_data, 
                       pino_foot_names,
                       correct_idxs):
    """
    Compute WB dynamics, mass matrix, bias forces, and contact forces
    for a single environment index i.
    """

    q       = shared.uni_2_q[i]
    qd      = shared.uni_2_qd[i]
    qdprev  = shared.uni_2_qd_prev[i]
    grf     = shared.uni_2_grf[i]      # shaped as (12)
    dt      = shared.uni_2_dt[0]

    # Acceleration (backward finite difference)
    acc = (qd - qdprev) / dt

    # 6-DoF torso accel
    shared.uni_2_acc6d[i] = acc[:6]

    # Pinocchio general coords
    aq0 = np.zeros_like(qd)

    b = pn.rnea(pino_model, pino_data, q, qd, aq0)
    M = pn.crba(pino_model, pino_data, q)

    wb_dyn = M @ acc + b
    shared.uni_2_wb_dynamics[i] = wb_dyn[correct_idxs]

    # Also save reduced mass matrix + bias
    M_ = M[np.ix_(correct_idxs, correct_idxs)]
    shared.uni_2_mass_mat[i] = M_
    shared.uni_2_bias[i] = b[correct_idxs]

    # Contact forces
    J_list = []
    for foot_name in pino_foot_names:
        fid = pino_model.getFrameId(foot_name)
        J = pn.computeFrameJacobian(pino_model, pino_data, q, fid, pn.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, :]
        J_list.append(J)

    J_total = np.concatenate(J_list, axis=0)
    tau = (J_total.transpose() @ grf).squeeze()

    shared.uni_2_wb_contacts[i] = tau[correct_idxs]

def worker_loop(worker_id, shared_names, shapes, dtypes,
                pino_model, pino_foot_names, correct_idxs,
                task_queue, done_queue,
                idle_sleep=0.01):
    """
    Long-running asynchronous worker.
    """

    # Reattach shared memory
    class S:
        pass

    shared = S()
    # shared.q = np.ndarray(shapes["q"], dtype=dtypes["q"],
    #                       buffer=SharedMemory(name=shared_names["q"]).buf)
    # shared.qd = np.ndarray(shapes["qd"], dtype=dtypes["qd"],
    #                        buffer=SharedMemory(name=shared_names["qd"]).buf)
    # shared.qd_prev = np.ndarray(shapes["qd_prev"], dtype=dtypes["qd_prev"],
    #                        buffer=SharedMemory(name=shared_names["qd_prev"]).buf)
    # shared.grf = np.ndarray(shapes["grf"], dtype=dtypes["grf"],
    #                        buffer=SharedMemory(name=shared_names["grf"]).buf)
    # shared.dt = np.ndarray(shapes["dt"], dtype=dtypes["dt"],
    #                        buffer=SharedMemory(name=shared_names["dt"]).buf)

    # shared.wb_dynamics = np.ndarray(shapes["wb_dynamics"], dtype=dtypes["wb_dynamics"],
    #                        buffer=SharedMemory(name=shared_names["wb_dynamics"]).buf)
    # shared.wb_contacts = np.ndarray(shapes["wb_contacts"], dtype=dtypes["wb_contacts"],
    #                        buffer=SharedMemory(name=shared_names["wb_contacts"]).buf)
    # shared.mass_mat = np.ndarray(shapes["mass_mat"], dtype=dtypes["mass_mat"],
    #                        buffer=SharedMemory(name=shared_names["mass_mat"]).buf)
    # shared.bias = np.ndarray(shapes["bias"], dtype=dtypes["bias"],
    #                        buffer=SharedMemory(name=shared_names["bias"]).buf)
    # shared.acc6d = np.ndarray(shapes["acc6d"], dtype=dtypes["acc6d"],
    #                        buffer=SharedMemory(name=shared_names["acc6d"]).buf)

    shared_shms = {}

    for name in shared_names:
        shm = SharedMemory(name=shared_names[name])
        shared_shms[name] = shm
        setattr(shared, name, np.ndarray(shapes[name], dtype=dtypes[name], buffer=shm.buf))


    pino_data = pino_model.createData()

    while True:
        try:
            # Try to get a task, but timeout if none available
            env_id = task_queue.get(timeout=idle_sleep)
        except queue.Empty:
            # No task, yield CPU
            time.sleep(idle_sleep)  # optional extra sleep
            continue

        if env_id == "STOP":
            break

        compute_single_env(env_id, shared, pino_model, pino_data,
                           pino_foot_names,
                           correct_idxs)

        done_queue.put(env_id)

    # Close handles (don't unlink, main process owns them)
    for shm in shared_shms.values():
        shm.close()


class PinocchioAsync:
    def __init__(self, pino_model, num_envs,
                 pino_foot_names,
                 correct_idxs,
                 wb_dim,
                 foot_dim,
                 num_cpu):

        self.pino_model = pino_model
        self.num_envs = num_envs
        self.pino_foot_names = pino_foot_names
        self.correct_idxs = correct_idxs

        # Shared memory
        self.shared = SharedTensors(num_envs, num_dof=wb_dim, foot_dim=foot_dim)

        # Spawn workers
        self.task_q = mp.Queue()
        self.done_q = mp.Queue()

        # Shared memory descriptors
        self.shared_names = {k: shm.name for k,(shm,arr) in self.shared.shm.items()}
        self.shapes = {k: arr.shape for k,(shm,arr) in self.shared.shm.items()}
        self.dtypes = {k: arr.dtype for k,(shm,arr) in self.shared.shm.items()}

        self.workers = []
        for w in range(num_cpu):
            p = mp.Process(
                target=worker_loop,
                args=(w, 
                      self.shared_names, 
                      self.shapes, 
                      self.dtypes,
                      pino_model,
                      pino_foot_names,
                      correct_idxs,
                      self.task_q, self.done_q)
            )
            p.daemon = True
            p.start()
            self.workers.append(p)

    def compute_async(self):
        for env_id in range(self.num_envs):
            self.task_q.put(env_id)

    def wait(self):
        completed = 0
        while completed < self.num_envs:
            _ = self.done_q.get()
            completed += 1

    def shutdown(self):
        for _ in self.workers:
            self.task_q.put("STOP")
        for p in self.workers:
            p.join()
        self.shared.close()
