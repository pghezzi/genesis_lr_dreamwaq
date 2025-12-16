from multiprocessing.shared_memory import SharedMemory

for name in [
    "q", "qd", "qd_prev", "mass_mat",
    "wb_dynamics", "wb_contacts", "bias",
    "grf", "acc6d", "dt"
]:
    try:
        SharedMemory(name=name).unlink()
    except FileNotFoundError:
        pass
