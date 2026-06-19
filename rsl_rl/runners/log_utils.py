def add_timing_info(
    current_learning_iteration,
    num_learning_iterations,
    it,
    ep_string,
    iteration_time,
    width,
    pad,
    tot_timesteps,
    tot_time,
):
    remaining_iters = (
        current_learning_iteration
        + num_learning_iterations
        - it
        - 1
    )

    eta_seconds = remaining_iters * iteration_time

    days, rem = divmod(int(eta_seconds), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    eta_str = f"{days}d {hours}h {minutes}m {seconds}s"
    log_string = ep_string
    log_string += f"{'-' * width}\n"
    log_string += f"{'Total timesteps:':>{pad}} {tot_timesteps}\n"
    log_string += f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
    log_string += f"{'Total time:':>{pad}} {tot_time:.2f}s\n"
    log_string += f"{'ETA:':>{pad}} {eta_str}\n"
    return log_string