Superseded bead runs (kept for their validation logs; not evaluated in the paper's tables).
ppo_4e5_ls-0.5                    PPO from scratch, 4e5 steps, log_std_init -0.5, seeds 0-1.
dagger_ppo_4e5_ls-0.5_lr3e-4      DAgger->PPO fine-tuning, 4e5 steps, log_std_init -0.5, lr 3e-4,
                                  no BC term; initialised from the 6-round DAgger checkpoints of
                                  seeds 0-1. Its validation curve (error jump at the first
                                  post-warm-up evaluation) motivated the revised fine-tuning
                                  settings (log_std_init -2, lr 1e-4, BC term).
Both were replaced by the runs in runs/bead/{ppo,dagger_ppo,ppo_acc,dagger_ppo_acc}.
