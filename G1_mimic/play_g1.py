from g1.g1_mimic import H1Mimic
from g1.g1_mimic_config import H1MimicCfg, H1MimicCfgPPO
import torch

env_cfg = H1MimicCfg()

train_cfg = H1MimicCfgPPO()



from utils.helpers import class_to_dict, get_args, parse_sim_params

# 1) 把 env_cfg.sim 里的默认值拉出来
sim_dict = {"sim": class_to_dict(env_cfg.sim)}
args = get_args()
# 2) 把 args（--dt、--substeps、--gravity……）合并进来
sim_params = parse_sim_params(args, sim_dict)



env = H1Mimic(cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=False)


env.set_camera(
    position=torch.tensor([3.0, 0.0, 1.5], device=env.device),
    lookat=torch.tensor([0.0, 0.0, 1.0], device=env.device)
)