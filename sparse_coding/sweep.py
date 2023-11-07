import torch


from process_tiny_stories_data import load_tinystories_validation_prompts, load_tinystories_tokens
from sparse_coding.train_autoencoder import get_config, main


if __name__ == "__main__":
    torch.cuda.empty_cache()
    cfg = get_config()
    cfg["cfg_file"] = "config/tiny_stories_seed_sweep_A6000.json"
    prompt_data = load_tinystories_tokens(cfg["data_path"])
    eval_prompts = load_tinystories_validation_prompts(cfg["data_path"])[:cfg["num_eval_prompts"]]
    # for l1_coeff in [0.00001, 0.0001, 0.0005, 0.001, 0.01]:
    #     torch.cuda.empty_cache()
    #     cfg["l1_coeff"] = l1_coeff
    #     main(cfg["model"], cfg["act"], cfg, prompt_data, eval_prompts)
    for seed in [47, 48, 49, 50]:
        torch.cuda.empty_cache()
        cfg["seed"] = seed
        main(cfg["model"], cfg["act"], cfg, prompt_data, eval_prompts)
