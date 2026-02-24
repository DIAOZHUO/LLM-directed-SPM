from datasets import load_from_disk
from spm_gpt.eval.generate_eval_cmd_dataset_answer import run, run_gpt, model_list



if __name__ == '__main__':
    dataset = load_from_disk("../finetune/spmcmd_test_planning_dataset")
    print(len(dataset))
    for it in range(len(model_list)):
        run(it, dataset, save_dir="eval_results_cmd_planning")

    # run_gpt(inference=True, _dataset=dataset, save_dir="eval_results_cmd_planning")


