from datasets import load_from_disk
from spm_gpt.eval.generate_eval_cmd_dataset_answer import run, run_gpt, model_list
from spm_gpt.eval.gpu_energy_meter import GPUEnergyMeter


if __name__ == '__main__':
    # Stage ii test set of the revised manuscript (67 queries): the 34 original queries (spmcmd_test_planning_dataset)
    # followed by the 33 queries with altered experimental conditions (spmcmd_planning_altered_dataset)
    dataset = load_from_disk("../finetune/spmcmd_planning_testv2_dataset")
    print(len(dataset))

    # latency and GPU energy of each query are recorded when a GPUEnergyMeter is passed (meter=None to skip)
    meter = GPUEnergyMeter(); meter.measure_idle()
    for it in range(len(model_list)):
        run(it, dataset, save_dir="eval_results_cmd_planningv2", meter=meter)

    # o4-mini was evaluated only on the 34 original queries (eval_results_cmd_planning/gpt4.json),
    # because it was no longer available through the OpenAI API when the test set was extended
    # dataset = load_from_disk("../finetune/spmcmd_test_planning_dataset")
    # run_gpt(inference=True, _dataset=dataset, save_dir="eval_results_cmd_planning")
