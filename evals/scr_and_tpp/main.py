import gc
import os
import shutil
import random
import time
from dataclasses import asdict
from typing import Optional

import einops
from pydantic import TypeAdapter
import torch
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer
import argparse
from datetime import datetime
import pickle

import evals.scr_and_tpp.dataset_creation as dataset_creation
from evals.scr_and_tpp.eval_config import ScrAndTppEvalConfig
from evals.scr_and_tpp.eval_output import (
    EVAL_TYPE_ID_SCR,
    EVAL_TYPE_ID_TPP,
    ScrEvalOutput,
    ScrMetricCategories,
    ScrResultDetail,
    ScrMetrics,
    TppEvalOutput,
    TppMetricCategories,
    TppResultDetail,
    TppMetrics,
)
import evals.sparse_probing.probe_training as probe_training
import sae_bench_utils.activation_collection as activation_collection
import sae_bench_utils.dataset_info as dataset_info
import sae_bench_utils.dataset_utils as dataset_utils
import sae_bench_utils.general_utils as general_utils
from sae_bench_utils import (
    get_eval_uuid,
    get_sae_lens_version,
    get_sae_bench_version,
)
from sae_bench_utils.sae_selection_utils import get_saes_from_regex

COLUMN2_VALS_LOOKUP = {
    "LabHC/bias_in_bios_class_set1": ("male", "female"),
    "canrager/amazon_reviews_mcauley_1and5": (1.0, 5.0),
}


@torch.no_grad()
def get_effects_per_class_precomputed_acts(
    sae: SAE,
    probe: probe_training.Probe,
    class_idx: str,
    precomputed_acts: dict[str, torch.Tensor],
    perform_scr: bool,
    sae_batch_size: int,
) -> torch.Tensor:
    inputs_train_BLD, labels_train_B = probe_training.prepare_probe_data(
        precomputed_acts, class_idx, perform_scr
    )

    assert inputs_train_BLD.shape[0] == len(labels_train_B)

    device = inputs_train_BLD.device
    dtype = inputs_train_BLD.dtype

    running_sum_pos_F = torch.zeros(sae.W_dec.data.shape[0], dtype=torch.float32, device=device)
    running_sum_neg_F = torch.zeros(sae.W_dec.data.shape[0], dtype=torch.float32, device=device)
    count_pos = 0
    count_neg = 0

    for i in range(0, inputs_train_BLD.shape[0], sae_batch_size):
        activation_batch_BLD = inputs_train_BLD[i : i + sae_batch_size]
        labels_batch_B = labels_train_B[i : i + sae_batch_size]

        activations_BL = einops.reduce(activation_batch_BLD, "B L D -> B L", "sum")
        nonzero_acts_BL = (activations_BL != 0.0).to(dtype=dtype)
        nonzero_acts_B = einops.reduce(nonzero_acts_BL, "B L -> B", "sum").to(torch.float32)

        # ### FIXED: Detach after encoding ###
        f_BLF = sae.encode(activation_batch_BLD).detach()
        f_BLF = f_BLF * nonzero_acts_BL[:, :, None]  # zero out masked tokens

        average_sae_acts_BF = (
            einops.reduce(f_BLF, "B L F -> B F", "sum").to(torch.float32) / nonzero_acts_B[:, None]
        )

        pos_mask = labels_batch_B == dataset_info.POSITIVE_CLASS_LABEL
        neg_mask = labels_batch_B == dataset_info.NEGATIVE_CLASS_LABEL

        running_sum_pos_F += einops.reduce(average_sae_acts_BF[pos_mask], "B F -> F", "sum")
        running_sum_neg_F += einops.reduce(average_sae_acts_BF[neg_mask], "B F -> F", "sum")

        count_pos += pos_mask.sum().item()
        count_neg += neg_mask.sum().item()

    average_pos_sae_acts_F = running_sum_pos_F / count_pos if count_pos > 0 else running_sum_pos_F
    average_neg_sae_acts_F = running_sum_neg_F / count_neg if count_neg > 0 else running_sum_neg_F

    average_acts_F = (average_pos_sae_acts_F - average_neg_sae_acts_F).to(dtype)

    probe_weight_F = probe.net.weight.to(dtype=dtype, device=device).squeeze()

    if not perform_scr:
        average_acts_F.clamp_(min=0.0)

    effects_F = (average_acts_F * probe_weight_F).to(dtype=torch.float32)

    if perform_scr:
        effects_F = effects_F.abs()

    return effects_F


def get_all_node_effects_for_one_sae(
    sae: SAE,
    probes: dict[str, probe_training.Probe],
    chosen_class_indices: list[str],
    perform_scr: bool,
    indirect_effect_acts: dict[str, torch.Tensor],
    sae_batch_size: int,
) -> dict[str, torch.Tensor]:
    node_effects = {}
    for ablated_class_idx in chosen_class_indices:
        node_effects[ablated_class_idx] = get_effects_per_class_precomputed_acts(
            sae,
            probes[ablated_class_idx],
            ablated_class_idx,
            indirect_effect_acts,
            perform_scr,
            sae_batch_size,
        )

    return node_effects


def select_top_n_features(effects: torch.Tensor, n: int, class_name: str) -> torch.Tensor:
    assert (
        n <= effects.numel()
    ), f"n ({n}) must not be larger than the number of features ({effects.numel()}) for ablation class {class_name}"

    non_zero_mask = effects != 0
    non_zero_effects = effects[non_zero_mask]
    num_non_zero = non_zero_effects.numel()

    if num_non_zero < n:
        print(
            f"WARNING: only {num_non_zero} non-zero effects found for ablation class {class_name}, which is less than the requested {n}."
        )

    k = min(n, num_non_zero)

    if k == 0:
        print(
            f"WARNING: No non-zero effects found for ablation class {class_name}. Returning an empty mask."
        )
        top_n_features = torch.zeros_like(effects, dtype=torch.bool)
    else:
        _, top_indices = torch.topk(effects, k)
        mask = torch.zeros_like(effects, dtype=torch.bool)
        mask[top_indices] = True
        top_n_features = mask

    return top_n_features


def ablated_precomputed_activations(
    ablation_acts_BLD: torch.Tensor,
    sae: SAE,
    to_ablate: torch.Tensor,
    sae_batch_size: int,
) -> torch.Tensor:
    """Now returns [B, F] latents suitable for the probe."""
    all_acts_list_BF = []

    for i in range(0, ablation_acts_BLD.shape[0], sae_batch_size):
        activation_batch_BLD = ablation_acts_BLD[i : i + sae_batch_size]
        dtype = activation_batch_BLD.dtype

        activations_BL = einops.reduce(activation_batch_BLD, "B L D -> B L", "sum")
        nonzero_acts_BL = (activations_BL != 0.0).to(dtype=dtype)
        nonzero_acts_B = einops.reduce(nonzero_acts_BL, "B L -> B", "sum").float()

        # Encode to get latents
        f_BLF = sae.encode(activation_batch_BLD).detach()
        x_hat_BLD = sae.decode(f_BLF)

        error_BLD = activation_batch_BLD - x_hat_BLD

        f_BLF[..., to_ablate] = 0.0  # zero out chosen features in latent space

        modified_acts_BLD = sae.decode(f_BLF) + error_BLD

        # Re-encode modified acts back into SAE latents to get [B, L, F]
        f_modified_BLF = sae.encode(modified_acts_BLD).detach()

        # Mean-reduce over sequence dimension L to get [B, F]
        average_sae_acts_BF = einops.reduce(f_modified_BLF, "B L F -> B F", "sum") / nonzero_acts_B[:, None]
        all_acts_list_BF.append(average_sae_acts_BF)

    # Concatenate all batches along B dimension
    all_acts_BF = torch.cat(all_acts_list_BF, dim=0)
    return all_acts_BF


def get_probe_test_accuracy(
    probes: dict[str, probe_training.Probe],
    all_class_list: list[str],
    all_activations: dict[str, torch.Tensor],
    probe_batch_size: int,
    perform_scr: bool,
) -> dict[str, float]:
    test_accuracies = {}
    for class_name in all_class_list:
        test_acts, test_labels = probe_training.prepare_probe_data(
            all_activations, class_name, perform_scr=perform_scr
        )

        test_acc_probe = probe_training.test_probe_gpu(
            test_acts,
            test_labels,
            probe_batch_size,
            probes[class_name],
        )
        test_accuracies[class_name] = test_acc_probe

    if perform_scr:
        scr_probe_accuracies = get_scr_probe_test_accuracy(
            probes, all_class_list, all_activations, probe_batch_size
        )
        test_accuracies.update(scr_probe_accuracies)

    return test_accuracies


def get_scr_probe_test_accuracy(
    probes: dict[str, probe_training.Probe],
    all_class_list: list[str],
    all_activations: dict[str, torch.Tensor],
    probe_batch_size: int,
) -> dict[str, float]:
    test_accuracies = {}
    for class_name in all_class_list:
        if class_name not in dataset_info.PAIRED_CLASS_KEYS:
            continue
        spurious_class_names = [key for key in dataset_info.PAIRED_CLASS_KEYS if key != class_name]
        test_acts, test_labels = probe_training.prepare_probe_data(
            all_activations, class_name, perform_scr=True
        )

        for spurious_class_name in spurious_class_names:
            test_acc_probe = probe_training.test_probe_gpu(
                test_acts,
                test_labels,
                probe_batch_size,
                probes[spurious_class_name],
            )
            combined_class_name = f"{spurious_class_name} probe on {class_name} data"
            test_accuracies[combined_class_name] = test_acc_probe

    return test_accuracies


def perform_feature_ablations(
    probes: dict[str, probe_training.Probe],
    sae: SAE,
    sae_batch_size: int,
    all_test_acts_BLD: dict[str, torch.Tensor],
    node_effects: dict[str, torch.Tensor],
    top_n_values: list[int],
    chosen_classes: list[str],
    probe_batch_size: int,
    perform_scr: bool,
) -> dict[str, dict[int, dict[str, float]]]:
    ablated_class_accuracies = {}
    for ablated_class_name in chosen_classes:
        ablated_class_accuracies[ablated_class_name] = {}
        for top_n in top_n_values:
            selected_features_F = select_top_n_features(
                node_effects[ablated_class_name], top_n, ablated_class_name
            )
            test_acts_ablated = {}
            for evaluated_class_name in all_test_acts_BLD.keys():
                test_acts_ablated[evaluated_class_name] = ablated_precomputed_activations(
                    all_test_acts_BLD[evaluated_class_name],
                    sae,
                    selected_features_F,
                    sae_batch_size,
                )

            ablated_class_accuracies[ablated_class_name][top_n] = get_probe_test_accuracy(
                probes,
                chosen_classes,
                test_acts_ablated,
                probe_batch_size,
                perform_scr,
            )
    return ablated_class_accuracies


def get_scr_plotting_dict(
    class_accuracies: dict[str, dict[int, dict[str, float]]],
    llm_clean_accs: dict[str, float],
) -> dict[str, float]:
    results = {}
    eval_probe_class_id = "male_professor / female_nurse"

    dirs = [1, 2]

    dir1_class_name = f"{eval_probe_class_id} probe on professor / nurse data"
    dir2_class_name = f"{eval_probe_class_id} probe on male / female data"

    dir1_acc = llm_clean_accs[dir1_class_name]
    dir2_acc = llm_clean_accs[dir2_class_name]

    for dir in dirs:
        if dir == 1:
            ablated_probe_class_id = "male / female"
            eval_data_class_id = "professor / nurse"
        elif dir == 2:
            ablated_probe_class_id = "professor / nurse"
            eval_data_class_id = "male / female"
        else:
            raise ValueError("Invalid dir.")

        for threshold in class_accuracies[ablated_probe_class_id]:
            clean_acc = llm_clean_accs[eval_data_class_id]

            combined_class_name = f"{eval_probe_class_id} probe on {eval_data_class_id} data"

            original_acc = llm_clean_accs[combined_class_name]

            changed_acc = class_accuracies[ablated_probe_class_id][threshold][combined_class_name]

            scr_score = (changed_acc - original_acc) / (clean_acc - original_acc)

            print(
                f"dir: {dir}, original_acc: {original_acc}, clean_acc: {clean_acc}, changed_acc: {changed_acc}, scr_score: {scr_score}"
            )

            metric_key = f"scr_dir{dir}_threshold_{threshold}"
            results[metric_key] = scr_score

            scr_metric_key = f"scr_metric_threshold_{threshold}"
            if dir1_acc < dir2_acc and dir == 1:
                results[scr_metric_key] = scr_score
            elif dir1_acc > dir2_acc and dir == 2:
                results[scr_metric_key] = scr_score

    return results


def create_tpp_plotting_dict(
    class_accuracies: dict[str, dict[int, dict[str, float]]],
    llm_clean_accs: dict[str, float],
) -> dict[str, float]:
    results = {}
    intended_diffs = {}
    unintended_diffs = {}

    classes = list(llm_clean_accs.keys())

    for class_name in classes:
        if " probe on " in class_name:
            raise ValueError("This is SCR, shouldn't be here.")

        intended_clean_acc = llm_clean_accs[class_name]

        for threshold in class_accuracies[class_name]:
            intended_patched_acc = class_accuracies[class_name][threshold][class_name]
            intended_diff = intended_clean_acc - intended_patched_acc

            if threshold not in intended_diffs:
                intended_diffs[threshold] = []
            intended_diffs[threshold].append(intended_diff)

        for intended_class_id in classes:
            for unintended_class_id in classes:
                if intended_class_id == unintended_class_id:
                    continue

                unintended_clean_acc = llm_clean_accs[unintended_class_id]

                for threshold in class_accuracies[intended_class_id]:
                    unintended_patched_acc = class_accuracies[intended_class_id][threshold][
                        unintended_class_id
                    ]
                    unintended_diff = unintended_clean_acc - unintended_patched_acc

                    if threshold not in unintended_diffs:
                        unintended_diffs[threshold] = []
                    unintended_diffs[threshold].append(unintended_diff)

        for threshold in intended_diffs:
            assert threshold in unintended_diffs

            average_intended_diff = sum(intended_diffs[threshold]) / len(intended_diffs[threshold])
            average_unintended_diff = sum(unintended_diffs[threshold]) / len(
                unintended_diffs[threshold]
            )
            average_diff = average_intended_diff - average_unintended_diff

            results[f"tpp_threshold_{threshold}_total_metric"] = average_diff
            results[f"tpp_threshold_{threshold}_intended_diff_only"] = average_intended_diff
            results[f"tpp_threshold_{threshold}_unintended_diff_only"] = average_unintended_diff

    return results


def get_dataset_activations(
    dataset_name: str,
    config: ScrAndTppEvalConfig,
    model: HookedTransformer,
    llm_batch_size: int,
    layer: int,
    hook_point: str,
    device: str,
    chosen_classes: list[str],
    column1_vals: Optional[tuple[str, str]] = None,
    column2_vals: Optional[tuple[str, str]] = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    train_data, test_data = dataset_creation.get_train_test_data(
        dataset_name,
        config.perform_scr,
        config.train_set_size,
        config.test_set_size,
        config.random_seed,
        column1_vals,
        column2_vals,
    )

    if not config.perform_scr:
        train_data = dataset_utils.filter_dataset(train_data, chosen_classes)
        test_data = dataset_utils.filter_dataset(test_data, chosen_classes)

    train_data = dataset_utils.tokenize_data_dictionary(
        train_data, model.tokenizer, config.context_length, device
    )
    test_data = dataset_utils.tokenize_data_dictionary(
        test_data, model.tokenizer, config.context_length, device
    )

    all_train_acts_BLD = activation_collection.get_all_llm_activations(
        train_data, model, llm_batch_size, layer, hook_point, mask_bos_pad_eos_tokens=True
    )
    all_test_acts_BLD = activation_collection.get_all_llm_activations(
        test_data, model, llm_batch_size, layer, hook_point, mask_bos_pad_eos_tokens=True
    )

    return all_train_acts_BLD, all_test_acts_BLD


def create_meaned_sae_activations(all_acts_BLD: dict[str, torch.Tensor], sae: SAE, sae_batch_size: int) -> dict[str, torch.Tensor]:
    result = {}
    for class_name, acts_BLD in all_acts_BLD.items():
        all_batches_BF = []
        for i in range(0, acts_BLD.shape[0], sae_batch_size):
            batch_BLD = acts_BLD[i : i + sae_batch_size]
            # ### FIXED: Detach after encoding ###
            f_BLF = sae.encode(batch_BLD).detach()

            activations_BL = einops.reduce(batch_BLD, "B L D -> B L", "sum")
            nonzero_acts_BL = (activations_BL != 0.0)
            nonzero_acts_B = einops.reduce(nonzero_acts_BL, "B L -> B", "sum").float()

            average_sae_acts_BF = (einops.reduce(f_BLF, "B L F -> B F", "sum") / nonzero_acts_B[:, None])
            all_batches_BF.append(average_sae_acts_BF)
        result[class_name] = torch.cat(all_batches_BF, dim=0)
    return result


def run_eval_single_dataset(
    dataset_name: str,
    config: ScrAndTppEvalConfig,
    sae: SAE,
    model: HookedTransformer,
    layer: int,
    hook_point: str,
    device: str,
    # artifacts_folder: str,  # No longer needed for this simplified version
    # save_activations: bool = True,  # No longer needed
    column1_vals: Optional[tuple[str, str]] = None,
) -> tuple[dict[str, dict[int, dict[str, float]]], dict[str, float]]:
    """
    Compute dataset activations, train the probe on SAE latents, compute node effects, 
    and perform feature ablations. This version does not save or load activations or probes.
    """
    column2_vals = COLUMN2_VALS_LOOKUP[dataset_name]

    if not config.perform_scr:
        chosen_classes = dataset_info.chosen_classes_per_dataset[dataset_name]
    else:
        chosen_classes = list(dataset_info.PAIRED_CLASS_KEYS.keys())

    # Compute LLM activations fresh every time
    if config.lower_vram_usage:
        model = model.to(device)
    all_train_acts_BLD, all_test_acts_BLD = get_dataset_activations(
        dataset_name,
        config,
        model,
        config.llm_batch_size,
        layer,
        hook_point,
        device,
        chosen_classes,
        column1_vals,
        column2_vals,
    )
    if config.lower_vram_usage:
        model = model.to("cpu")

    # Create SAE latents (meaned)
    all_meaned_train_sae_acts_BF = create_meaned_sae_activations(all_train_acts_BLD, sae, config.sae_batch_size)
    all_meaned_test_sae_acts_BF = create_meaned_sae_activations(all_test_acts_BLD, sae, config.sae_batch_size)

    # Train probe from scratch
    torch.set_grad_enabled(True)
    llm_probes, llm_test_accuracies = probe_training.train_probe_on_activations(
        all_meaned_train_sae_acts_BF,
        all_meaned_test_sae_acts_BF,
        select_top_k=None,
        use_sklearn=False,
        batch_size=config.probe_train_batch_size,
        epochs=config.probe_epochs,
        lr=config.probe_lr,
        perform_scr=config.perform_scr,
        early_stopping_patience=config.early_stopping_patience,
        l1_penalty=config.probe_l1_penalty,
    )
    torch.set_grad_enabled(False)

    # Evaluate the trained probe on test set
    llm_test_accuracies = get_probe_test_accuracy(
        llm_probes,
        chosen_classes,
        all_meaned_test_sae_acts_BF,
        config.probe_test_batch_size,
        config.perform_scr,
    )

    torch.set_grad_enabled(False)

    # Compute node effects using the trained probe
    sae_node_effects = get_all_node_effects_for_one_sae(
        sae,
        llm_probes,
        chosen_classes,
        config.perform_scr,
        all_train_acts_BLD,
        config.sae_batch_size,
    )

    # Perform feature ablations and re-encode through SAE to get latents before probe testing
    ablated_class_accuracies = perform_feature_ablations(
        llm_probes,
        sae,
        config.sae_batch_size,
        all_test_acts_BLD,
        sae_node_effects,
        config.n_values,
        chosen_classes,
        config.probe_test_batch_size,
        config.perform_scr,
    )

    return ablated_class_accuracies, llm_test_accuracies


def run_eval_single_sae(
    config: ScrAndTppEvalConfig,
    sae: SAE,
    model: HookedTransformer,
    device: str,
) -> dict[str, float | dict[str, float]]:
    """
    Compute results for one SAE without saving/loading intermediate artifacts.
    """
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    dataset_results = {}
    averaging_names = []

    for dataset_name in config.dataset_names:
        if config.perform_scr:
            column1_vals_list = config.column1_vals_lookup[dataset_name]
            for column1_vals in column1_vals_list:
                run_name = f"{dataset_name}_scr_{column1_vals[0]}_{column1_vals[1]}"
                raw_results, llm_clean_accs = run_eval_single_dataset(
                    dataset_name,
                    config,
                    sae,
                    model,
                    sae.cfg.hook_layer,
                    sae.cfg.hook_name,
                    device,
                    column1_vals=column1_vals,
                )

                processed_results = get_scr_plotting_dict(raw_results, llm_clean_accs)
                dataset_results[f"{run_name}_results"] = processed_results
                averaging_names.append(run_name)
        else:
            run_name = f"{dataset_name}_tpp"
            raw_results, llm_clean_accs = run_eval_single_dataset(
                dataset_name,
                config,
                sae,
                model,
                sae.cfg.hook_layer,
                sae.cfg.hook_name,
                device,
            )

            processed_results = create_tpp_plotting_dict(raw_results, llm_clean_accs)
            dataset_results[f"{run_name}_results"] = processed_results
            averaging_names.append(run_name)

    results_dict = general_utils.average_results_dictionaries(dataset_results, averaging_names)
    results_dict.update(dataset_results)

    if config.lower_vram_usage:
        model = model.to(device)

    return results_dict


def run_eval(
    config: ScrAndTppEvalConfig,
    selected_saes: list[tuple[str, SAE]] | list[tuple[str, str]],
    device: str,
    output_path: str,
    force_rerun: bool = False,
    clean_up_activations: bool = False,
):
    """
    Evaluate multiple SAEs on multiple datasets without saving/loading activations or probes.
    """
    eval_instance_id = get_eval_uuid()
    sae_lens_version = get_sae_lens_version()
    sae_bench_commit_hash = get_sae_bench_version()

    if config.perform_scr:
        eval_type = EVAL_TYPE_ID_SCR
    else:
        eval_type = EVAL_TYPE_ID_TPP

    output_path = os.path.join(output_path, eval_type)
    os.makedirs(output_path, exist_ok=True)

    # No artifacts folder usage here since we're not saving activations
    results_dict = {}

    llm_dtype = general_utils.str_to_dtype(config.llm_dtype)
    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=llm_dtype
    )

    for sae_release, sae_id in tqdm(
        selected_saes, desc="Running SAE evaluation on all selected SAEs"
    ):
        del_sae = False
        if isinstance(sae_id, str):
            sae = SAE.from_pretrained(
                release=sae_release,
                sae_id=sae_id,
                device=device,
            )[0]
            del_sae = True
        else:
            sae = sae_id
            sae_id = "custom_sae"

        sae = sae.to(device=device, dtype=llm_dtype)

        # Just run the evaluation; no loading from disk.
        scr_or_tpp_results = run_eval_single_sae(
            config,
            sae,
            model,
            device,
        )

        if eval_type == EVAL_TYPE_ID_SCR:
            eval_output = ScrEvalOutput(
                eval_type_id=eval_type,
                eval_config=config,
                eval_id=eval_instance_id,
                datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
                eval_result_metrics=ScrMetricCategories(
                    scr_metrics=ScrMetrics(
                        **{k: v for k, v in scr_or_tpp_results.items() if not isinstance(v, dict)}
                    )
                ),
                eval_result_details=[
                    ScrResultDetail(
                        dataset_name=dataset_name,
                        **result,
                    )
                    for dataset_name, result in scr_or_tpp_results.items()
                    if isinstance(result, dict)
                ],
                sae_bench_commit_hash=sae_bench_commit_hash,
                sae_lens_id=sae_id,
                sae_lens_release_id=sae_release,
                sae_lens_version=sae_lens_version,
            )
        elif eval_type == EVAL_TYPE_ID_TPP:
            eval_output = TppEvalOutput(
                eval_type_id=eval_type,
                eval_config=config,
                eval_id=eval_instance_id,
                datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
                eval_result_metrics=TppMetricCategories(
                    tpp_metrics=TppMetrics(
                        **{k: v for k, v in scr_or_tpp_results.items() if not isinstance(v, dict)}
                    )
                ),
                eval_result_details=[
                    TppResultDetail(
                        dataset_name=dataset_name,
                        **result,
                    )
                    for dataset_name, result in scr_or_tpp_results.items()
                    if isinstance(result, dict)
                ],
                sae_bench_commit_hash=sae_bench_commit_hash,
                sae_lens_id=sae_id,
                sae_lens_release_id=sae_release,
                sae_lens_version=sae_lens_version,
            )
        else:
            raise ValueError(f"Invalid eval type: {eval_type}")

        sae_result_file = f"{sae_release}_{sae_id}_eval_results.json"
        sae_result_file = sae_result_file.replace("/", "_")
        sae_result_path = os.path.join(output_path, sae_result_file)
        eval_output.to_json_file(sae_result_path, indent=2)

        results_dict[f"{sae_release}_{sae_id}"] = asdict(eval_output)

        if del_sae:
            del sae
        gc.collect()
        torch.cuda.empty_cache()

    return results_dict


def create_config_and_selected_saes(
    args,
) -> tuple[ScrAndTppEvalConfig, list[tuple[str, str]]]:
    config = ScrAndTppEvalConfig(
        model_name=args.model_name,
        perform_scr=args.perform_scr,
    )

    if args.llm_batch_size is not None:
        config.llm_batch_size = args.llm_batch_size
    else:
        config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name]

    if args.llm_dtype is not None:
        config.llm_dtype = args.llm_dtype
    else:
        config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

    if args.random_seed is not None:
        config.random_seed = args.random_seed

    if args.lower_vram_usage:
        config.lower_vram_usage = True

    if args.sae_batch_size is not None:
        config.sae_batch_size = args.sae_batch_size

    selected_saes = get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)
    assert len(selected_saes) > 0, "No SAEs selected"

    releases = set([release for release, _ in selected_saes])

    print(f"Selected SAEs from releases: {releases}")

    for release, sae in selected_saes:
        print(f"Sample SAEs: {release}, {sae}")

    return config, selected_saes


def arg_parser():
    parser = argparse.ArgumentParser(description="Run SCR or TPP evaluation")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--sae_regex_pattern",
        type=str,
        required=True,
        help="Regex pattern for SAE selection",
    )
    parser.add_argument(
        "--sae_block_pattern",
        type=str,
        required=True,
        help="Regex pattern for SAE block selection",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="eval_results",
        help="Output folder",
    )
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun of experiments")
    parser.add_argument(
        "--clean_up_activations",
        action="store_true",
        help="Clean up activations after evaluation",
    )
    parser.add_argument(
        "--save_activations",
        action="store_false",
        help="Save the generated LLM activations for later use",
    )

    def str_to_bool(value):
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise argparse.ArgumentTypeError("Boolean value expected.")

    parser.add_argument(
        "--perform_scr",
        type=str_to_bool,
        required=True,
        help="If true, do Spurious Correlation Removal (SCR). If false, do TPP.",
    )
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=None,
        help="Batch size for LLM. If None, uses a default based on the model name.",
    )
    parser.add_argument(
        "--llm_dtype",
        type=str,
        default=None,
        choices=[None, "float32", "float64", "float16", "bfloat16"],
        help="Data type for LLM. If None, a default is used.",
    )
    parser.add_argument(
        "--sae_batch_size",
        type=int,
        default=None,
        help="Batch size for SAE. If None, default config value is used.",
    )
    parser.add_argument(
        "--lower_vram_usage",
        action="store_true",
        help="Lower GPU memory usage by moving model to CPU when not required.",
    )

    return parser


if __name__ == "__main__":
    args = arg_parser().parse_args()
    device = general_utils.setup_environment()

    start_time = time.time()

    config, selected_saes = create_config_and_selected_saes(args)
    print(selected_saes)

    os.makedirs(args.output_folder, exist_ok=True)

    results_dict = run_eval(
        config,
        selected_saes,
        device,
        args.output_folder,
        args.force_rerun,
        args.clean_up_activations,
        args.save_activations,
    )

    end_time = time.time()
    print(f"Finished evaluation in {end_time - start_time} seconds")
