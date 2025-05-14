import torch
import torchmetrics.text as tm_text
import subprocess
import os

from typing import List, Dict, Any, Optional


NODE_RENDER_SCRIPT_NAME = "ml_pipeline/js/validate_md.js"


class RenderMarkdownMetric:
    def __init__(self, node_script_path: str, log_obj: Optional[Any] = None):
        self.node_script_path = node_script_path
        self.renderable_count = 0
        self.total_count = 0
        self.log_obj = log_obj
        self.script_found = True

        if not os.path.exists(self.node_script_path):
            msg = (
                f"Node.js markdown render script not found at: {self.node_script_path}"
            )
            self.script_found = False
            if self.log_obj:
                self.log_obj.info(f"CRITICAL_METRIC_SETUP_ERROR: {msg}")
            else:
                print(f"CRITICAL_METRIC_SETUP_ERROR: {msg}")

    def update(self, markdown_strings: List[str]):
        if not self.script_found:
            return

        for md_string in markdown_strings:
            self.total_count += 1
            try:
                # Pass markdown string directly as argument
                process = subprocess.run(
                    ["node", self.node_script_path, md_string],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=15,
                )
                if process.returncode == 0:
                    self.renderable_count += 1
                else:
                    if self.log_obj:
                        err_snippet = (
                            process.stderr[:500].replace("\n", " ")
                            if process.stderr
                            else "No stderr"
                        )
                        # Exit codes: 1=renderError, 2=missingArg, 3=importError, 4=pluginError
                        self.log_obj.info(
                            f"MarkdownRenderScriptError (Code {process.returncode}): {err_snippet}"
                        )
            except subprocess.TimeoutExpired:
                if self.log_obj:
                    self.log_obj.info(f"MarkdownRenderScript timed out for a string.")
            except (
                subprocess.CalledProcessError
            ) as e:
                if self.log_obj:
                    self.log_obj.info(
                        f"MarkdownRenderScript CalledProcessError: {e.stderr[:500] if e.stderr else e}"
                    )
            except (
                Exception
            ) as e:
                if self.log_obj:
                    self.log_obj.info(
                        f"Exception calling markdown render script: {type(e).__name__} - {str(e)}"
                    )

    def compute(self) -> float:
        if not self.script_found or self.total_count == 0:
            return -1.0
        return self.renderable_count / self.total_count

    def reset(self):
        self.renderable_count = 0
        self.total_count = 0


class Seq2SeqMetricsTracker:
    def __init__(
        self,
        tokenizer: Any,
        device: torch.device,
        log_obj: Optional[Any] = None,
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.log_obj = log_obj

        self.bleu = tm_text.BLEUScore().to(self.device)
        self.rouge = tm_text.ROUGEScore().to(self.device)
        self.wer = tm_text.WordErrorRate().to(self.device)
        self.cer = tm_text.CharErrorRate().to(self.device)

        self.render_metric = RenderMarkdownMetric(NODE_RENDER_SCRIPT_NAME, log_obj)

        self.loss_values: List[float] = []
        self.prediction_samples: List[Dict[str, str]] = []
        self.max_samples_to_log = 5

    def update(
        self,
        generated_ids: torch.Tensor,
        reference_ids: torch.Tensor,
        loss_val: Optional[float] = None,
    ):
        decoded_preds = self.tokenizer.batch_decode(
            generated_ids.cpu(), skip_special_tokens=True
        )
        decoded_targets = []
        for single_ref_ids in reference_ids:
            ids_cpu_list = single_ref_ids.cpu().tolist()
            valid_ids = [token_id for token_id in ids_cpu_list if token_id != -100]

            decoded_text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
            decoded_targets.append(decoded_text.strip()) 

        list_of_list_targets = [[t] for t in decoded_targets]

        self.bleu.update(decoded_preds, list_of_list_targets)
        self.rouge.update(decoded_preds, decoded_targets)
        self.wer.update(decoded_preds, decoded_targets)
        self.cer.update(decoded_preds, decoded_targets)
        self.render_metric.update(decoded_preds)

        if loss_val is not None:
            self.loss_values.append(loss_val)

        if len(self.prediction_samples) < self.max_samples_to_log:
            for i in range(len(decoded_preds)):
                if len(self.prediction_samples) < self.max_samples_to_log:
                    self.prediction_samples.append(
                        {"generated": decoded_preds[i], "reference": decoded_targets[i]}
                    )
                else:
                    break

    def compute_all(self) -> Dict[str, float]:
        metrics_results: Dict[str, float] = {}
        if self.loss_values:
            metrics_results["loss"] = sum(self.loss_values) / len(self.loss_values)
        else:
            metrics_results["loss"] = 0.0

        metrics_results["BLEU"] = self.bleu.compute().item()

        rouge_scores_dict = self.rouge.compute()
        for key, value_tensor in rouge_scores_dict.items():
            metrics_results[key.upper()] = value_tensor.item()

        metrics_results["WER"] = self.wer.compute().item()
        metrics_results["CER"] = self.cer.compute().item()
        metrics_results["RenderPercent"] = self.render_metric.compute()

        return metrics_results

    def get_prediction_samples(self) -> List[Dict[str, str]]:
        return self.prediction_samples.copy()

    def reset(self):
        self.bleu.reset()
        self.rouge.reset()
        self.wer.reset()
        self.cer.reset()
        self.render_metric.reset()
        self.loss_values = []
        self.prediction_samples = []
