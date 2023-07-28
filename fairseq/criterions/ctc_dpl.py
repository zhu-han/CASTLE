# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional
import collections

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import post_process
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round


@dataclass
class CtcDplCriterionConfig(FairseqDataclass):
    pl_weight: float = field(
        default="1.0",
        metadata={"help": "weight of pseudo-labeling loss"},
    )
    pl_start_updates: int = field(
        default=-1,
        metadata={"help": "start updates for the use of the pseudo-labeling loss"},
    )
    ema_decay_factor: float = field(
        default=0.0,
        metadata={"help": "decay factor for exponential moving average of the teacher model (0.0 is always use current model)"},
    )
    ema_update: int = field(
        default=1,
        metadata={"help": "decay every ema_update updates for exponential moving average of the teacher model"},
    )
    confidence_threshold: float = field(
        default="0.0",
        metadata={"help": "confidence threshold"},
    )
    addition_mask_channel_prob: float = field(
        default=-1.0,
        metadata={"help": "channel mask prob for target domain (use it if >= 0.0)"},
    )
    addition_mask_prob: float = field(
        default=-1.0,
        metadata={"help": "time mask prob for target domain (use it if >= 0.0)"},
    )
    two_proj: bool = field(
        default=False,
        metadata={"help": "whether use two proj"},
    )
    two_proj_weight: float = field(
        default=1.0,
        metadata={"help": "loss weight of the two proj"},
    )
    
    # original ctc related
    zero_infinity: bool = field(
        default=False,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    post_process: str = field(
        default="letter",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
            "wordpiece, BPE symbols, etc. "
            "See fairseq.data.data_utils.post_process() for full list of options"
        },
    )
    wer_kenlm_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "if this is provided, use kenlm to compute wer (along with other wer_* args)"
        },
    )
    wer_lexicon: Optional[str] = field(
        default=None,
        metadata={"help": "lexicon to use with wer_kenlm_model"},
    )
    wer_lm_weight: float = field(
        default=2.0,
        metadata={"help": "lm weight to use with wer_kenlm_model"},
    )
    wer_word_score: float = field(
        default=-1.0,
        metadata={"help": "lm word score to use with wer_kenlm_model"},
    )

    wer_args: Optional[str] = field(
        default=None,
        metadata={
            "help": "DEPRECATED: tuple of (wer_kenlm_model, wer_lexicon, wer_lm_weight, wer_word_score)"
        },
    )


@register_criterion("ctc_dpl", dataclass=CtcDplCriterionConfig)
class CtcDplCriterion(FairseqCriterion):
    def __init__(self, cfg: CtcDplCriterionConfig, task: FairseqTask):
        super().__init__(task)
        self.blank_idx = task.target_dictionary.index(task.blank_symbol) if hasattr(task, 'blank_symbol') else 0
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.post_process = cfg.post_process

        if cfg.wer_args is not None:
            (
                cfg.wer_kenlm_model,
                cfg.wer_lexicon,
                cfg.wer_lm_weight,
                cfg.wer_word_score,
            ) = eval(cfg.wer_args)

        if cfg.wer_kenlm_model is not None:
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            dec_args = Namespace()
            dec_args.nbest = 1
            dec_args.criterion = "ctc"
            dec_args.kenlm_model = cfg.wer_kenlm_model
            dec_args.lexicon = cfg.wer_lexicon
            dec_args.beam = 50
            dec_args.beam_size_token = min(50, len(task.target_dictionary))
            dec_args.beam_threshold = min(50, len(task.target_dictionary))
            dec_args.lm_weight = cfg.wer_lm_weight
            dec_args.word_score = cfg.wer_word_score
            dec_args.unk_weight = -math.inf
            dec_args.sil_weight = 0

            self.w2l_decoder = W2lKenLMDecoder(dec_args, task.target_dictionary)
        else:
            self.w2l_decoder = None

        self.zero_infinity = cfg.zero_infinity
        self.sentence_avg = cfg.sentence_avg

        self.epoch = None
        self.pl_weight = cfg.pl_weight
        
        self.pl_start_updates = cfg.pl_start_updates
           
        self.ema_decay_factor = cfg.ema_decay_factor
        self.ema_update = cfg.ema_update
        self.last_ema_update = 0
        self.teacher_model = None

        self.target_dictionary = task.target_dictionary
        self.teacher_model_state_dict = None

        self.addition_mask_channel_prob = cfg.addition_mask_channel_prob
        self.addition_mask_prob = cfg.addition_mask_prob
        self.confidence_threshold = cfg.confidence_threshold
        self.two_proj = cfg.two_proj
        self.two_proj_weight = cfg.two_proj_weight

    def forward(self, model, sample, reduce=True):
        aux_target = sample["aux_target"] # B
        source_mask = aux_target == 0
        target_mask = aux_target == 1

        if model.w2v_encoder.num_updates >= self.pl_start_updates:
            if self.ema_decay_factor > 0.0:
                if self.teacher_model is None:
                    from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtc
                    from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecEncoder
                    from fairseq.models.wav2vec.wav2vec2_asr import Linear
                    device = next(model.parameters()).device
                    teacher_w2v_encoder = Wav2VecEncoder(model.cfg, self.target_dictionary)
                    if self.two_proj:
                        teacher_two_proj = Linear(model.cfg.encoder_embed_dim, len(self.target_dictionary))
                        teacher_w2v_encoder.add_module("two_proj", teacher_two_proj)
                    self.teacher_model = Wav2VecCtc(model.cfg, teacher_w2v_encoder).half().to(device)
                    self.teacher_model.load_state_dict(model.state_dict())
                    self.teacher_model_state_dict = self.teacher_model.state_dict()
                    self.teacher_model.eval()
                else:
                    if model.w2v_encoder.num_updates % self.ema_update == 0 and model.w2v_encoder.num_updates != self.last_ema_update:
                        self.last_ema_update = model.w2v_encoder.num_updates
                        self.teacher_model_state_dict = EMA(model.state_dict(), self.teacher_model_state_dict, self.ema_decay_factor)
                        self.teacher_model.load_state_dict(self.teacher_model_state_dict)
                        self.teacher_model.eval()
            elif self.ema_decay_factor == 0.0:
                self.teacher_model = model
                if model.training:
                    is_train = True
                    self.teacher_model.eval()
                else:
                    is_train = False
            else:
                raise NotImplementedError(f"EMA decay factor '{self.ema_decay_factor} < 0.0' is not right")

            with torch.no_grad():
                if model.w2v_encoder.num_updates >= self.pl_start_updates:
                    unmasked_target_net_output = self.teacher_model(**sample["net_input"], batch_mask=target_mask, no_mask=True)
                if self.ema_decay_factor == 0.0:
                    if is_train:
                        self.teacher_model.train()
                        is_train = False
        
        if model.training:
            source_net_output = model(**sample["net_input"], batch_mask=source_mask)
        if not model.training or model.w2v_encoder.num_updates >= self.pl_start_updates: 
            target_net_output = model(**sample["net_input"], batch_mask=target_mask, addition_mask_prob=self.addition_mask_prob, addition_mask_channel_prob=self.addition_mask_channel_prob)

        if model.training:
            # When semi-supervised training, using source data for ctc computation
            ctc_mask = source_mask
            ctc_net_output = source_net_output
        else:
            # When validation, using target data for ctc computation
            ctc_mask = target_mask
            ctc_net_output = target_net_output

        # CTC loss computation
        
        ## get log probs
        lprobs = model.get_normalized_probs(
            ctc_net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder

        ## get target
        ctc_target = sample["target"][ctc_mask]

        pad_mask = (ctc_target != self.pad_idx) & (
            ctc_target != self.eos_idx
        )
        targets_flat = ctc_target.masked_select(pad_mask)

        ## get input lengths
        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
            input_lengths = input_lengths[ctc_mask]
        else:
            non_padding_mask = ~ctc_net_output["padding_mask"]
            input_lengths = non_padding_mask.long().sum(-1)
        
        ## get target lengths
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)
        target_lengths = target_lengths[ctc_mask]

        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )
        if self.two_proj:
            ## get log probs
            lprobs_two = model.get_two_proj_normalized_probs(
                ctc_net_output, log_probs=True
            ).contiguous()  # (T, B, C) from the encoder
            with torch.backends.cudnn.flags(enabled=False):
                ctc_loss += self.two_proj_weight * F.ctc_loss(
                    lprobs_two,
                    targets_flat,
                    input_lengths,
                    target_lengths,
                    blank=self.blank_idx,
                    reduction="sum",
                    zero_infinity=self.zero_infinity,
                )
    
        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = ctc_mask.sum() if self.sentence_avg else ntokens

        pseudo_w_errs = 0
        pseudo_w_len = 0
        pseudo_c_errs = 0
        pseudo_c_len = 0
        pseudo_selected_w_errs = 0
        pseudo_selected_w_len = 0
        pseudo_selected_c_errs = 0
        pseudo_selected_c_len = 0
        # PL loss computation
        if model.w2v_encoder.num_updates >= self.pl_start_updates:

            ## get input lengths
            if "src_lengths" in sample["net_input"]:
                pl_input_lengths = sample["net_input"]["src_lengths"]
                pl_input_lengths = pl_input_lengths[target_mask]
            else:
                pl_non_padding_mask = ~target_net_output["padding_mask"]
                pl_input_lengths = pl_non_padding_mask.long().sum(-1)

            ## get target
            truth_pl_ctc_target = sample["target"][target_mask]

            truth_pl_pad_mask = (truth_pl_ctc_target != self.pad_idx) & (
                truth_pl_ctc_target != self.eos_idx
            )



            pl_masked_lprobs = model.get_normalized_probs(
                target_net_output, log_probs=True
            ).transpose(0,1) # (T, B, C) -> (B, T, C)

            with torch.no_grad():
                if self.two_proj:
                    pl_unmasked_lprobs = self.teacher_model.get_two_proj_normalized_probs(
                        unmasked_target_net_output, log_probs=True, dropout=False
                    ).transpose(0,1)  # (T, B, C) -> (B, T, C)
                else:
                    pl_unmasked_lprobs = self.teacher_model.get_normalized_probs(
                            unmasked_target_net_output, log_probs=True
                        ).transpose(0,1)  # (T, B, C) -> (B, T, C)

            pl_ctc_targets_flat = None
            pl_target_lengths = []
            confidences = []
            confidence_mask = []
            if self.two_proj:
                for pl_unmasked_lprob, pl_inp_l, truth_units_arr, truth_units_arr_pad_mask in zip(
                    pl_unmasked_lprobs,
                    pl_input_lengths,
                    truth_pl_ctc_target,
                    truth_pl_pad_mask
                ):
                    pl_unmasked_lprob = pl_unmasked_lprob[:pl_inp_l]
                    max_lprobs, max_toks = pl_unmasked_lprob.max(dim=-1)
                    confidence = torch.exp(max_lprobs[max_toks != self.blank_idx].mean())
                    toks = max_toks.unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx]
                    confidences.append(confidence)
                    import editdistance
                    truth_units_arr = truth_units_arr[truth_units_arr_pad_mask]
                    truth_c_err = editdistance.eval(pred_units_arr.tolist(), truth_units_arr.tolist())
                    truth_c_len = len(truth_units_arr.tolist())
                    pseudo_c_errs += truth_c_err
                    pseudo_c_len += truth_c_len

                    pred_units = self.task.target_dictionary.string(pred_units_arr.tolist())
                    pred_words = post_process(pred_units, self.post_process).split()
                    truth_units = self.task.target_dictionary.string(truth_units_arr.tolist())
                    truth_words = post_process(truth_units, self.post_process).split()
                    truth_w_err = editdistance.eval(pred_words, truth_words)
                    truth_w_len = len(truth_words)
                    pseudo_w_errs += truth_w_err
                    pseudo_w_len += truth_w_len

                    if (self.confidence_threshold <= 0 or confidence > self.confidence_threshold):
                        pseudo_selected_c_errs += truth_c_err
                        pseudo_selected_c_len += truth_c_len
                        pseudo_selected_w_errs += truth_w_err
                        pseudo_selected_w_len += truth_w_len
                        pl_target_lengths.append(pred_units_arr.size(0))   
                        if pl_ctc_targets_flat == None:
                            pl_ctc_targets_flat = pred_units_arr
                        else:
                            pl_ctc_targets_flat = torch.cat((pl_ctc_targets_flat, pred_units_arr), 0)
                        confidence_mask.append(True)
                    else:
                        confidence_mask.append(False)
            else:
                for pl_unmasked_lprob, pl_inp_l, truth_units_arr, truth_units_arr_pad_mask in zip(
                    pl_unmasked_lprobs,
                    pl_input_lengths,
                    truth_pl_ctc_target,
                    truth_pl_pad_mask
                ):
                    pl_unmasked_lprob = pl_unmasked_lprob[:pl_inp_l]
                    max_lprobs, max_toks = pl_unmasked_lprob.max(dim=-1)
                    confidence = torch.exp(max_lprobs[max_toks != self.blank_idx].mean())
                    toks = max_toks.unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx]
                    pred_units_arr = pred_units_arr
                                        
                    confidences.append(confidence)

                    import editdistance
                    truth_units_arr = truth_units_arr[truth_units_arr_pad_mask]
                    truth_c_err = editdistance.eval(pred_units_arr.tolist(), truth_units_arr.tolist())
                    truth_c_len = len(truth_units_arr.tolist())
                    pseudo_c_errs += truth_c_err
                    pseudo_c_len += truth_c_len


                    pred_units = self.task.target_dictionary.string(pred_units_arr.tolist())
                    pred_words = post_process(pred_units, self.post_process).split()
                    truth_units = self.task.target_dictionary.string(truth_units_arr.tolist())
                    truth_words = post_process(truth_units, self.post_process).split()
                    truth_w_err = editdistance.eval(pred_words, truth_words)
                    truth_w_len = len(truth_words)
                    pseudo_w_errs += truth_w_err
                    pseudo_w_len += truth_w_len

                    if self.confidence_threshold <= 0 or confidence > self.confidence_threshold:
                        pseudo_selected_c_errs += truth_c_err
                        pseudo_selected_c_len += truth_c_len
                        pseudo_selected_w_errs += truth_w_err
                        pseudo_selected_w_len += truth_w_len
                        pl_target_lengths.append(pred_units_arr.size(0))   
                        if pl_ctc_targets_flat == None:
                            pl_ctc_targets_flat = pred_units_arr
                        else:
                            pl_ctc_targets_flat = torch.cat((pl_ctc_targets_flat, pred_units_arr), 0)
                        confidence_mask.append(True)
                    else:
                        confidence_mask.append(False)
                        
            pl_target_lengths = torch.as_tensor(pl_target_lengths, dtype=pred_units_arr.dtype, device=pred_units_arr.device)
            confidences = torch.as_tensor(confidences, dtype=pl_unmasked_lprob.dtype, device=pl_unmasked_lprob.device)
            confidence_mask = torch.as_tensor(confidence_mask, device=pl_unmasked_lprob.device)

            pl_masked_lprobs = pl_masked_lprobs[confidence_mask].transpose(0,1).contiguous()
            pl_input_lengths = pl_input_lengths[confidence_mask]
            target_sentence = confidence_mask.size(0)
            selected_target_sentence = confidence_mask.sum()

            if pl_target_lengths.size(0) != 0:
                with torch.backends.cudnn.flags(enabled=False):
                    pl_loss = F.ctc_loss(
                        pl_masked_lprobs,
                        pl_ctc_targets_flat,
                        pl_input_lengths,
                        pl_target_lengths,
                        blank=self.blank_idx,
                        reduction="sum",
                        zero_infinity=self.zero_infinity,
                    ) 
            else:
                pl_loss = torch.zeros((1), device=sample["target"].device)
        
        else:
            pl_loss = torch.zeros((1), device=sample["target"].device)
            target_sentence = torch.zeros((1), device=sample["target"].device)
            selected_target_sentence = torch.zeros((1), device=sample["target"].device)


        # Total loss
        loss = ctc_loss

        if model.w2v_encoder.num_updates >= self.pl_start_updates:
            loss = loss + self.pl_weight * pl_loss
            
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ctc_loss": utils.item(ctc_loss.data),
            "pl_loss": utils.item(pl_loss.data),
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            "target_sentence": target_sentence,
            "selected_target_sentence": selected_target_sentence,
        }
        logging_output["pseudo_w_errors"] = pseudo_w_errs
        logging_output["pseudo_w_total"] = pseudo_w_len
        logging_output["pseudo_c_errors"] = pseudo_c_errs
        logging_output["pseudo_c_total"] = pseudo_c_len
        logging_output["pseudo_selected_w_errors"] = pseudo_selected_w_errs
        logging_output["pseudo_selected_w_total"] = pseudo_selected_w_len
        logging_output["pseudo_selected_c_errors"] = pseudo_selected_c_errs
        logging_output["pseudo_selected_c_total"] = pseudo_selected_c_len

        if not model.training:
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                    lprobs_t,
                    sample["target_label"][ctc_mask]
                    if "target_label" in sample
                    else sample["target"][ctc_mask],
                    input_lengths,
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    decoded = None
                    if self.w2l_decoder is not None:
                        decoded = self.w2l_decoder.decode(lp)
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]

                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ctc_loss_sum = utils.item(sum(log.get("ctc_loss", 0) for log in logging_outputs))
        pl_loss_sum = utils.item(sum(log.get("pl_loss", 0) for log in logging_outputs))

        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        target_sentence = utils.item(
            sum(log.get("target_sentence", 0) for log in logging_outputs)
        )
        selected_target_sentence = utils.item(
            sum(log.get("selected_target_sentence", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ctc_loss", ctc_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "pl_loss", pl_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        metrics.log_scalar("target_sentence", nsentences)
        metrics.log_scalar("target_proportion", target_sentence / nsentences, ntokens, round=3)
        metrics.log_scalar("selected_target_sentence", selected_target_sentence)
        metrics.log_scalar("selected_proportion", selected_target_sentence / target_sentence if target_sentence != 0 else target_sentence, ntokens, round=3)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", ctc_loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        pseudo_c_errors = sum(log.get("pseudo_c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("pseudo_c_errors", pseudo_c_errors)
        pseudo_c_total = sum(log.get("pseudo_c_total", 0) for log in logging_outputs)
        metrics.log_scalar("pseudo_c_total", pseudo_c_total)
        pseudo_w_errors = sum(log.get("pseudo_w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("pseudo_w_errors", pseudo_w_errors)
        pseudo_w_total = sum(log.get("pseudo_w_total", 0) for log in logging_outputs)
        metrics.log_scalar("pseudo_w_total", pseudo_w_total)
        pseudo_selected_c_errors = sum(log.get("pseudo_selected_c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("pseudo_selected_c_errors", pseudo_selected_c_errors)
        pseudo_selected_c_total = sum(log.get("pseudo_selected_c_total", 0) for log in logging_outputs)
        metrics.log_scalar("pseudo_selected_c_total", pseudo_selected_c_total)
        pseudo_selected_w_errors = sum(log.get("pseudo_selected_w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("pseudo_selected_w_errors", pseudo_selected_w_errors)
        pseudo_selected_w_total = sum(log.get("pseudo_selected_w_total", 0) for log in logging_outputs)
        metrics.log_scalar("pseudo_selected_w_total", pseudo_selected_w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

        if  pseudo_c_total > 0:
            metrics.log_derived(
                "pseudo_uer",
                lambda meters: safe_round(
                    meters["pseudo_c_errors"].sum * 100.0 / meters["pseudo_c_total"].sum, 3
                )
                if meters["pseudo_c_total"].sum > 0
                else float("nan"),
            )
        if pseudo_w_total > 0:
            metrics.log_derived(
                "pseudo_wer",
                lambda meters: safe_round(
                    meters["pseudo_w_errors"].sum * 100.0 / meters["pseudo_w_total"].sum, 3
                )
                if meters["pseudo_w_total"].sum > 0
                else float("nan"),
            )
        if  pseudo_selected_c_total > 0:
            metrics.log_derived(
                "pseudo_selected_uer",
                lambda meters: safe_round(
                    meters["pseudo_selected_c_errors"].sum * 100.0 / meters["pseudo_selected_c_total"].sum, 3
                )
                if meters["pseudo_selected_c_total"].sum > 0
                else float("nan"),
            )
        if pseudo_selected_w_total > 0:
            metrics.log_derived(
                "pseudo_selected_wer",
                lambda meters: safe_round(
                    meters["pseudo_selected_w_errors"].sum * 100.0 / meters["pseudo_selected_w_total"].sum, 3
                )
                if meters["pseudo_selected_w_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


def EMA(new_model, last_model, decay_factor):
    model_params_keys = list(new_model.keys())
    params_keys = list(last_model.keys())
    if params_keys != model_params_keys:
        raise KeyError(
            "expected list of params: {}, "
            "but found: {}".format(params_keys, model_params_keys)
        )

    latest_model = collections.OrderedDict()

    for k in params_keys:
        p_new = new_model[k].float()
        p_last = last_model[k].float()
        latest_model[k] = decay_factor * p_last + (1 - decay_factor) * p_new
    return latest_model
