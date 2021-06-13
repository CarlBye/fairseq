# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion

# from fairseq.distill.teacher_model import roberta_0, roberta_1, roberta_2, roberta_3

@register_criterion('sentence_prediction_distill')
class SentencePredictionDistillCriterion(FairseqCriterion):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--save-predictions', metavar='FILE',
                            help='file to save predictions to')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert hasattr(model, 'classification_heads') and \
            'sentence_classification_head' in model.classification_heads, \
            "model must provide sentence classification head for --criterion=sentence_prediction_distill"

        # lprobs_teacher = self.get_avg(sample, True) #avglogits
        lprobs_teacher = self.get_avg(sample, False) #avgprobs

        logits_student, _ = model(
            **sample['net_input'],
            features_only=True,
            classification_head_name='sentence_classification_head',
        )
        targets = model.get_targets(sample, [logits_student]).view(-1)
        sample_size = targets.numel()
        lprobs_student_log = F.log_softmax(logits_student, dim=-1, dtype=torch.float32)

        # if not self.args.regression_target:
        #     loss = F.nll_loss(
        #         F.log_softmax(logits, dim=-1, dtype=torch.float32),
        #         targets,
        #         reduction='sum',
        #     )
        # else:
        #     logits = logits.squeeze().float()
        #     targets = targets.float()
        #     loss = F.mse_loss(
        #         logits,
        #         targets,
        #         reduction='sum',
        #     )

        loss_nll = F.nll_loss(lprobs_student_log, targets, reduction='sum')
        loss_kl = F.kl_div(lprobs_student_log, lprobs_teacher, reduction="sum")

        loss = 0.5 * loss_kl + 0.5 * loss_nll
        # loss = loss_kl

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }

        if not self.args.regression_target:
            preds = logits_student.max(dim=1)[1]
            logging_output.update(
                ncorrect=(preds == targets).sum().item()
            )
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            agg_output.update(accuracy=ncorrect/nsentences)

        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output

    def get_avg(self, sample, isLogits):
        roberta_0.cuda()
        roberta_1.cuda()
        roberta_2.cuda()
        roberta_3.cuda()

        roberta_0.eval()
        roberta_1.eval()
        roberta_2.eval()
        roberta_3.eval()

        logits_0, _ = roberta_0.model(
            **sample["net_input"],
            features_only=True,
            classification_head_name='sentence_classification_head',
        )
        logits_1, _ = roberta_1.model(
            **sample["net_input"],
            features_only=True,
            classification_head_name='sentence_classification_head',
        )
        logits_2, _ = roberta_2.model(
            **sample["net_input"],
            features_only=True,
            classification_head_name='sentence_classification_head',
        )
        logits_3, _ = roberta_3.model(
            **sample["net_input"],
            features_only=True,
            classification_head_name='sentence_classification_head',
        )

        if isLogits:
            avg_logits = (logits_0 + logits_1 + logits_2 + logits_3) / 4
            lprobs_teacher = F.softmax(avg_logits, dim=-1, dtype=torch.float32)
        else:
            prob_0 = F.softmax(logits_0, dim=-1, dtype=torch.float32)
            prob_1 = F.softmax(logits_1, dim=-1, dtype=torch.float32)
            prob_2 = F.softmax(logits_2, dim=-1, dtype=torch.float32)
            prob_3 = F.softmax(logits_3, dim=-1, dtype=torch.float32)
            lprobs_teacher = (prob_0 + prob_1 + prob_2 + prob_3) / 4
            # lprobs_teacher = F.softmax(avg_prob, dim=-1, dtype=torch.float32)
        return lprobs_teacher