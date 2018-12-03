#!/usr/bin/env python3
from math import ceil
import time

import numpy as np
import dynet as dy
from dynn import io

from .util import Logger
from .evaluation import bleu_score
from .objectives import NLLObjective


def train_epoch(epoch, model, objective, optimizer, train_batches, log=None):
    log = log or Logger()
    start = time.time()
    running_loss = n_processed = 0
    for src, tgt in train_batches:
        # Renew the computation graph
        dy.renew_cg()
        # Initialize layers
        model.init(test=False, update=True)
        # Compute logits
        logits = model(src, tgt)
        # Compute losses
        loss = objective(logits, tgt)
        # Backward pass
        loss.backward()
        # Update the parameters
        optimizer.update()
        # Keep track of the running loss
        n_tgt_tokens = sum(tgt.lengths)
        running_loss += loss.value() * n_tgt_tokens
        n_processed += n_tgt_tokens
        # Print the current loss from time to time
        if train_batches.just_passed_multiple(ceil(len(train_batches) / 10)):
            running_loss /= n_processed
            ppl = np.exp(running_loss)
            elapsed = time.time() - start
            tok_per_s = n_processed / elapsed
            log(
                f"Epoch {epoch}@{train_batches.percentage_done():.0f}%: "
                f"loss={running_loss:.3f} ppl={ppl:.2f} "
                f"({elapsed:.1f}s, {tok_per_s:.1f} tok/s)"
            )
            running_loss = n_processed = 0
            start = time.time()


def eval_ppl(model, eval_batches, log=None):
    """Evaluate model perplexity"""
    log = log or Logger()
    nll = 0
    nll_loss = NLLObjective()
    for src, tgt in eval_batches:
        # Renew the computation graph
        dy.renew_cg()
        # Initialize layers
        model.init(test=True, update=False)
        # Compute logits
        logits = model(src, tgt)
        # Aggregate NLL
        nll += nll_loss(logits, tgt).value() * sum(tgt.lengths)
    # Average NLL
    nll /= eval_batches.tgt_size
    # Perplexity
    ppl = np.exp(nll)
    return ppl


def eval_bleu(translator, eval_batches, src_words, tgt_words, detok, log=None):
    """Compute BLEU score over a given dataset"""
    log = log or Logger()
    hyps = []
    refs = []
    # Generate from the source data
    for src, tgt in eval_batches:
        # Retrieve original source and target words
        batch_src_words = src_words[src.original_idxs]
        batch_tgt_words = tgt_words[tgt.original_idxs]
        # Translate
        hyp_sents = translator(src, src_word=batch_src_words)
        # Record hypotheses
        hyps.extend(hyp_sents)
        # Also record the detokenized reference
        refs.extend([translator.detok(words) for words in batch_tgt_words])
    # BLEU
    bleu = bleu_score(hyps, refs)
    return bleu


def translate(translator, eval_batches, src_words, log=None):
    """Translate a batch of source sentences (tokenized and numberized)"""
    log = log or Logger()
    src_words = np.asarray(src_words)
    idx_to_hyp = {}
    start = time.time()
    n_processed = 0
    # Generate from the source data
    for src in eval_batches:
        # Retrieve original source and target words
        batch_src_words = src_words[src.original_idxs]
        # Translate
        hyp_sents = translator(src, src_words=batch_src_words)
        # Record hypotheses
        for b, idx in enumerate(src.original_idxs):
            idx_to_hyp[idx] = hyp_sents[b]
        # Number of tokens translated
        n_processed += sum(src.lengths)
        # Print progress
        if eval_batches.just_passed_multiple(ceil(len(eval_batches) / 10)):
            elapsed = time.time() - start
            tok_per_s = n_processed / elapsed
            log(
                f"{eval_batches.percentage_done():.0f}% done "
                f"({elapsed:.1f}s, {tok_per_s:.1f} tok/s)"
            )
            n_processed = 0
            start = time.time()
    # Return hypotheses in order
    hyps = [idx_to_hyp[i] if i in idx_to_hyp else ""
            for i in range(len(idx_to_hyp))]
    return hyps


def translate_sents_itr(translator, src_sents, log=None):
    """Translate a list of sentences"""
    log = log or Logger()
    for src_sent in src_sents:
        yield translator(src_sent)


def train(
    args,
    model,
    objective,
    optimizer,
    train_batches,
    valid_batches,
    log=None
):
    log = log or Logger()
    # Start training
    log("Starting training")
    best_ppl = np.inf
    deadline = 0
    for epoch in range(1, args.n_epochs + 1):
        # Train for one epoch
        train_epoch(epoch, model, objective, optimizer, train_batches, log)
        # Validate
        valid_ppl = eval_ppl(model, valid_batches, log)
        log(f"Epoch {epoch}: validation ppl {valid_ppl:.1f}")
        # Early stopping
        if valid_ppl < best_ppl:
            deadline = 0
            best_ppl = valid_ppl
            io.save(model.pc, args.model_save_file)
        else:
            if deadline > args.patience:
                log(
                    f"Early stopping t epoch {epoch} after {deadline} of no "
                    f"improvement. Best validation perplexity: {best_ppl:.1f}"
                )
                break
            else:
                deadline += 1
                if args.lr_decay != 1:
                    log("Decreasing learning rate")
                    optimizer.decay_lr(args.lr_decay)
                    log(f"New learning rate: {optimizer.learning_rate}")
    # Load best model yet
    io.populate(model.pc, args.model_save_file)
