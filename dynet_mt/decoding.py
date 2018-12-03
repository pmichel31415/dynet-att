#!/usr/bin/env python3
import dynet as dy
import numpy as np


class Beam(object):
    def __init__(
            self,
            state,
            context,
            words,
            logprob,
            align=[0],
            is_over=False
    ):
        self.state = state
        self.words = words
        self.context = context
        self.logprob = logprob
        self.align = align
        self.is_over = is_over


class Decoding(object):

    def __call__(self, src):
        raise NotImplementedError()


class BeamSearch(Decoding):

    def __init__(self, beam_size, lenpen):
        self.beam_size = beam_size
        self.lenpen = lenpen

    def __call__(self, model, src):
        dy.renew_cg()
        model.init(test=True, update=False)
        # Handle batch size
        batch_size = src.batch_size
        # Defer batch size > 1 to multiple calls
        if batch_size > 1:
            sents, aligns = [], []
            for b in range(batch_size):
                sent, align = self.__call__(model, src[b])
                sents.append(sent[0])
                aligns.append(align[0])
            return sents, aligns
        # Encode
        # ------
        X = model.encode(src)
        # Decode
        # ------
        # Mask for attention
        attn_mask = src.get_mask(base_val=0, mask_val=-np.inf)
        # Max length
        max_len = 2 * src.max_length
        # Initialize beams
        first_beam = {
            "wemb": model.sos,  # Previous word embedding
            "state": model.initial_decoder_state,  # Decoder state
            "score": 0.0,  # score
            "words": [],  # generated words
            "align": [],  # Alignments given by attention
            "is_over": False,  # is over
        }
        beams = [first_beam]
        # Start decoding
        while not beams[-1]["is_over"] and len(beams[-1]["words"]) < max_len:
            new_beams = []
            active_idxs = [idx for idx, beam in enumerate(beams)
                           if not beam["is_over"]]
            active = [beams[idx] for idx in active_idxs]
            states, log_ps, aligns = model.decode_step(
                X,
                dy.concatenate_to_batch([beam["wemb"] for beam in active]),
                model.batch_states([beam["state"]for beam in active]),
                attn_mask,
            )
            for b_i, beam in enumerate(beams):
                # Don't do anything if the beam is over
                if beam["is_over"]:
                    new_beams.append(beam)
                    continue
                # Retrieve log_p, alignement and state for this beam
                log_p = log_ps[active_idxs[b_i]]
                align = aligns[active_idxs[b_i]]
                state = model.pick_state_batch_elem(states, active_idxs[b_i])
                # top k words
                next_words = log_p.argsort()[-self.beam_size:]
                # Add to new beam
                for word in next_words:
                    # Handle stop condition
                    if word == model.dic_tgt.eos_idx:
                        new_beam = {
                            "words": beam["words"],
                            "score": beam["score"] + log_p[word],
                            "align": beam["align"],
                            "is_over": True,
                        }
                    else:
                        new_beam = {
                            "wemb": model.embed_word(word, tgt=True),
                            "state": state,
                            "words": beam["words"] + [word],
                            "score": beam["score"] + log_p[word],
                            "align": beam["align"] + [align],
                            "is_over": False,
                        }
                    new_beams.append(new_beam)

            def beam_score(beam):
                """Helper to score a beam with length penalty"""
                return beam["score"] / (len(beam["words"]) + 1)**self.lenpen
            # Only keep topk new beams
            beams = sorted(new_beams, key=beam_score)[-self.beam_size:]

        # Return top beam
        return [beams[-1]["words"]], [beams[-1]["align"]]
