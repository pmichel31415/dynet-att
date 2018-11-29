#!/usr/bin/env python3

import numpy as np
import dynet as dy

from dynn.layers import Affine
from dynn.layers import Embeddings
from dynn.layers import StackedLSTM
from dynn.layers import Transduction, Bidirectional
from dynn.layers import MLPAttention
from dynn.layers import Sequential

from dynn.operations import stack
from dynn.parameter_initialization import UniformInit

from .base import BaseSeq2Seq


class AttBiLSTM(BaseSeq2Seq):
    """This custom layer implements an attention BiLSTM model"""

    def __init__(
        self,
        dic_src,
        dic_tgt,
        n_layers,
        embed_dim,
        hidden_dim,
        dropout,
        tie_decoder_embeds=True,
        tie_all_embeds=False,
    ):
        super(AttBiLSTM, self).__init__(dic_src, dic_tgt)
        # Hyperparameters
        self.V_src = len(dic_src)
        self.V_tgt = len(dic_tgt)
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        # Encoder
        # -------
        # Source Word embeddings
        if tie_all_embeds:
            if self.V_src != self.V_tgt:
                raise ValueError(
                    f"Can't tie source and target embeddings because the "
                    f"dictionary sizes don't match {self.V_src}!={self.V_tgt}"
                )
            init = UniformInit(1 / np.sqrt(embed_dim))
            E_src = self.pc.add_parameters((self.V_src, embed_dim), init=init)
        else:
            E_src = None
        self.src_embed = Embeddings(self.pc, dic_src, embed_dim, params=E_src)
        self.src_embed_all = Transduction(self.src_embed)
        # BiLSTM
        self.enc_fwd = StackedLSTM(
            self.pc,
            n_layers,
            embed_dim,
            hidden_dim,
            dropout,
            dropout
        )
        self.enc_bwd = StackedLSTM(
            self.pc,
            n_layers,
            embed_dim,
            hidden_dim,
            dropout,
            dropout
        )
        self.bilstm = Bidirectional(self.enc_fwd, self.enc_bwd)
        # Attention
        # --------
        self.attend = MLPAttention(
            self.pc,
            hidden_dim + embed_dim,
            hidden_dim,
            hidden_dim
        )
        # Decoder
        # -------
        # Target word embeddings
        if tie_all_embeds:
            E_tgt = E_src
        elif tie_decoder_embeds:
            init = UniformInit(1 / np.sqrt(embed_dim))
            E_tgt = self.pc.add_parameters((self.V_tgt, embed_dim), init=init)
        else:
            E_tgt = None
        self.tgt_embed = Embeddings(self.pc, dic_tgt, embed_dim, params=E_tgt)
        self.tgt_embed_all = Transduction(self.tgt_embed)
        # Start of sentence embedding
        self.sos_p = self.pc.add_lookup_parameters(
            (1, embed_dim),
            init=UniformInit(1 / np.sqrt(embed_dim)),
            name="sos",
        )
        # Recurrent decoder
        self.dec_cell = StackedLSTM(
            self.pc,
            n_layers,
            hidden_dim + embed_dim,
            hidden_dim,
            dropout,
            dropout
        )
        # Final projection layers
        self.project = Sequential(
            # First project to embedding dim
            Affine(self.pc, hidden_dim, embed_dim),
            # Then logit layer with weights tied to the word embeddings
            Affine(self.pc, embed_dim, self.V_tgt, dropout=dropout, W_p=E_tgt)
        )
        self.project_all = Transduction(self.project)

    @staticmethod
    def add_args(parser):
        bilstm_group = parser.add_argument_group("BiLSTM")
        bilstm_group.add_argument("--n-layers", type=int, default=2)
        bilstm_group.add_argument("--embed-dim", type=int, default=256)
        bilstm_group.add_argument("--hidden-dim", type=int, default=512)
        bilstm_group.add_argument("--dropout", type=float, default=0.2)
        bilstm_group.add_argument("--tie-decoder-embeds", action="store_true")
        bilstm_group.add_argument("--tie-all-embeds", action="store_true")

    @staticmethod
    def from_args(args, dic_src, dic_tgt):
        return AttBiLSTM(
            dic_src,
            dic_tgt,
            args.n_layers,
            args.embed_dim,
            args.hidden_dim,
            args.dropout,
            args.tie_decoder_embeds,
            args.tie_all_embeds
        )

    def init(self, test=False, update=True):
        self.src_embed_all.init(test=test, update=update)
        self.bilstm.init(test=test, update=update)
        self.attend.init(test=test, update=update)
        self.tgt_embed_all.init(test=test, update=update)
        self.dec_cell.init(test=test, update=update)
        self.project_all.init(test=test, update=update)

    def encode(self, src):
        # Embed input words
        src_embs = self.src_embed_all(src.sequences)
        # Encode
        fwd, bwd = self.bilstm(src_embs, lengths=src.lengths, output_only=True)
        # Sum forward and backward and concatenate all to a dh x L expression
        return stack(fwd, d=-1) + stack(bwd, d=-1)

    def logits(self, src, tgt):
        # Encode
        # ------
        X = self.encode(src)
        # Decode
        # ------
        # Mask for attention
        attn_mask = src.get_mask(base_val=0, mask_val=-np.inf)
        # Embed all words (except EOS)
        tgt_embs = [self.sos_p.batch([0] * tgt.batch_size)]
        tgt_embs.extend(self.tgt_embed_all([w for w in tgt.sequences[:-1]]))
        # Initialize decoder state
        dec_state = self.dec_cell.initial_value(tgt.batch_size)
        # Iterate over target words
        dec_outputs = []
        for x in tgt_embs:
            # Attention query: previous hidden state and current word embedding
            query = dy.concatenate([x, self.dec_cell.get_output(dec_state)])
            # Attend
            ctx, _ = self.attend(query, X, X, mask=attn_mask)
            # Both context and target word embedding will be fed to the decoder
            dec_input = dy.concatenate([x, ctx])
            # Update decoder state
            dec_state = self.dec_cell(dec_input, *dec_state)
            # Save output
            dec_outputs.append(self.dec_cell.get_output(dec_state))
        # Compute logits
        logits = self.project_all(dec_outputs)

        return logits

    @property
    def sos(self):
        return self.sos_p[0]

    @property
    def initial_decoder_state(self):
        return self.dec_cell.initial_value(1)

    def embed(self, word, tgt=False):
        if tgt:
            return self.tgt_embed(word)
        else:
            return self.src_embed(word)

    def decode_step(self, X, wemb, state, attn_mask=None):
        prev_h = self.dec_cell.get_output(state)
        query = dy.concatenate([wemb, prev_h])
        # Attend
        ctx, attn_weights = self.attend(query, X, X, mask=attn_mask)
        # Both context and target word embedding will be fed
        # to the decoder
        dec_input = dy.concatenate([wemb, ctx])
        # Update decoder state
        dec_state = self.dec_cell(dec_input, *state)
        # Save output
        h = self.dec_cell.get_output(dec_state)
        # Get log_probs
        log_p = dy.log_softmax(self.project(h)).npvalue()
        # alignments from attention
        align = attn_weights.npvalue().argmax()
        # Return
        return dec_state, log_p, align
