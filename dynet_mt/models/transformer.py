#!/usr/bin/env python3

import numpy as np
import dynet as dy

from dynn.layers import StackedTransformers, StackedCondTransformers
from dynn.layers import Embeddings
from dynn.layers import Affine
from dynn.layers import Sequential

from dynn.util import sin_embeddings
from dynn.parameter_initialization import UniformInit

from .base import BaseSeq2Seq


class Transformer(BaseSeq2Seq):
    """This custom layer implements a transofrmer model for translation"""

    def __init__(
            self,
            dic_src,
            dic_tgt,
            n_layers,
            embed_dim,
            hidden_dim,
            n_heads,
            dropout,
            tie_decoder_embeds=True,
            tie_all_embeds=False
    ):
        super(Transformer, self).__init__(dic_src, dic_tgt)
        # Hyperparameters
        self.V_src = len(dic_src)
        self.V_tgt = len(dic_tgt)
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        # Encoder
        # -------
        # Source word embeddings
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
        # Position embeddings
        self.pos_embeds = sin_embeddings(2000, embed_dim, transposed=True)
        # Encoder transformer
        self.enc = StackedTransformers(
            self.pc,
            n_layers,
            embed_dim,
            hidden_dim,
            n_heads,
            dropout=dropout
        )
        # Decoder
        # --------
        # Word embeddings
        # The embedding matrix will be shared with the softmax projection
        # Therefore we declare it here
        if tie_all_embeds:
            E_tgt = E_src
        elif tie_decoder_embeds:
            init = UniformInit(1 / np.sqrt(embed_dim))
            E_tgt = self.pc.add_parameters((self.V_tgt, embed_dim), init=init)
        else:
            E_tgt = None
        self.tgt_embed = Embeddings(self.pc, dic_tgt, embed_dim, params=E_tgt)
        # Start of sentence embedding
        self.sos_p = self.pc.add_lookup_parameters((1, embed_dim, 1))
        # Transformer
        self.dec = StackedCondTransformers(
            self.pc,
            n_layers,
            embed_dim,
            hidden_dim,
            embed_dim,
            n_heads,
            dropout=dropout
        )
        # Projection to logits
        self.project = Sequential(
            # First project to embedding dim
            Affine(self.pc, embed_dim, embed_dim),
            # Then logit layer with weights tied to the word embeddings
            Affine(self.pc, embed_dim, self.V_tgt, dropout=dropout, W_p=E_tgt)
        )

    @staticmethod
    def add_args(parser):
        bilstm_group = parser.add_argument_group("BiLSTM")
        bilstm_group.add_argument("--n-layers", type=int, default=4)
        bilstm_group.add_argument("--embed-dim", type=int, default=256)
        bilstm_group.add_argument("--hidden-dim", type=int, default=1024)
        bilstm_group.add_argument("--n-heads", type=int, default=4)
        bilstm_group.add_argument("--dropout", type=float, default=0.2)
        bilstm_group.add_argument("--tie-decoder-embeds", action="store_true")
        bilstm_group.add_argument("--tie-all-embeds", action="store_true")

    @staticmethod
    def from_args(args, dic_src, dic_tgt):
        return Transformer(
            dic_src,
            dic_tgt,
            args.n_layers,
            args.embed_dim,
            args.hidden_dim,
            args.n_heads,
            args.dropout,
            args.tie_decoder_embeds,
            args.tie_all_embeds,
        )

    def init(self, test=False, update=True):
        self.src_embed.init(test=test, update=update)
        self.enc.init(test=test, update=update)
        self.tgt_embed.init(test=test, update=update)
        self.dec.init(test=test, update=update)
        self.project.init(test=test, update=update)

    def encode(self, src):
        # Embed input words
        src_embs = self.src_embed(src.sequences, length_dim=1)
        src_embs = src_embs * np.sqrt(self.embed_dim)
        # Add position encodings
        src_embs += dy.inputTensor(self.pos_embeds[:, :src.max_length])
        # Encode
        X = self.enc(src_embs, lengths=src.lengths)
        #  Return list of encodings for each layer
        return X

    def embed_word(self, word, tgt=False):
        if tgt:
            return self.tgt_embed(word) * np.sqrt(self.embed_dim)
        else:
            return self.src_embed(word) * np.sqrt(self.embed_dim)

    @property
    def sos(self):
        return self.sos_p[0] * np.sqrt(self.embed_dim)

    @property
    def initial_decoder_state(self):
        return None

    def __call__(self, src, tgt):
        # Encode
        # ------
        # Each element of X has shape ``dh x l``
        X = self.encode(src)
        # Decode
        # ------
        L = tgt.max_length
        # Mask for attention
        attn_mask = src.get_mask(base_val=0, mask_val=-np.inf)
        # Embed all words (except EOS)
        tgt_embs = self.tgt_embed(tgt.sequences[:-1], length_dim=1)
        # Add SOS embedding
        sos_embed = self.sos_p.batch([0] * tgt.batch_size)
        tgt_embs = dy.concatenate([sos_embed, tgt_embs], d=1)
        # Scale embeddings
        tgt_embs = tgt_embs * np.sqrt(self.embed_dim)
        # Add positional encoding (tgt_embs has shape ``dh x L``)
        tgt_embs += dy.inputTensor(self.pos_embeds[:, :L])
        # Decode (h_dec has shape ``dh x L``)
        h_dec = self.dec(tgt_embs, X, mask_c=attn_mask, triu=True)
        # Logits (shape |V| x L)
        logits = self.project(h_dec)
        # Return list of logits (one per position)
        return [dy.pick(logits, index=pos, dim=1) for pos in range(L)]

    def decode_step(self, X, wemb, state, attn_mask=None):
        # Run one step of the decoder
        new_state, h, _, attn_weights = self.dec.step(
            state,
            wemb,
            X,
            mask_c=attn_mask,
            return_att=True
        )
        # Get log_probs
        log_p = dy.log_softmax(self.project(h)).npvalue()
        # Alignments from attention (average weights from each head)
        align = dy.average(attn_weights).npvalue()[:, -1].argmax()
        # Return
        return new_state, log_p, align

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
