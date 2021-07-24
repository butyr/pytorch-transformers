"""Test GPT-2 model."""

import unittest
from src.transformer.modeling_gpt2 import *

d_key = 4
nheads = 2
model_dim = d_key*nheads
batch_size = 14
sent_len = 16
hidden_dim = 16
vocab_size = 14

max_len = sent_len * 100
depth = 2
dropout_p = 0.1

torch.manual_seed(1234)


class TestTensorShapes(unittest.TestCase):

    def test_attention(self):
        mhatt = MultiHeadAttention(nheads*d_key, nheads, mask='triu')

        A = torch.ones((batch_size, sent_len, nheads, d_key))
        B = torch.ones((batch_size, int(sent_len/2), nheads, d_key))

        ret = mhatt.attention(A, B, B)

        self.assertEqual(
            (batch_size, nheads, sent_len, int(sent_len/2)), mhatt.att.shape
        )
        self.assertEqual(
            (batch_size, sent_len, nheads, d_key), ret.shape
        )

    def test_multi_head_attention(self):
        mhatt = MultiHeadAttention(model_dim, nheads, mask='triu')

        A = torch.ones((batch_size, sent_len, model_dim))
        B = torch.ones((batch_size, sent_len // 2, model_dim))

        ret = mhatt(A, B, B)

        self.assertEqual((batch_size, sent_len, model_dim), ret.shape)

    def test_transformer(self):
        model = GPT2(
            vocab_size,
            model_dim,
            hidden_dim,
            nheads,
            depth,
            dropout_p,
            max_len,
        )

        x = torch.ones(
            (batch_size, sent_len), dtype=torch.long
        )

        ret = model(x)

        self.assertEqual(
            (batch_size, sent_len, vocab_size), ret.shape
        )

    def test_pe(self):
        A = torch.ones((batch_size, sent_len, model_dim))
        pe = PositionalEncoder(model_dim, max_len)

        ret = pe(A)

        self.assertEqual((batch_size, sent_len, model_dim), ret.shape)

    def test_decoder_layer(self):
        A = torch.ones((batch_size, sent_len, model_dim))
        decoder = DecoderLayer(model_dim, hidden_dim, nheads)

        dec = decoder(A)

        self.assertEqual((batch_size, sent_len, model_dim), dec.shape)


class TestEmbedding(unittest.TestCase):

    def setUp(self):
        self.encoder_shape = (batch_size, sent_len, model_dim)
        self.decoder_shape = (batch_size, sent_len, vocab_size)

        self.model = Embedding(vocab_size, model_dim)

        self.input_a = torch.ones((batch_size, sent_len), dtype=torch.long)
        self.input_b = torch.ones((batch_size, sent_len, model_dim))

        self.target_a = torch.ones(
            (batch_size, sent_len, model_dim)
        )
        self.target_b = torch.ones(
            (batch_size, sent_len, vocab_size)
        )

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.9
        )

        self.weights_in = copy.deepcopy(self.model.encoder.weight)
        self.weights_out = copy.deepcopy(self.model.decoder.weight)

    def test_decoder_tie_weights(self):
        self.optimizer.zero_grad()
        output_b = self.model(self.input_b, inverse=True)
        loss = torch.sum(output_b - self.target_b)

        loss.backward()
        self.optimizer.step()

        self.assertNotEqual(
            torch.sum(self.weights_in),
            torch.sum(self.model.encoder.weight)
        )
        self.assertNotEqual(
            torch.sum(self.weights_out),
            torch.sum(self.model.decoder.weight)
        )
        self.assertEqual(
            torch.sum(self.model.encoder.weight),
            torch.sum(self.model.decoder.weight)
        )
        self.assertEqual(self.decoder_shape, output_b.shape)


class TestSanityChecks(unittest.TestCase):

    def test_init_loss(self):
        """Compares loss@init with theoretical loss."""

        model = GPT2(
            vocab_size,
            model_dim,
            hidden_dim,
            nheads,
            depth,
            dropout_p,
            max_len,
        )

        inputs = torch.arange(
            batch_size, dtype=torch.long
        ).repeat(sent_len).reshape(batch_size, sent_len)

        targets = torch.arange(
            batch_size, dtype=torch.long
        ).repeat(sent_len).reshape(batch_size, sent_len)

        outputs = F.softmax(model(inputs), dim=-1)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(
            outputs.reshape(-1, vocab_size),
            targets.reshape(-1)
        )

        self.assertAlmostEqual(
            -np.log(1./vocab_size), loss.detach().numpy(), delta=0.1
        )


class TestGradientFlows(unittest.TestCase):

    def test_one_step_grad(self):
        """Tests whether all parameters are updated."""

        model = GPT2(
            vocab_size,
            model_dim,
            hidden_dim,
            nheads,
            depth,
            dropout_p,
            max_len,
        )

        inputs = torch.ones(
            (batch_size, sent_len), dtype=torch.long
        )

        optimizer = torch.optim.SGD(
            model.parameters(), lr=100.0, momentum=0.9
        )
        model_t0 = copy.deepcopy(model)

        optimizer.zero_grad()
        outputs = model(inputs)
        targets = torch.argmax(
            (outputs*(-1)).reshape(-1, vocab_size), dim=-1
        )

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(
            outputs.reshape(-1, vocab_size),
            targets
        )

        loss.backward()
        optimizer.step()

        for p0, p in zip(
                model_t0.named_parameters(),
                model.named_parameters()
        ):
            self.assertNotEqual(
                0.0,
                torch.sum(torch.square(p[1]-p0[1]))
            )

    def test_batch_dim(self):
        """Tests consistency of batch dimension."""

        model = GPT2(
            vocab_size,
            model_dim,
            hidden_dim,
            nheads,
            depth,
            dropout_p,
            max_len,
        )

        inputs = torch.ones(
            (batch_size, sent_len), dtype=torch.long
        )

        outputs = model(inputs)

        for i in range(batch_size):
            loss = outputs[i, :, :].sum()
            grad = torch.autograd.grad(
                loss, model.inputs_embedding, retain_graph=True
            )[0]

            self.assertNotEqual(0.0, loss)

            self.assertEqual(
                0.0,
                grad[:i, :, :].sum()
            )
            self.assertEqual(
                0.0,
                grad[i+1:, :, :].sum()
            )
            self.assertNotEqual(
                0.0,
                grad[i, :, :].sum()
            )

    def test_mask(self):
        """Tests masking of decoder inputs."""

        model = GPT2(
            vocab_size,
            model_dim,
            hidden_dim,
            nheads,
            depth,
            dropout_p,
            max_len,
        )

        inputs = torch.ones(
            (batch_size, sent_len), dtype=torch.long
        )

        outputs = model(inputs)

        for i in range(sent_len):
            loss = outputs[:, i, :].sum()
            grad = torch.autograd.grad(
                loss, model.inputs_embedding, retain_graph=True
            )[0]

            self.assertEqual(
                0.0,
                grad[:, i+1:, :].sum()
            )
            self.assertNotEqual(
                0.0,
                grad[:, :i+1, :].sum()
            )
