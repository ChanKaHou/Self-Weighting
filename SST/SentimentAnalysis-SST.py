import os
import torch
import torchtext
import pytorch_lightning

class CARUCell(torch.nn.Module): #Content-Adaptive Recurrent Unit
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.LW = torch.nn.Linear(out_feature, out_feature)
        self.LL = torch.nn.Linear(in_feature, out_feature)
        self.Weight = torch.nn.Linear(out_feature, out_feature)
        self.Linear = torch.nn.Linear(in_feature, out_feature)

    def forward(self, word, hidden):
        feature = self.Linear(word)
        if hidden is None:
            return torch.tanh(feature)
        n = torch.tanh(self.Weight(hidden) + feature)
        l = torch.sigmoid(feature)*torch.sigmoid(self.LW(hidden) + self.LL(word))
        return torch.lerp(hidden, n, l)

class CARU(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.CARUCell = CARUCell(in_feature, out_feature)

    def forward(self, sequence):
        hidden = None
        for feature in sequence:
            hidden = self.CARUCell(feature, hidden)
        return hidden

class Self_Weighting(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.Weighted = torch.nn.Sequential(
            torch.nn.Linear(in_feature, out_feature),
            torch.nn.Sigmoid(),
            )
        self.Feature = torch.nn.Sequential(
            torch.nn.Linear(in_feature, out_feature),
            torch.nn.Tanh(),
            )

    def forward(self, input):
        return self.Weighted(input)*self.Feature(input)

class Model(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        return

    def setup(self, stage):
        if (stage != 'fit'):
            return

        #python -m spacy download en_core_web_sm
        self.Text = torchtext.data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', lower=True)
        self.Label = torchtext.data.LabelField(is_target=True)

        self.trainSet, self.valSet, self.testSet = torchtext.datasets.SST.splits(self.Text, self.Label, root='../data')
        print('Dataset Size:', len(self.trainSet), len(self.valSet), len(self.testSet)) #8544 1101 2210

        self.Text.build_vocab(self.trainSet.text, vectors_cache='../vector_cache', min_freq=4, vectors='glove.6B.100d')
        self.Label.build_vocab(self.trainSet.label)
        print('Text Vocabulary Size:', len(self.Text.vocab))
        print('Label Vocabulary Size:', len(self.Label.vocab))

        self.Embedding = torch.nn.Sequential(
            torch.nn.Embedding.from_pretrained(self.Text.vocab.vectors),
            torch.nn.Dropout(),
            )
        self.Self_Weighting = Self_Weighting(100, 256)
        self.CARU = CARU(256, 512)
        self.Classifier = torch.nn.Sequential(
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, len(self.Label.vocab)),
            )

    def forward(self, sentence):
        embedded = self.Embedding(sentence) #[S, batch_size, E]
        feature = self.Self_Weighting(embedded)
        hidden = self.CARU(feature)
        return self.Classifier(hidden)
        
    def train_dataloader(self):
        return torchtext.data.BucketIterator(self.trainSet, batch_size=100, shuffle=True)

    def val_dataloader(self):
        return torchtext.data.BucketIterator(self.valSet, batch_size=100)

    def test_dataloader(self):
        return torchtext.data.BucketIterator(self.testSet, batch_size=100)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'reduce_on_plateau': True, 'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        pred_label = self(batch.text)
        return torch.nn.functional.cross_entropy(pred_label, batch.label)

    def validation_step(self, batch, batch_idx):
        pred_label = self(batch.text)
        loss = torch.nn.functional.cross_entropy(pred_label, batch.label)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        acc = torch.mean(pred_label.argmax(-1) == batch.label, dtype=torch.float)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        pred_label = self(batch.text)
        acc = torch.mean(pred_label.argmax(-1) == batch.label, dtype=torch.float)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_epoch_end(self):
        print()

model = Model()
trainer = pytorch_lightning.Trainer(gpus=1, max_epochs=150, weights_summary='full', resume_from_checkpoint=None)
trainer.fit(model)
trainer.test(model)
