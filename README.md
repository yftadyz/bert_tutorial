# bert_tutorial
Tutorial of how to use bert with tf 2.x and tf.hub.

##  Background
BERT official repository(https://github.com/google-research/bert) provides an excellent tutorial example "predicting_movie_reviews_with_bert_on_tf_hub.ipynb"(I just name it 'official tutorial' in the following). However, this example is based on tensorflow 1.x and could not be used under tensorflow 2.x. I searched the whole Internet and ended up with finding no detailed tutorial about how to use bert under tensorflow 2.x. So I referred all sources I could find and finally worked out this simple tutorial. Just hope this repos could help those bert starters quickly master how to use BERT with tf 2.x and save their time (I spent so much time walking around to find a way out, cry...).

## Requirements
* python==3.x
* tensorflow>=2.0

## Usage
I use the same data preprocess as the 'official tutorial'. Tokenization and input build functions are almost the same as official repos except some small changes to make them OK under tensorflow2.x. The main difference exists in model structure for downstream task.

```
class BERT(tf.keras.Model):

  def __init__(self,para=None):

    super().__init__()

    drop_rate=para['drop_rate']

    self.bert_layer = hub.KerasLayer("bert_en_uncased_L-12_H-768_A-12_2",
                            trainable=True,name='bert_layer')

    self.dp_layer=tf.keras.layers.Dropout(drop_rate)

    self.task_output=tf.keras.layers.Dense(1,activation=tf.nn.sigmoid,name='task_specific_output_layer')

  def call(self,ft,training=False):

    pooled_output, sequence_output = self.bert_layer([ft['input_ids'], ft['input_mask'], ft['segment_ids']])

    pooled_output=self.dp_layer(pooled_output,training=training)

    return self.task_output(pooled_output)
```

I use hub.KerasLayer to load pretrained BERT model. The hub usage in 'official tutorial' doesn't work well under tensorflow 2.x. Note that I use an offline BERT pretrained model. I download the pretrained model to my laptop from hub official website. Online usage is also fine:
```
self.bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
                            trainable=True,name='bert_layer')
```

Moreover, tf.estimator is not used in my tutorial. I implement training process in my own style. Tf.estimator style often makes me feel coding in a maze.

Even though this tutorial plays a text classification task, it's very easy to expand the stucture to all other different NLP tasks. For sentence level task, like sentiment analysis, or natural language inference, pooled_output is needed. For token level task, like question answering, use sequence_output.

## Suggestion
I advise playing this tutorial on google colab, it would be much faster with a GPU. If you got an OOM, set MAX_SEQ_LENGTH to be a lower number, like 64 instead of 128.

Finally I give thanks to official tutorial. Without it, I hardly know where to put my first step. Good luck.
