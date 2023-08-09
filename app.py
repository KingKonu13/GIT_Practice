from fastai.vision.all import *
import gradio as gr

learn = load_learner('export.pkl')

categories = ('black','grizzly','teddy')
def classify_image(img):
  pred,idx,probs = learn.predict(img)
  return dict(zip(categories, map(float,probs)))

title = "Grizzly, Teddy, or Black Bear Classifier"
description = "A classifier trained on bears from a ddg dataset"
examples = ["teddybear.jpg"]

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs. Label()

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples, title =title, description=description)
intf.launch(inline=False, share=False)

enable_queue=True
