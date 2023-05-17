import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time
import numpy as np
from ktrain.text.qa import SimpleQA
INDEXDIR = '/tmp/newqa'



############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    pass

############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:

        try:
            qa = SimpleQA(INDEXDIR, framework ="pt")

        except Exception:
            docs = np.load('merged_dataset.npy')

            SimpleQA.initialize_index(INDEXDIR)
            SimpleQA.index_from_list(docs, INDEXDIR, commit_every=len(docs),
                         multisegment=True, procs=4, # these args speed up indexing
                         breakup_docs=True         # this slows indexing but speeds up answer retrieval
                            )
            qa = SimpleQA(INDEXDIR, framework ="pt")

        
        answers = qa.ask(text)
        for answer in answers[:3]:
            output.append(answer['full_answer'])

    return SimpleText(dict(text=output))
