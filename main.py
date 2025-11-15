from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer(
    "./KaLM-Embedding-Gemma3-12B-2511",
    trust_remote_code=True,
    local_files_only=True,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",  # Optional
    },
)
model.max_seq_length = 512

sentences = ["This is an example sentence", "Each sentence is converted"]
prompt = "Instruct: Classifying the category of french news.\nQuery:"
embeddings = model.encode(
    sentences,
    prompt=prompt,
    normalize_embeddings=True,
    batch_size=256,
    show_progress_bar=True,
)
print(embeddings)
'''
[[-0.01867676  0.02319336  0.00280762 ... -0.02075195  0.00196838
  -0.0703125 ]
 [-0.0067749   0.03491211  0.01434326 ... -0.0043335   0.00509644
  -0.04174805]]
'''
