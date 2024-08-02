



















import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_embeddings(text, model, tokenizer):
    """ Generate embeddings from the model for given text. """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Use the last layer hidden state
    embeddings = outputs.hidden_states[-1][:, 0, :].detach().numpy()
    return embeddings

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

# Sample data
texts = ["Hello world", "I love machine learning", "The cat sits on the mat"]

# Get embeddings
embeddings = torch.cat([get_embeddings(text, model, tokenizer) for text in texts])

# Reduce dimensions to 3 using PCA
pca = PCA(n_components=3)
reduced_embeddings = pca.fit_transform(embeddings)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2])

for i, txt in enumerate(texts):
    ax.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], reduced_embeddings[i, 2], txt)

ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
plt.title('3D PCA of LLM Embeddings')
plt.show()
#------------






#['PERSON', 'GPE', 'iBookG4', 'PERCENT', 'QUANTITY', 'EVENT', 'LANGUAGE', 'PERCENT', 'WORK_OF_ART', 'INTONE']

