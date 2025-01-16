# Importar bibliotecas necessárias
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import nltk
from bertopic.representation import KeyBERTInspired

# Baixar stopwords do NLTK (se necessário)
nltk.download('stopwords')

# Carregar o arquivo CSV (substitua pelo caminho correto do seu arquivo)
df = pd.read_csv('data/val_texts.csv', header=None)  # Sem cabeçalho (substitua pelo caminho correto do seu arquivo)
documents = df[0].tolist()  # Acessar a primeira coluna (coluna A)

# Remover textos vazios ou nulos
documents = [doc for doc in documents if isinstance(doc, str) and len(doc.strip()) > 0]
print(f"Número de documentos após limpeza: {len(documents)}")

# Carregar o modelo de embeddings multilíngue compatível com português
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Fine-tune your topic representations
representation_model = KeyBERTInspired()
topic_model = BERTopic(representation_model=representation_model)

# Criar e treinar o modelo BERTopic
topic_model = BERTopic(embedding_model=embedding_model, language="multilingual", representation_model=representation_model)
topics, probs = topic_model.fit_transform(documents)

# Exibir as informações dos tópicos gerados
topic_info = topic_model.get_topic_info()
print("\nInformações dos Tópicos Gerados:")
print(topic_info)

# Salvar as informações dos tópicos em um arquivo CSV
output_file = "data/output.csv"  # Caminho local onde o arquivo será salvo
topic_info.to_csv(output_file, index=False)
print(f"Informações dos tópicos salvas em {output_file}")

# Se necessário, visualize os tópicos gerados (caso queira ver interativamente)
topic_model.visualize_topics()
