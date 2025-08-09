from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List


class PDFRetrievalChain:
    def __init__(self):
        self.top_k = 3

        self.store = "cache"

        self.docs = None
        self.vectorstore = None

        self.retriever = None
        self.chain = None

    def load_documents(self, source_uris: List[str]):
        docs = []
        for source_uri in source_uris:
            loader = PDFPlumberLoader(source_uri)
            docs.extend(loader.load())
        self.docs = docs
        return docs

    def split_documents(self, docs, text_splitter):
        """text splitter를 사용하여 문서를 분할합니다."""
        return text_splitter.split_documents(docs)

    def create_text_splitter(self):
        return RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    def create_embedding(self):
        # embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        embedder = "jhgan/ko-sroberta-nli"
        embedding = HuggingFaceEmbeddings(model_name=embedder)

        store_dir = self.store+"/"+embedder
        store = LocalFileStore(store_dir)

        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embedding,
            document_embedding_cache=store,
            namespace=embedding.model_name,  # 기본 임베딩과 저장소를 사용하여 캐시 지원 임베딩을 생성
        )
        return cached_embedder

    def create_vectorstore(self, split_docs):
        return FAISS.from_documents(
            documents=split_docs, embedding=self.create_embedding()
        )

    def create_retriever(self, vectorstore):
        # MMR을 사용하여 검색을 수행하는 retriever를 생성합니다.
        dense_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.top_k}
        )
        return dense_retriever

    def build_retriever(self):
        docs = self.docs
        text_splitter = self.create_text_splitter()
        split_docs = self.split_documents(docs, text_splitter)

        self.vectorstore = self.create_vectorstore(split_docs)
        self.retriever = self.create_retriever(self.vectorstore)

    def build_chain(self):
        if self.retriever is None:
            self.build_retriever()


