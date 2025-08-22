from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFRetrievalChain:
    def __init__(self, docs: list[str] = []):
        self.top_k = 3

        self.store = "cache"
        self.docs = docs
        self.chain = None

    def load_documents(self, source_uris: list[str]) -> list[Document]:
        docs = []
        for source_uri in source_uris:
            loader = PDFPlumberLoader(source_uri)
            docs.extend(loader.load())
        return docs

    def split_documents(
        self, docs, text_splitter: RecursiveCharacterTextSplitter
    ) -> list[Document]:
        """text splitter를 사용하여 문서를 분할합니다."""
        return text_splitter.split_documents(docs)

    def create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    def create_embedding(self):
        # embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        embedder = "jhgan/ko-sroberta-nli"
        embedding = HuggingFaceEmbeddings(model_name=embedder)

        store_dir = self.store + "/" + embedder
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
        docs = self.load_documents(self.docs)
        text_splitter = self.create_text_splitter()
        split_docs = self.split_documents(docs, text_splitter)

        vectorstore = self.create_vectorstore(split_docs)
        retriever = self.create_retriever(vectorstore)
        return retriever

    def build_chain(self):
        self.chain = self.build_retriever()
