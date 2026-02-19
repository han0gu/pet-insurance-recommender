from langchain_core.documents import Document

chunk = Document(
    page_content=('다)에는 회사는 지급한 보험금의 한도내에서 아래의 권리를 가집니다. 다만, 회사가 보상한 금액이 피보험자가 입은 손 해의 일부인 경우에는 '
 '피보험자의 권리를 침해하지 않는 범 위내에서 그 권리를 가집니다.\n'
 '① 피보험자가 제3자로부터 손해배상을 받을 수 있는 경 우에는 그 손해배상청구권 ② 피보험자가 손해배상을 함으로써 대위 취득하는 것이 '
 '있을 경우에는 그 대위권'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 181},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000607',
              'chunk_char_len': 202,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
