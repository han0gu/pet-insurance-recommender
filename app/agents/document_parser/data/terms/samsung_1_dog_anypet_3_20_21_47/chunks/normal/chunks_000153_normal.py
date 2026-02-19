from langchain_core.documents import Document

chunk = Document(
    page_content=('제10조(대위권)\n'
 '① 회사가 보험금을 지급한 때(현물보상한 경우를 포함합니다)에는 회사는 지급한 보험금의 한도내에 서 아래의 권리를 가집니다. 다만, '
 '회사가 보상한 금액이 피보험자가 입은 손해의 일부인 경우에는 피보험자의 권리를 침해하지 않는 범위내에서 그 권리를 가집니다.\n'
 '1. 피보험자가 제3자로부터 손해배상을 받을 수 있는 경우에는 그 손해배상청구권 2. 피보험자가 손해배상을 함으로써 대위 취득하는 것이 '
 '있을 경우에는 그 대위권'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 28},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000153',
              'chunk_char_len': 243,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
