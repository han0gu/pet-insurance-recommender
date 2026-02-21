from langchain_core.documents import Document

chunk = Document(
    page_content=('금(이자를 포함합니다)의 회수청구권을 회사에 양도하여야\n'
 '합니다.# 제11조(대위권)\uf000 회사가 보험금을 지급한 때(현물보상한 경우를 포함합니180다)에는 회사는 지급한 보험금의 '
 '한도내에서 아래의 권리를\n'
 '가집니다. 다만, 회사가 보상한 금액이 피보험자가 입은 손\n'
 '해의 일부인 경우에는 피보험자의 권리를 침해하지 않는 범\n'
 '위내에서 그 권리를 가집니다.- ① 피보험자가 제3자로부터 손해배상을 받을 수 있는 경\n'
 '- 우에는 그 손해배상청구권\n'
 '- ② 피보험자가 손해배상을 함으로써 대위 취득하는 것이\n'
 '- 있을 경우에는 그 대위권'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000503',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
