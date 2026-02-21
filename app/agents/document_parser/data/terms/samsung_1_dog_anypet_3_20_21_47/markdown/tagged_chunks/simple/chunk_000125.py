from langchain_core.documents import Document

chunk = Document(
    page_content=('피보험자의 권리를 침해하지 않는 범위내에서 그 권리를 가집니다.- 1. 피보험자가 제3자로부터 손해배상을 받을 수 있는 경우에는 그 '
 '손해배상청구권\n'
 '- 2. 피보험자가 손해배상을 함으로써 대위 취득하는 것이 있을 경우에는 그 대위권\n'
 '- ② 계약자 또는 피보험자는 제1항에 의하여 회사가 취득한 권리를 행사하거나 지키는 것에 관하여 조\n'
 '- 치를 하여야 하며, 또한 회사가 요구하는 증거 및 서류를 제출하여야 합니다.\n'
 '- ③ 회사는 제1항, 제2항에도 불구하고 타인을 위한 보험계약의 경우에는 계약자에 대한 대위권을 포\n'
 '- 기합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000125',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
