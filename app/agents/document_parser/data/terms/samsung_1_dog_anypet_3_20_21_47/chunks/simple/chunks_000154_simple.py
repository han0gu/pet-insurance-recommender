from langchain_core.documents import Document

chunk = Document(
    page_content=('② 계약자 또는 피보험자는 제1항에 의하여 회사가 취득한 권리를 행사하거나 지키는 것에 관하여 조 치를 하여야 하며, 또한 회사가 '
 '요구하는 증거 및 서류를 제출하여야 합니다. ③ 회사는 제1항, 제2항에도 불구하고 타인을 위한 보험계약의 경우에는 계약자에 대한 '
 '대위권을 포 기합니다. ④ 회사는 제1항에 따른 권리가 계약자 또는 피보험자와 생계를 같이 하는 가족에 대한 것인 경우에 는 그 권리를 '
 '취득하지 못합니다. 다만, 손해가 그 가족의 고의로 인하여 발생한 경우에는 그 권리 를 취득합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 28},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000154',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
