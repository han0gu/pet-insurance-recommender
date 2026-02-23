from langchain_core.documents import Document

chunk = Document(
    page_content=('다만, 의무보험의 해지를 요구하려는 경우에는 동종의 다른 의무보험에 가입되어 있어야 합니다.【위법 계약】 금융상품판매업자 등이 '
 '「금융소비자보호에 관한 법률」 제47조에서 정한 적합성원칙, 적\n'
 '정성원칙, 설명의무, 불공정영업행위 금지 또는 부당권유행위 금지를 위반한 계약을 말합니다.- ② 회사는 해지요구를 받은 날부터 10일 '
 '이내에 수락여부를 계약자에 통지하여야 하며, 거절할 때에\n'
 '- 는 거절 사유를 함께 통지하여야 합니다.\n'
 '- ③ 계약자는 회사가 정당한 사유 없이 제1항의 요구를 따르지 않는 경우 해당 계약을 해지할 수 있습'),
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
 'indexing': {'chunk_id': 'chunk_000078',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
