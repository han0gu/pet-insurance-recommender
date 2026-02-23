from langchain_core.documents import Document

chunk = Document(
    page_content=('을 실행하는 것을 말합니다.【국세 및 지방세 체납처분 절차】국세 또는 지방세를 체납할 경우 국세 기본법 및 지방세법에 의하여 체납된 '
 '세금에\n'
 '대하여 가산금징수, 독촉장 발부 및 재산 압류 등의 집행을 하는 것을 말합니다.- 17 -제6관 계약의 해지 및 보험료의 환급 '
 '등제30조(계약의 해지)계약자는 계약이 소멸하기 전에는 언제든지 계약을 해지할 수 있으며, 이 경우 회사가 환\n'
 '급하여야 할 보험료가 있을 경우에는 제33조(보험료의 환급)에 따른 보험료를 계약자에게'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000096',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
