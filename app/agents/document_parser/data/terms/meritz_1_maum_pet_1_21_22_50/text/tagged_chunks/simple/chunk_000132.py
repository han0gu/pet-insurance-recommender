from langchain_core.documents import Document

chunk = Document(
    page_content=('초과할 때에 한하여 그 초과액만을 보상합니다. 다만, 의무보험이 다수인 경우에는 제\n'
 '10조(보험금의 분담)를 따릅니다.\n'
 '② 제1항의 의무보험은 피보험자가 법률에 의하여 의무적으로 가입하여야 하는 보험으로\n'
 '서 공제계약을 포함합니다.【공제계약】공제계약이란 동일한 직업 또는 사업에 종사하는 다수의 주체가 상호구제를 위하여\n'
 '보험료에 상당하는 금전을 납입하고 그 가입자에게 소정의 사고가 발생한 경우 공동'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000132',
              'chunk_char_len': 222,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
