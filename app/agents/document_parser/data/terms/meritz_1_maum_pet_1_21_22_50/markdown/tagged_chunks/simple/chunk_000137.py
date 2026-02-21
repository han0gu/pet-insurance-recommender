from langchain_core.documents import Document

chunk = Document(
    page_content=('- 서 공제계약을 포함합니다.\n'
 '# 【공제계약】공제계약이란 동일한 직업 또는 사업에 종사하는 다수의 주체가 상호구제를 위하여\n'
 '보험료에 상당하는 금전을 납입하고 그 가입자에게 소정의 사고가 발생한 경우 공동\n'
 '의 구제를 하는 계약으로, 가입자의 범위가 구성원 또는 특정 지역으로 한정됨.③ 피보험자가 의무보험에 가입하여야 함에도 불구하고 가입하지 '
 '않은 경우에는 그가 가입\n'
 '했더라면 의무보험에서 보상했을 금액을 제1항의 의무보험에서 보상하는 금액으로 봅'),
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
 'indexing': {'chunk_id': 'chunk_000137',
              'chunk_char_len': 249,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
