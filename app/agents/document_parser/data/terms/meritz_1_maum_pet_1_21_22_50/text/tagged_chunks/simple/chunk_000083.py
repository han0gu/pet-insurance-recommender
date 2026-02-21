from langchain_core.documents import Document

chunk = Document(
    page_content=('2. 제17조(알릴 의무 위반의 효과)를 준용하여 회사가 보장을 하지 않을 수 있는 경우\n'
 '3. 진단계약에서 보험금 지급사유가 발생할 때까지 진단을 받지 않은 경우. 다만, 진단\n'
 '계약에서 진단을 받지 않은 경우라도 상해로 보험금 지급사유가 발생하는 경우에는\n'
 '보장을 해드립니다.④ 계약이 갱신되는 경우에는 제1항 내지 제3항에 의한 보장은 기존 계약에 의한 보장이 종\n'
 '료하는 때부터 적용합니다.제26조(제2회 이후 보험료의 납입)계약자는 제2회 이후의 보험료를 납입기일까지 납입하여야 하며, 회사는 '
 '계약자가 보험료'),
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
 'indexing': {'chunk_id': 'chunk_000083',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
