from langchain_core.documents import Document

chunk = Document(
    page_content=('계약은 해지된 것으로<br>하며 새로이 증가 또는 교체되는 보험의 목적의 보험기간은 이 계약의 남은 보험기간<br>으로 하고, 이로 '
 '인하여 발생되는 추가 또는 환급보험료는 일단위로 계산하여 받거나<br>돌려 드립니다.<br>③ 회사는 제1항 및 제2항을 위반하였을 '
 '경우에 새로이 증가 또는 교체되는 해당 보험의<br>목적에 대하여는 보상하여 드리지 않습니다.<br>④ 제1항에 따라 보험의 목적이 '
 '교체되는 경우에는 보험의 목적 교체전 계약과 동일한 보<br>장조건 및 인수기준에 따라 가입될 수 있으며, 보험의 목적 교체시점부터 잔여'),
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
 'indexing': {'chunk_id': 'chunk_000319',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
