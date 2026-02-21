from langchain_core.documents import Document

chunk = Document(
    page_content=('위험의 동질성이 유지되어야 합니다.제4조(보험의 목적의 증가 감소 또는 교체)① 계약을 맺은 후 보험의 목적을 증가, 감소 또는 교체코자 '
 '하는 경우에는 계약자 또는 피\n'
 '보험자는 지체없이 서면으로 그 사실을 회사에 알리고 회사의 승인을 받아야 합니다.\n'
 '② 이 계약기간 중 보험의 목적 감소의 경우는 당해 보험의 목적의 계약은 해지된 것으로\n'
 '하며 새로이 증가 또는 교체되는 보험의 목적의 보험기간은 이 계약의 남은 보험기간\n'
 '으로 하고, 이로 인하여 발생되는 추가 또는 환급보험료는 일단위로 계산하여 받거나\n'
 '돌려 드립니다.'),
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
 'indexing': {'chunk_id': 'chunk_000175',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
