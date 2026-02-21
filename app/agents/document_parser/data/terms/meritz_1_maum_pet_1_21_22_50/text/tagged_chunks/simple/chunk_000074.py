from langchain_core.documents import Document

chunk = Document(
    page_content=('회사의 고의 또는 과실로 계약이 무효로 된 경우와 회사가 승낙 전에 무효임을 알았거나\n'
 '알 수 있었음에도 보험료를 반환하지 않은 경우에는 보험료를 납입한 날의 다음날부터 반\n'
 '환일까지의 기간에 대하여 회사는 이 계약의 ‘보험개발원이 공시하는 보험계약대출이율’을\n'
 '연단위 복리로 계산한 금액을 더하여 돌려 드립니다.1. 계약을 체결할 때 계약에서 정한 피보험자 및 반려동물의 나이에 미달되었거나 '
 '초과되\n'
 '었을 경우. 다만, 회사가 나이의 착오를 발견하였을 때 이미 계약나이에 도달한 경우에'),
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
 'indexing': {'chunk_id': 'chunk_000074',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
