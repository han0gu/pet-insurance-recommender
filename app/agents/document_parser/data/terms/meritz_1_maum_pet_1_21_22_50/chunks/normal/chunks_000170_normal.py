from langchain_core.documents import Document

chunk = Document(
    page_content=('제15조(계약 후 알릴 의무)\n'
 '① 계약을 맺은 후 보험의 목적에 아래와 같은 사실이 생긴 경우에는 계약자나 피보험자는 지체없이 서면으로 회사에 알리고 보험증권에 확인을 '
 '받아야 합니다.\n'
 '1. 청약서의 기재사항을 변경하고자 할 때 또는 변경이 생겼음을 알았을 때 2. 이 계약에서 보장하는 위험과 동일한 위험을 보장하는 '
 '계약을 다른 보험자와 체결하'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 27},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000170',
              'chunk_char_len': 192,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
