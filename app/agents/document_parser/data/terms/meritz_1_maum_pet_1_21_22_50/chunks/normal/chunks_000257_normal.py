from langchain_core.documents import Document

chunk = Document(
    page_content=('주) 1. 지급이자의 계산은 연단위 복리로 계산합니다. 2. 계약자 등의 책임 있는 사유로 보험금 지급이 지연된 때에는 그 해당 기간에 '
 '대한 이자는 지급되지 않을 수 있습니다. 다만, 회사는 계약자 등이 분쟁조정을 신청했 다는 사유만으로 이자지급을 거절하지 않습니다. 3. '
 '가산이율 적용시 제9조(보험금의 지급절차) 제2항 각 호의 어느 하나에 해당되는 사 유로 지연된 경우에는 해당기간에 대하여 가산이율을 '
 '적용하지 않습니다. 4'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 48},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000257',
              'chunk_char_len': 240,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
