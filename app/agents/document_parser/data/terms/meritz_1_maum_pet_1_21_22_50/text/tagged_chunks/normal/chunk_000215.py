from langchain_core.documents import Document

chunk = Document(
    page_content=('| 보장관련 보험금 | 지급기일의 61일이후부터 90일이내 기간 | 보험계약대출이율+ 가산이율(6.0%) |\n'
 '| 보장관련 보험금 | 지급기일의 91일이후 기간 | 보험계약대출이율+ 가산이율(8.0%) |\n'
 '주) 1. 지급이자의 계산은 연단위 복리로 계산합니다.\n'
 '2. 계약자 등의 책임 있는 사유로 보험금 지급이 지연된 때에는 그 해당 기간에 대한\n'
 '이자는 지급되지 않을 수 있습니다. 다만, 회사는 계약자 등이 분쟁조정을 신청했\n'
 '다는 사유만으로 이자지급을 거절하지 않습니다.'),
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
 'indexing': {'chunk_id': 'chunk_000215',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
