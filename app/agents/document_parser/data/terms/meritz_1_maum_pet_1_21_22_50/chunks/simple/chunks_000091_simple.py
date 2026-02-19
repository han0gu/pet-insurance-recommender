from langchain_core.documents import Document

chunk = Document(
    page_content=('제24조(계약의 소멸)\n'
 '반려동물의 사망 등으로 인하여 이 약관에서 규정하는 보험금 지급사유가 더 이상 발생할 수 없는 경우에는 이 계약은 그 때부터 효력이 '
 '없습니다.\n'
 '제5관 보험료의 납입\n'
 '제25조(제1회 보험료 및 회사의 보장개시)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 15},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000091',
              'chunk_char_len': 129,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
