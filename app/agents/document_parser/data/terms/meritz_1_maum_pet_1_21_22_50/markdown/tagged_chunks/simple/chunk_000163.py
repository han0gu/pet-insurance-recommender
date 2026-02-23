from langchain_core.documents import Document

chunk = Document(
    page_content=('- 서 회사에 보험금의 지급을 청구할 수 있습니다.\n'
 '# 【타인을 위한 계약】계약자가 다른 사람의 이익을 위해 자기의 이름으로 체결하는 보험계약을 말합니다.# 제19조(특별약관의 소멸)이 '
 '특별약관의 반려동물의 사망으로 이 특별약관에서 규정하는 보험금 지급사유가 더 이상\n'
 '발생할 수 없는 경우에는 이 특별약관은 그 때부터 효력이 없습니다.- 29 -# 제20조(계약자의 임의해지)계약자는 손해가 발생하기 '
 '전에는 언제든지 계약을 해지할 수 있습니다. 다만, 타인을 위'),
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
 'indexing': {'chunk_id': 'chunk_000163',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
