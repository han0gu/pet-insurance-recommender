from langchain_core.documents import Document

chunk = Document(
    page_content=('- 부가되어집니다.\n'
 '- ② 제1조(적용대상)의 보험계약이 해지 또는 기타 사유에 의하여 효력을 가지지 않게 되는\n'
 '- 경우에는 이 특약은 더 이상 효력을 가지지 않습니다.\n'
 '# 제3조(지정대리청구인의 지정)① 보험계약자는 보통약관 또는 특별약관에서 정한 보험금을 직접 청구할 수 없는 특별한\n'
 '사정이 있을 경우를 대비하여 계약을 체결할 때 또는 계약체결 이후 다음 각 호의 1에\n'
 '해당하는 자 중에서 보험금의 대리청구인(이하「지정대리청구인」이라 합니다)을 2인'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000194',
              'chunk_char_len': 253,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
