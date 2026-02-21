from langchain_core.documents import Document

chunk = Document(
    page_content=('- 간(보험의 목적 교체전 계약의 보험기간 만료일)까지 보상하여 드립니다.\n'
 '# 제5조(개별계약으로의 전환)- ① 피보험자가 퇴직 등의 사유로 인하여 피보험단체에서 탈퇴하는 경우 피보험자가 보험\n'
 '- 료의 일부를 부담한 경우에 한하여 탈퇴일로부터 1개월 이내에 계약자 또는 피보험자\n'
 '- 는 회사의 승낙을 얻어 개별계약으로 전환할 수 있으며, 이 경우 피보험자는 개별계약의\n'
 '- 계약자가 됩니다.\n'
 '- ② 제1항에 따라 개별계약으로 전환시에는 전환후 피보험자의 보험기간은 이 계약의 남은'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000182',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
