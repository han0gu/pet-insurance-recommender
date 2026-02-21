from langchain_core.documents import Document

chunk = Document(
    page_content=('경우에는 계약자의 서류를 열람할 수 있습니다.\n'
 '3. 회사는 보험기간 만료 후 보험의 목적의 정보의 변경에 따라 산출된 확정보험료와 계약\n'
 '을 체결할 때 산출한 예치보험료를 비교하여 그 차액을 정산합니다.\n'
 '4. 제1호에도 불구하고 보험의 목적의 정보의 변경에 관한 서류 제출 시기는 계약자와 별'),
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
 'indexing': {'chunk_id': 'chunk_000182',
              'chunk_char_len': 162,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
