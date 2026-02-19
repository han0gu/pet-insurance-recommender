from langchain_core.documents import Document

chunk = Document(
    page_content=('. 3. 회사는 보험기간 만료 후 보험의 목적의 정보의 변경에 따라 산출된 확정보험료와 계약 을 체결할 때 산출한 예치보험료를 비교하여 '
 '그 차액을 정산합니다. 4. 제1호에도 불구하고 보험의 목적의 정보의 변경에 관한 서류 제출 시기는 계약자와 별 도로 협의하여 변경할 수 '
 '있습니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 39},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000218',
              'chunk_char_len': 158,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
