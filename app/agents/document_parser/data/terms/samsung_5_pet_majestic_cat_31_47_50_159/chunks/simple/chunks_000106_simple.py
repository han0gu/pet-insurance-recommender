from langchain_core.documents import Document

chunk = Document(
    page_content=('3. 계약을 체결할 때 계약에서 정한 피보험자의 나이에 미달되었거나 초과되었을 경 우. 다만, 회사가 나이의 착오를 발견하였을 때 이미 '
 '계약나이에 도달한 경우에는 유효한 계약으로 보나, 제2호의 만 15세 미만자에 관한 예외가 인정되는 것은 아 닙니다.\n'
 '제24조 (계약내용의 변경 등)\n'
 '① 계약자는 회사의 승낙을 얻어 다음의 사항을 변경할 수 있습니다. 이 경우 승낙을 서'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 40},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000106',
              'chunk_char_len': 208,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
