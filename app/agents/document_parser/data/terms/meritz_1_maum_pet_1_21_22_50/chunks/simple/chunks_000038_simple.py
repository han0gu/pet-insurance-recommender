from langchain_core.documents import Document

chunk = Document(
    page_content=('다. 이 경우 상시고용된 수의사의 범위, 신고방법, 처방전 발급 및 보존 방법, 진료부 작성 및 보고, 교육, 준수사항 등 그 밖에 '
 '필요한 사항은 농림축산식품 부령으로 정한다.\n'
 '제9조(보험금의 지급절차)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 7},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000038',
              'chunk_char_len': 113,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
