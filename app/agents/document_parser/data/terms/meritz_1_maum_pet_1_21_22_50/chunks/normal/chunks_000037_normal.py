from langchain_core.documents import Document

chunk = Document(
    page_content=('. ⑤ 제1항에도 불구하고 농림축산식품부장관에게 신고한 축산농장에 상시고용된 수 의사와 「동물원 및 수족관의 관리에 관한 법률」 제8조에 '
 '따라 허가받은 동물 원 또는 수족관에 상시고용된 수의사는 해당 농장, 동물원 또는 수족관의 동물 에게 투여할 목적으로 처방대상 동물용 '
 '의약품에 대한 처방전을 발급할 수 있'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 6},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000037',
              'chunk_char_len': 172,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
