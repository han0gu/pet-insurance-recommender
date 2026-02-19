from langchain_core.documents import Document

chunk = Document(
    page_content=('【소득세법 시행규칙 제61조의3 (공제대상보험료의 범위)】\n'
 '영 제118조의4 제2항 각 호 외의 부분에서 "기획재정부령으로 정하는 것"이란 만기에'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 44},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000240',
              'chunk_char_len': 81,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
