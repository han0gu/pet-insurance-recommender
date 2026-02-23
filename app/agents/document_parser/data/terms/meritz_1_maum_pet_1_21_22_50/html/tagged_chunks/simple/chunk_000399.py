from langchain_core.documents import Document

chunk = Document(
    page_content=("id='90' style='font-size:14px'>- 49 -</footer><caption id='91' "
 "style='font-size:14px'><부표3> 단기요율표</caption><table id='92' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>보 험 기 간</td><td>단 기 요 "
 '율</td></tr><tr><td>7일까지</td><td>연요율의 10%</td></tr><tr><td>15일까지</td><td>연요율의'),
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
 'indexing': {'chunk_id': 'chunk_000399',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
