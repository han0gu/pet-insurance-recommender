from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[안면부, 상지, 하지]\n'
 '1. 안면부란 이마를 포함하여 목까지의 얼굴부분을 말합니다. 2. 상지란 어깨관절 이하의 팔과 손가락 부분을 말합니다.\n'
 '3. 하지란 엉덩이관절 이하 다리와 발가락 부분을 말합니다.\n'
 '제2조 (보험금 지급에 관한 세부규정)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 79},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000424',
              'chunk_char_len': 143,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
