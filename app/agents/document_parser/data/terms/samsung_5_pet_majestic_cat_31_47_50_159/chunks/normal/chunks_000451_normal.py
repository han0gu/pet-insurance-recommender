from langchain_core.documents import Document

chunk = Document(
    page_content=('<관련법규>\n'
 '[의료법 제3조(의료기관)에 규정한 종합병원]\n'
 '100개 이상의 병상 구비, 병상수에 따라 일정 개수의 진료과목을 갖추고, 각 진료과목마다 전속하 는 전문의를 둔 병원을 말합니다.\n'
 '제 3조 (입원의 정의와 장소)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 83},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000451',
              'chunk_char_len': 124,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
