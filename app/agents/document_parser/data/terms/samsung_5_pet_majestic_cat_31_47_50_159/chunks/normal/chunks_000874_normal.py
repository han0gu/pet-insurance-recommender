from langchain_core.documents import Document

chunk = Document(
    page_content=('가) 한 눈의 안구(눈동자)의 주시야(머리를 움직이지 않고 눈만을 움직여서 볼 수 있는 범위)의 운동범위가 정상의 1/2 이하로 감소된 '
 '경우 나) 중심 20도 이내에서 복시(물체가 둘로 보이거나 겹쳐 보임)를 남긴 경우'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 137},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000874',
              'chunk_char_len': 122,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
