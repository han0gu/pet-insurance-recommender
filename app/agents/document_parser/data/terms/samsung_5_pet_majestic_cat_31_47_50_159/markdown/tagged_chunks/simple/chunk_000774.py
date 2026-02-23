from langchain_core.documents import Document

chunk = Document(
    page_content=('- 를 유합(아물어 붙음) 또는 고정한 상태\n'
 '- 나) 머리뼈(두개골)와 제1경추 또는 제1경추와 제2경추를 유합 또는 고정한 상\n'
 '- 태\n'
 '- 다) 머리뼈(두개골)와 상위목뼈(상위경추: 제1, 2경추) 사이에 CT 검사 상, 두개\n'
 '- 대후두공의 기저점(basion)과 축추 치돌기 상단사이의 거리(BDI : Basion-\n'
 '- Dental Interval)에 뚜렷한 이상전위가 있는 상태\n'
 '- 라) 상위목뼈(상위경추: 제1, 2경추) CT 검사상, 환추 전방 궁(arch)의 후방과'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000774',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
