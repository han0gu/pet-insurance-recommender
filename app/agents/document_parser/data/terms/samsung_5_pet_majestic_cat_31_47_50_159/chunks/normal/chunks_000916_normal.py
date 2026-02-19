from langchain_core.documents import Document

chunk = Document(
    page_content=('라) 상위목뼈(상위경추: 제1, 2경추) CT 검사상, 환추 전방 궁(arch)의 후방과 치상돌기의 전면과의 거리(ADI: '
 'Atlanto-Dental Interval)에 뚜렷한 이상전위 가 있는 상태'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 141},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint', 'head']},
 'indexing': {'chunk_id': 'chunk_000916',
              'chunk_char_len': 111,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
