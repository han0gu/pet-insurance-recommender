from langchain_core.documents import Document

chunk = Document(
    page_content=('가) 완전 강직(관절굳음)\n'
 '나) 근전도 검사상 완전손상(complete injury) 소견이 있으면서 도수근력검사 (MMT)에서 근력이 "0등급(zero)" 인 '
 '경우\n'
 '8) "관절 하나의 기능에 심한 장해를 남긴 때" 라 함은 아래의 경우 중 하나에 해 당하는 때를 말한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 145},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000942',
              'chunk_char_len': 152,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
