from langchain_core.documents import Document

chunk = Document(
    page_content=('다) "정신행동에 극심한 장해를 남긴 때" 라 함은 장해판정 직전 1년 이상 지\n'
 '속적인 정신건강의학과의 치료를 받았으며 GAF 30점 이하인 상태를 말한\n'
 '다.\n'
 '라) "정신행동에 심한 장해를 남긴 때" 라 함은 장해판정 직전 1년 이상 지속\n'
 '적인 정신건강의학과의 치료를 받았으며 GAF 40점 이하인 상태를 말한다.\n'
 '마) "정신행동에 뚜렷한 장해를 남긴 때" 라 함은 장해판정 직전 1년 이상 지\n'
 '속적인 정신건강의학과의 치료를 받았으며, 보건복지부고시 「장애정도판'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000836',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
