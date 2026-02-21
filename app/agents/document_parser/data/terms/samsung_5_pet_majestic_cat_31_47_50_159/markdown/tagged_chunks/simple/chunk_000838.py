from langchain_core.documents import Document

chunk = Document(
    page_content=('㉱ 규칙적인 통원 + 약물 복용, ㉲ 소지품 및 금전관리나 적절한 구매행\n'
 '위, ㉳ 대중교통이나 일반공공시설의 이용- 바) "정신행동에 약간의 장해를 남긴 때" 라 함은 장해판정 직전 1년 이상 지\n'
 '- 속적인 정신건강의학과의 치료를 받았으며, 보건복지부고시 「장애정도판\n'
 '- 정기준」 의 "능력장애측정기준" 상 6개 항목 중 2개 항목 이상에서 독립\n'
 '- 적 수행이 불가능하여 타인의 도움이 필요하고 GAF 60점 이하인 상태를\n'
 '- 말한다.\n'
 '- 사) "정신행동에 경미한 장해를 남긴 때" 라 함은 장해판정 직전 1년 이상 지'),
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
 'indexing': {'chunk_id': 'chunk_000838',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
