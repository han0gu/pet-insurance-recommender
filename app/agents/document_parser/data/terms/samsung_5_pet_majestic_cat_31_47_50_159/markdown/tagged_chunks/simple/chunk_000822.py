from langchain_core.documents import Document

chunk = Document(
    page_content=('- 8) 발가락 관절의 운동범위 측정은 장해평가시점의 「산업재해보상보험법 시행규\n'
 '- 칙」 제47조 제1항 및 제3항의 정상인의 신체 각 관절에 대한 평균 운동가능\n'
 '- 영역을 기준으로 정상각도 및 측정방법 등을 따른다.\n'
 '![image](/image/placeholder)\n'
 '말절골 (Datal)\n'
 '중절골 (Middle) 원위지관절(제2지관절)\n'
 '기절골(hodma)\n'
 '(Dipjoint)\n'
 '중족골 (Metatartal boned 지관절\n'
 '내측설상골\n'
 '근위지관절(제1지관절)\n'
 '(Pipjoint)\n'
 '중간설상골 외측설상골 (Lataral'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000822',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
