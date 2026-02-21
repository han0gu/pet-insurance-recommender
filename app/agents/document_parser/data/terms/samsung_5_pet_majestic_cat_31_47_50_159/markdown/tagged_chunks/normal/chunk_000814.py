from langchain_core.documents import Document

chunk = Document(
    page_content=('- 규칙」 제47조 제1항 및 제3항의 정상인의 신체 각 관절에 대한 평균 운동가\n'
 '- 능영역을 기준으로 정상각도 및 측정방법 등을 따른다.\n'
 '![image](/image/placeholder)\n'
 '③ ②\n'
 '④ 말절골(Distalphalaror)\n'
 '말절골 조면\n'
 '(Tuberosityofdistal |phalanx) 중절골 (Medial phalarod\n'
 '⑤ 원위지관절(제2지관절)\n'
 '기절골(Secondproximalphata) (Dipjoint)\n'
 '지골두(Headofphalarx) ①'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000814',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
