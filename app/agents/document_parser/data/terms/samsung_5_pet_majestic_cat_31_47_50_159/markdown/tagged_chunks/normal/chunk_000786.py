from langchain_core.documents import Document

chunk = Document(
    page_content=('천골 (Sacrum)\n'
 '관골구 미골(Coccyx)\n'
 '(Acetabulum)\n'
 '치골 치골결절 (Pubictubercle)\n'
 '(Pubis)\n'
 '치골결합 (Pubicsymphysis)\n'
 '좌골\n'
 '(Ischium) 치골하각 (Subpubicangle)\n'
 '치골궁\n'
 '(Pubicarch)\n'
 '대골반(가골반, Greater pelvis)\n'
 '소골반(진골반, Lesser pelvis)\n'
 'く 골반뼈 > |\n'
 '# 8. 팔의 장해# 가. 장해의 분류| 장 해 의 분 류 | 지급률(%) |\n'
 '| --- | --- |\n'
 '| 1) 두 팔의 손목 이상을 잃었을 때 | 100 |'),
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
 'indexing': {'chunk_id': 'chunk_000786',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
