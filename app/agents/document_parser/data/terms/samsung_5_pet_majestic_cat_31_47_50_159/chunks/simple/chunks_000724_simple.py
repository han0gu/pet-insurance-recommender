from langchain_core.documents import Document

chunk = Document(
    page_content=('하고 가입동물이 보험에 가입한 동물과 동일함을 확인 후 보험금을 지급합니다.\n'
 '4. 사고증명서(진료비 내역서(진료항목이 기재되어 있는 명세서, 수의사 처방전 포함) 및 의료비 영수증 등)\n'
 '가. 이물제거 목적으로 인한 치료여부가 확인 가능한 동물병원 진료기록부 나. 이물제거(내시경) 시행한 경우: 이물제거(내시경) 처치가 '
 '명시된 진료비 영수증 (치료비 세부내역 포함), 내시경영상검사결과지 다. 이물제거(구토유도약물) 시행한 경우: 구토유도약물 처방이 명시된 '
 '동물병원 진료비 영수증(치료비 세부내역 포함) 및 수의사처방전'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 115},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000724',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
