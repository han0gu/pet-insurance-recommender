from langchain_core.documents import Document

chunk = Document(
    page_content=('가. 이물제거 목적으로 인한 치료여부가 확인 가능한 동물병원 진료기록부 나. 이물제거(내시경) 시행한 경우: 이물제거(내시경) 처치가 '
 '명시된 동물병원 진 료비 내역서(진료항목이 기재되어 있는 명세서, 수의사 처방전 포함), 의료비 영 수증 및 내시경영상검사결과지 등 다. '
 '이물제거(구토유도약물) 시행한 경우: 구토유도약물 처방이 명시된 동물병원 진료비 내역서(진료항목이 기재되어 있는 명세서, 수의사 처방전 '
 '포함), 의료 비 영수증 및 수의사처방전 등'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 79},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive', 'other']},
 'indexing': {'chunk_id': 'chunk_000482',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
