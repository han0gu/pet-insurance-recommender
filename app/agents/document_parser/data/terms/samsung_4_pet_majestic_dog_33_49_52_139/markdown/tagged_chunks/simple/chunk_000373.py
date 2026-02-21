from langchain_core.documents import Document

chunk = Document(
    page_content=('- 의 조치\n'
 '- 3. 신경(神經) BLOCK(신경의 차단)\n'
 '- 4. 상해 원인 외 단순 미용성형 목적의 수술\n'
 '- 5. 피임(避妊) 목적의 수술\n'
 '- 6. 검사 및 진단을 위한 수술(생검(生檢), 복강경검사(腹腔鏡檢査) 등)\n'
 '- 83 -7. 기타 수술의 정의에 해당하지 않는 시술| <예시안내> |\n'
 '| --- |'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000373',
              'chunk_char_len': 172,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
