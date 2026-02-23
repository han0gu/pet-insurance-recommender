from langchain_core.documents import Document

chunk = Document(
    page_content=('| 21 | 질 및 외음부 |\n'
 '| 22 | 전립선 |\n'
 '| 23 | 유방(유선 포함) |\n'
 '| 24 | 자궁[자궁체부(자궁몸통) 포함] |\n'
 '| 25 | 자궁체부(자궁몸통)(제왕절개술을 받은 경우에 한함) |\n'
 '- 136 -| 구분 | 특 정 신 체 부 위 |\n'
 '| --- | --- |\n'
 '| 26 | 난소 및 난관 |\n'
 '| 27 | 고환[고환초막(고환집막) 포함], 부고환, 정관, 정삭 및 정낭 |\n'
 '| 28 | 갑상선 |\n'
 '| 29 | 부갑상선 |'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000739',
              'chunk_char_len': 244,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
