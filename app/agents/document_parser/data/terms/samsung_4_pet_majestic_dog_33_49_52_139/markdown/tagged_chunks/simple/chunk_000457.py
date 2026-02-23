from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 반려견이 제1항의 사고로 치료를 받던 중에 보험기간이 만료된 경우에도 만료일부터\n'
 '- 90일 이내의 의료비는 보상하여 드립니다. 다만, 사고일 또는 발병일부터 180일이내\n'
 '- 의 치료인 경우에 한합니다.\n'
 '③ 제1항의 손해에 대한 보장개시일(책임개시일)은 이 특별약관의 보험계약일(이하 「보\n'
 '험계약일」 이라 합니다)부터 그 날을 포함하여 30일(단, 암, 백내장, 녹내장, 심장질환\n'
 ', 신장질환, 방광질환 및 각종 결석의 경우 90일)이 지난 날의 다음날로 합니다. 다만,'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'eye', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000457',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
