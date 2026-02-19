from langchain_core.documents import Document

chunk = Document(
    page_content=('. ② 제1항의 수술은 보건복지부 산하 신의료기술평가위원회(향후 제도변경 시에는 동 위 원회와 동일한 기능을 수행하는 기관)로부터 '
 '안전성과 치료효과를 인정받은 최신 수 술기법으로 생체에 절단, 절제 등의 조작을 가하는 것을 포함합니다. 또한 레이저 (Laser)를 '
 '이용하여 생체에 절단, 절제 등의 조작을 가하는 것을 포함합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 75},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000391',
              'chunk_char_len': 184,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
