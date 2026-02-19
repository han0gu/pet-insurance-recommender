from langchain_core.documents import Document

chunk = Document(
    page_content=('② 제1항의 수술은 보건복지부 산하 신의료기술평가위원회(향후 제도변경 시에는 동 위 원회와 동일한 기능을 수행하는 기관)로부터 안전성과 '
 '치료효과를 인정받은 최신 수 술기법으로 생체에 절단, 절제 등의 조작을 가하는 것을 포함합니다. 또한 레이저 (Laser)를 이용하여 '
 '생체에 절단, 절제 등의 조작을 가하는 것도 포함됩니다.\n'
 '<용어풀이>\n'
 '[신의료기술평가위원회]\n'
 '의료법 제54조(신의료기 술평가위원회의 설치 등)에 의거 설치된 위원회로서 신의료기술에 관한 최고의 심의기구를 말합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 77},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000415',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
