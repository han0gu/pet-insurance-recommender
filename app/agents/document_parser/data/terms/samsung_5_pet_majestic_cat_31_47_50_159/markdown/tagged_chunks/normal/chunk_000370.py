from langchain_core.documents import Document

chunk = Document(
    page_content=('- 상해흉터복원」의 치료를 직접적인 목적으로 의료기구를 사용하여 생체(生體)에 절단\n'
 '- (切斷, 특정부위를 잘라내는 것), 절제(切除, 특정부위를 잘라 없애는 것) 등의 조작(操\n'
 '- 作 )을 가하는 것을 말합니다.\n'
 '- ② 제1항의 수술은 보건복지부 산하 신의료기술평가위원회(향후 제도변경 시에는 동 위\n'
 '- 원회와 동일한 기능을 수행하는 기관)로부터 안전성과 치료효과를 인정받은 최신 수\n'
 '- 술기법으로 생체에 절단, 절제 등의 조작을 가하는 것을 포함합니다. 또한 레이저'),
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
 'indexing': {'chunk_id': 'chunk_000370',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
