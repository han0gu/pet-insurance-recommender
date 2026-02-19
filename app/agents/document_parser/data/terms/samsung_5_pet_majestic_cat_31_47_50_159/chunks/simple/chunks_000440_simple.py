from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[신의료기술평가위원회]\n'
 '의료법 제54조(신의료기술평가위원회의 설치 등)에 의거 설치된 위원회로서 신의료기술에 관한 최 고의 심의기구를 말합니다.\n'
 '③ 제1항의 수술에서 아래에 정한 사항은 제외합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 81},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000440',
              'chunk_char_len': 117,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
