from langchain_core.documents import Document

chunk = Document(
    page_content=('구 분 | 안면부 | 상지·하지\n'
 '지급액 | 수술 1cm당 14만원 | 수술 1cm당 7만원 (단, 3cm이상의 경우에 한합니다)\n'
 '② 제1항에서 길이측정이 불가한 피부이식수술 등의 경우 수술 cm는 최장직경으로 합니 다. ③ 제1항에서 정한 상해흉터복원(성형) '
 '수술비는 하나의 사고에 대하여 500만원을 한도 로 지급합니다. 다만, 동일부위에 대한 성형수술을 2회 이상 받은 경우에는 최초로 받 은 '
 '수술에 대해서만 지급합니다.\n'
 '<용어풀이>\n'
 '[안면부, 상지, 하지]'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 79},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['skin', 'head', 'joint']},
 'indexing': {'chunk_id': 'chunk_000423',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
