from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 계약자 및 피보험자의 고의, 중대한 과실 2. 지진, 분화, 풍수해 및 이와 유사한 자연재해로 생긴 손해 3. 전쟁, 혁명, 내란, '
 '폭동, 소요 기타 유사한 사태로 생긴 손해 4. 피보험자의 질병, 심신상실 또는 정신질환으로 인한 손해 5. 핵연료물질 또는 핵연료물질에 '
 '의하여 오염된 물질의 방사성, 폭발성, 그 밖의 유해한 특성 또 는 이들의 특성에 의한 사고로 생긴 손해. 그리고, 위 이외의 방사선 '
 '조사(照射) 또는 방사능 오 염으로 인한 손해. 6. 보험개시일 이전에 이미 감염 또는 발병한 질병 및 상해'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 6},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['head',
                             'dental',
                             'skin',
                             'joint',
                             'urinary',
                             'eye',
                             'digestive',
                             'other']},
 'indexing': {'chunk_id': 'chunk_000014',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
