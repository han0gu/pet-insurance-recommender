from langchain_core.documents import Document

chunk = Document(
    page_content=('- 4. 핵연료 물질(사용이 끝난 연료를 포함합니다. 이하 같습니다.) 또는 핵연료 물질에 의하여 오염된\n'
 '- 물질(원자핵분열 생성물을 포함합니다.)의 방사성, 폭발성 또는 그 밖의 유해한 특성에 의한 사고\n'
 '- 5. 위 제4호 이외의 방사선을 쬐는 것 또는 방사능 오염\n'
 '- 6. 최초계약의 보험개시일 이전에 이미 감염 또는 발병한 질병 및 상해\n'
 '- 7. 보험개시일로부터 그 날을 포함하여 30일 이내에 발생한 질병. 단, 이 계약이 갱신계약인 경우\n'
 '- 에는 적용하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000012',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
