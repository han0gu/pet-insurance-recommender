from langchain_core.documents import Document

chunk = Document(
    page_content=('제3조(단체요율의 적용)\n'
 '① 제1조의 단체는 단체요율을 적용할 수 있습니다. 다만, 제3종 단체는 구성원이 명확하고 위험의 동 질성이 확보되어야 합니다. ② 단체 '
 '구성원의 일부만을 대상으로 가입하는 경우에는 대상단체의 위험과 피보험단체의 위험의 동 질성이 유지되어야 합니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 27},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000134',
              'chunk_char_len': 153,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
