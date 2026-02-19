from langchain_core.documents import Document

chunk = Document(
    page_content=('제3조(단체요율의 적용)\n'
 '① 제1조의 단체는 단체요율을 적용할 수 있습니다. 다만, 제3종 단체는 구성원이 명확하고 위험의 동 질성이 확보되어야 합니다. ② 단체 '
 '구성원의 일부만을 대상으로 가입하는 경우에는 대상단체의 위험과 피보험단체의 위험의 동 질성이 유지되어야 합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 35},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000177',
              'chunk_char_len': 153,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
