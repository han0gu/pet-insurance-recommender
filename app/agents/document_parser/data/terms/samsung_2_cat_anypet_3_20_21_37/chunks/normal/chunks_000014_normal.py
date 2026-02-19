from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 계약자, 피보험자, 이들의 가족 또는 사용인의 고의 또는 중대한 과실 2. 지진, 분화, 해일, 홍수 또는 이와 유사한 자연재해로 '
 '생긴 손해 3. 전쟁, 외국의 무력행사, 혁명, 내란, 사변, 폭동, 소요, 기타 이들과 유사한 사태 4. 핵연료 물질(사용이 끝난 '
 '연료를 포함합니다. 이하 같습니다.) 또는 핵연료 물질에 의하여 오염된 물질(원자핵분열 생성물을 포함합니다.)의 방사성, 폭발성 또는 그 '
 '밖의 유해한 특성에 의한 사고 5. 위 제4호 이외의 방사선을 쬐는 것 또는 방사능 오염 6'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 6},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000014',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
