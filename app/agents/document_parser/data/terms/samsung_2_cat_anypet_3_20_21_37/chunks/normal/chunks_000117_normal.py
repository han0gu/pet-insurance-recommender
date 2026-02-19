from langchain_core.documents import Document

chunk = Document(
    page_content=('. 지진, 분화, 해일, 홍수 또는 이와 비슷한 천재지변 6. 전쟁, 외국의 무력행사, 혁명, 내란, 사변, 폭동, 소요, 기타 이들과 '
 '유사한 사태 7. 핵연료물질(사용이 끝난 연료를 포함합니다. 이하 같습니다) 또는 핵연료 물질에 의하여 오염된 물 질(원자핵분열 생성물을 '
 '포함합니다)의 방사성, 폭발성 또는 그 밖의 유해한 특성에 의한 사고 8. 위 제7호 이외의 방사선을 쬐는 것 또는 방사능 오염 9. '
 '국가 및 지방자치단체의 명령 또는 법률에 의한 살처분 또는 이와 유사한 사태 10'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 23},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000117',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
