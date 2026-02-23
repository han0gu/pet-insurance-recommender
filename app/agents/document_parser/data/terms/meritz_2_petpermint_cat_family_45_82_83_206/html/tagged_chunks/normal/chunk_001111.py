from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>205</footer><table id='60' "
 "style='font-size:18px'><thead><tr><td>유형</td><td>제한정도에 따른 "
 '지급률</td></tr></thead><tbody><tr><td>배변 배뇨</td><td>- 배설을 돕기 위해 설치한 의료장치나 외과적 '
 '시술물을 사용함에 있어 타인의 계속적인 도움 이 필요한 상태 또는 지속적인 유치도뇨관 삽 입상태, 방광루, 요도루, 장루상태(20%) - '
 '화장실에 가서 변기위에 앉는 일(요강을 사용 하는 일 포함)과 대소변'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_001111',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
