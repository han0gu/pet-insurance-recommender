from langchain_core.documents import Document

chunk = Document(
    page_content=("제2조(정의)에서 정한 국내 동물병원(이하 「동물</p><footer id='18' "
 "style='font-size:14px'>158</footer><p id='19' data-category='paragraph' "
 "style='font-size:16px'>병원」이라 합니다)에 입원하여 수의사법 제2조(정의)에서<br>정한 수의사(이하 「수의사」라 "
 '합니다)에게 치료를 받은<br>때에는 피보험자가 부담한 반려동물의 치료비(각종 할인 및<br>감면, 사후환급금액 등을 제외한 실수납액을 '
 '말합니다)를<br>이 약관에 따라 보험수익자에게'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000784',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
