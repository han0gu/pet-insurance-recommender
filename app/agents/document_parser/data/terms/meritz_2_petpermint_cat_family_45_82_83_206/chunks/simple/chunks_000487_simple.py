from langchain_core.documents import Document

chunk = Document(
    page_content=('병원」이라 합니다)에 통원하여 수의사법 제2조(정의)에서 정한 수의사(이하 「수의사」라 합니다)에게 치료를 받은 때에는 피보험자가 부담한 '
 '반려동물의 치료비(각종 할인 및 감면, 사후환급금액 등을 제외한 실수납액을 말합니다)를 이 약관에 따라 보험수익자에게 1일당 제2항에서 '
 '정한 지급 한도 내에서 보상합니다. 다만, 연간 지급하는 총 보험금은 보험증권에 기재된 연간 총 보상한도액(700만원)을 한도로 '
 '합니다.\n'
 '【수의사법 제2조(정의)】\n'
 '이 법에서 사용하는 용어의 뜻은 다음과 같다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 147},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000487',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
