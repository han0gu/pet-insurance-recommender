from langchain_core.documents import Document

chunk = Document(
    page_content=("몸통) 한 개의 압박률이 40%이상<br>인 경우 또는 한 운동단위 내에 두 개 이상 척추</p><footer id='7' "
 "style='font-size:14px'>187</footer><p id='8' data-category='paragraph' "
 "style='font-size:20px'>체(척추뼈 몸통)의 압박골절로 각 척추체(척추뼈<br>몸통)의 압박률의 합이 60% 이상일 "
 "때</p><br><p id='9' data-category='paragraph' style='font-size:16px'>11) 약간의 "
 '기형이란 다음 중'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000996',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
