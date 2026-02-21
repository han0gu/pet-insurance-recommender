from langchain_core.documents import Document

chunk = Document(
    page_content=("평가하지 않는<br>다.<br>6) 심한 운동장해란 다음 중 어느 하나에 해당하는 경우를<br>말한다.</p><br><p id='84' "
 "data-category='list' style='font-size:20px'>가) 척추체(척추뼈 몸통)에 골절 또는 탈구로 4개 "
 '이<br>상의 척추체(척추뼈 몸통)를 유합(아물어 붙음)<br>또는 고정한 상태<br>나) 머리뼈(두개골), 제1경추, 제2경추를 모두 '
 "유합</p><footer id='85' style='font-size:14px'>186</footer><h1 id='0'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'head', 'joint']},
 'indexing': {'chunk_id': 'chunk_000989',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
