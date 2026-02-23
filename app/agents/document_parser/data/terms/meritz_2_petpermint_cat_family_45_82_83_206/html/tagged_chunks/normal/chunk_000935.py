from langchain_core.documents import Document

chunk = Document(
    page_content=("못하는 경우<br>를 말한다.<br>4) “약간의 장해를 남긴 때”라 함은 순음청력검사 결과</p><footer id='19' "
 "style='font-size:14px'>179</footer><p id='20' data-category='paragraph' "
 "style='font-size:20px'>평균순음역치가 70dB이상인 경우에 해당되어, 50cm<br>이상의 거리에서는 보통의 말소리를 "
 "알아듣지 못하<br>는 경우를 말한다.</p><br><p id='21' data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000935',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
