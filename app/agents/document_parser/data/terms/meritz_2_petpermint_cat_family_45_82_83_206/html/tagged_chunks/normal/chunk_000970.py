from langchain_core.documents import Document

chunk = Document(
    page_content=(". 뚜렷한 추상(추한 모습)</h1><br><h1 id='57' style='font-size:20px'>1) 얼굴</h1><br><p "
 "id='58' data-category='paragraph' style='font-size:16px'>가) 손바닥 크기 1/2 이상의 "
 '추상(추한 모습)<br>나) 길이 10cm 이상의 추상 반흔(추한 모습의 흉터)<br>다) 지름 5cm 이상의 조직함몰<br>라) 코의 '
 "1/2이상 결손</p><br><h1 id='59' style='font-size:20px'>2) 머리</h1><br><p"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000970',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
