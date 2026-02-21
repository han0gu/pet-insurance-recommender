from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>163</footer><p id='62' data-category='paragraph' "
 "style='font-size:20px'>제4조(입원의 정의와 장소)</p><br><p id='63' "
 "data-category='paragraph' style='font-size:16px'>이 특별약관에 있어서 「입원」이라 함은 수의사가 "
 '상해 또<br>는 질병의 치료가 필요하다고 인정한 경우로서, 자택 등에<br>서의 치료가 곤란하여 동물병원에 입실하여 수의사의 '
 '관리<br>하에 치료에 전념하는 것을'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000820',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
