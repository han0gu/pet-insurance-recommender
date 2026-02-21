from langchain_core.documents import Document

chunk = Document(
    page_content=('오염된 물질의<br>방사성, 폭발성, 그 밖의 유해한 특성 또는 이들의<br>특성에 의한 사고<br>⑤ 제4호 이외의 방사선을 쬐는 것 '
 "또는 방사능 오염</p><br><p id='26' data-category='paragraph' "
 "style='font-size:20px'>【핵연료물질】</p><br><p id='27' data-category='paragraph' "
 "style='font-size:20px'>사용된 연료를 포함합니다.</p><br><p id='28' "
 "data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000721',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
