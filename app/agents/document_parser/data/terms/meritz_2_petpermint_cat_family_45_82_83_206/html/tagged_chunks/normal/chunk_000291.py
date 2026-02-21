from langchain_core.documents import Document

chunk = Document(
    page_content=('제1항 제2호의 사고증명서는 수의사법 제12조(진단서<br>등)에서 규정한 내용에 따라 국내의 동물병원에서 수의사에<br>의해 발급한 '
 "것이어야 합니다.</p><br><h1 id='19' style='font-size:20px'>【수의사법 제12조(진단서 "
 "등)】</h1><br><p id='20' data-category='list' style='font-size:16px'>① 수의사는 "
 '자기가 직접 진료하거나 검안하지 아니하고<br>는 진단서, 검안서, 증명서 또는 처방전을 발급하지<br>못하며, 「약사법」 '
 '제85조제6항에 따른'),
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
 'indexing': {'chunk_id': 'chunk_000291',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
