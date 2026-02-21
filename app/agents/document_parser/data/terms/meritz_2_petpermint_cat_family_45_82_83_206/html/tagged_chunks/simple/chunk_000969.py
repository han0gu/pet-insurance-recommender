from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 길이가 5mm 미만의 반흔(흉<br>터)은 합산대상에서 제외한다.<br>5) 추상(추한 모습)이 얼굴과 머리 또는 목 부위에 '
 '걸<br>쳐 있는 경우에는 머리 또는 목에 있는 흉터의 길이<br>또는 면적의 1/2을 얼굴의 추상(추한 모습)으로 보<br>아 '
 "산정한다.</p><h1 id='56' style='font-size:16px'>다"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000969',
              'chunk_char_len': 192,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
