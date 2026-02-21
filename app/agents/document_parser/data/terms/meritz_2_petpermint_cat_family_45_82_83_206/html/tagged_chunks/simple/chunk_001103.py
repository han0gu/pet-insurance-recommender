from langchain_core.documents import Document

chunk = Document(
    page_content=("뇌전증</h1><br><p id='51' data-category='list' style='font-size:16px'>가) "
 '“뇌전증”이라 함은 돌발적 뇌파이상을 나타내는<br>뇌질환으로 발작(경련, 의식장해 등)을 반복하는<br>것을 말한다.<br>나) 뇌전증 '
 '발작의 빈도 및 양상은 지속적인 항뇌전<br>증제(항경련제) 약물로도 조절되지 않는 뇌전증<br>을 말하며, 진료기록에 기재되어 '
 "객관적으로 확<br>인되는 뇌전증 발작의 빈도 및 양상을 기준으로</p><footer id='52'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_001103',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
