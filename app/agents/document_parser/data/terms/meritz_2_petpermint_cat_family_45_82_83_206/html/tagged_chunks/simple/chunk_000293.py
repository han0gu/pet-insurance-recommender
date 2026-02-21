from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 직접 진료하거나 검안한 수의사가<br>부득이한 사유로 진단서, 검안서 또는 증명서를 발급<br>할 수 없을 때에는 같은 '
 '동물병원에 종사하는 다른<br>수의사가 진료부 등에 의하여 발급할 수 있다.<br>② 제1항에 따른 진료 중 폐사(斃死)한 경우에 '
 '발급하는<br>폐사 진단서는 다른 수의사에게서 발급받을 수 있다.<br>③ 수의사는 직접 진료하거나 검안한 동물에 대한 진단<br>서, '
 '검안서, 증명서 또는 처방전의 발급을 요구받았<br>을 때에는 정당한 사유 없이 이를 거부하여서는 아니<br>된다.<br>④ 제1항부터'),
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
 'indexing': {'chunk_id': 'chunk_000293',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
