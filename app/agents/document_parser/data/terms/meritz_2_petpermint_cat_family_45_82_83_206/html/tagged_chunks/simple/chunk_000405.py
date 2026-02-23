from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만,<br>이미 보험금 지급사유가 발생한 경우에는 이에 대한<br>보험금은 지급합니다.</p><br><h1 id='61' "
 "style='font-size:20px'>【 예시 】</h1><br><p id='62' data-category='paragraph' "
 "style='font-size:16px'>입원특약에 가입한 피보험자가 20일간 입원하였음에도 불<br>구하고 입원확인서를 변조하여 "
 '입원일수 30일에 해당하는<br>보험금을 청구한 경우, 회사는 그 사실을 안 날로부터 1<br>개월 이내에 계약을 해지할 수 있습니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000405',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
