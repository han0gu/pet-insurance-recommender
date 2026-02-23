from langchain_core.documents import Document

chunk = Document(
    page_content=('않는 기간의 종료일을 포함하여 계속하여 입원한 경우<br>그 입원에 대해서는 회사가 보험금을 지급하지 않는 기간<br>종료일의 다음날을 '
 '입원의 개시일로 인정하여 보험금을 지<br>급합니다.<br>\uf000 반려동물에게 보험금의 지급사유가 발생했을 경우, 그<br>보험금의 '
 '지급사유가 특정질병을 직접적인 원인으로 발생한<br>보험금의 지급사유인지 아닌지는 수의사의 진단서와 의견을<br>주된 판단자료로 하여 '
 "결정합니다.</p><h1 id='3' style='font-size:16px'>제3조(특별약관의 부활(효력회복))</h1><br><p"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000841',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
