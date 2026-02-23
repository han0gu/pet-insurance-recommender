from langchain_core.documents import Document

chunk = Document(
    page_content=('등)에 따라<br>승낙한 경우에 건강진단서 사본 등에 명기되어 있는<br>사항으로 보험금 지급사유가 발생하였을 때(계약자 또<br>는 '
 '피보험자가 회사에 제출한 기초자료의 내용 중 중<br>요사항을 고의로 사실과 다르게 작성한 때에는 계약을<br>해지할 수 '
 '있습니다)<br>⑤ 보험설계사 등이 계약자 또는 피보험자에게 알릴 기회<br>를 주지 않았거나 계약자 또는 피보험자가 '
 "사실대로<br>알리는 것을 방해한 경우, 계약자 또는 피보험자에게</p><footer id='75'"),
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
 'indexing': {'chunk_id': 'chunk_000331',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
