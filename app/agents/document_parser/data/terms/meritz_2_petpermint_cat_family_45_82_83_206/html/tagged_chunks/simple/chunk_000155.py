from langchain_core.documents import Document

chunk = Document(
    page_content=('. 전자문서에 보험금 지급사유, 보험금액, 보험계약자와<br>보험수익자의 신원, 보험기간이 적혀 있을 것<br>2. 전자문서에 법 '
 '제731조제1항에 따른 전자서명(이하<br>“전자서명”이라 한다)을 하기 전에 전자서명을 할<br>사람을 직접 만나서 전자서명을 하는 '
 '사람이 보험계약<br>에 동의하는 본인임을 확인하는 절차를 거쳐 작성될<br>것<br>3'),
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
 'indexing': {'chunk_id': 'chunk_000155',
              'chunk_char_len': 197,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
