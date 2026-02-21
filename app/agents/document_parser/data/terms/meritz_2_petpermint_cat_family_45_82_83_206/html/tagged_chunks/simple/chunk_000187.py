from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 계약자가<br>서면 이외에 인터넷 또는 전화(음성녹음) 등으로 자동대출<br>납입을 신청할 경우 회사는 자동대출납입 신청내역을 '
 '서면<br>또는 전화(음성녹음) 등으로 계약자에게 알려드립니다.<br>\uf000 제1항의 규정에 따른 대출금과 보험료의 자동대출 '
 '납입<br>일의 다음날부터 그 다음 보험료의 납입최고(독촉)기간까지<br>의 이자(보험계약대출이율을 적용하여 계산)를 더한 '
 '금액이<br>해당 보험료가 납입된 것으로 계산한 해약환급금과 계약자<br>에게 지급할 기타 모든 지급금의 합계액에서 계약자의 '
 '회사<br>에 대한 모든'),
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
 'indexing': {'chunk_id': 'chunk_000187',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
