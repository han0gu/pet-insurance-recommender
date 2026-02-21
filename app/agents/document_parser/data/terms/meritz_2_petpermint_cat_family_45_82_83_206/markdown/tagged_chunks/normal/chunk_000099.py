from langchain_core.documents import Document

chunk = Document(
    page_content=('기간이 지나기 전까지 회사가 정한 방법에 따라 보험료의\n'
 '자동대출납입을 신청할 수 있으며, 이 경우 제36조(보험계\n'
 '약대출) 제1항에 따른 보험계약대출금으로 보험료가 자동으\n'
 '로 납입되어 계약은 유효하게 지속됩니다. 다만, 계약자가\n'
 '서면 이외에 인터넷 또는 전화(음성녹음) 등으로 자동대출\n'
 '납입을 신청할 경우 회사는 자동대출납입 신청내역을 서면\n'
 '또는 전화(음성녹음) 등으로 계약자에게 알려드립니다.\n'
 '\uf000 제1항의 규정에 따른 대출금과 보험료의 자동대출 납입\n'
 '일의 다음날부터 그 다음 보험료의 납입최고(독촉)기간까지'),
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
 'indexing': {'chunk_id': 'chunk_000099',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
