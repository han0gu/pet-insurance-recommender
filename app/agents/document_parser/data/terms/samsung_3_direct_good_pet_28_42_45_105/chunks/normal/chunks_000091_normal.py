from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 회사는 계약자가 제1회 보험료를 납입한 때부터 1년 이상 지난 유효한 계약으로서 그 보험종목의 변경을 요청할 때에는 회사의 '
 '사업방법서에서 정하는 방법에 따라 이를 변경하여 드립니다. ④ 회사는 계약자가 제1항 제5호에 따라 보험가입금액을 감액하고자 할 때에는 '
 '그 감액 된 부분은 해지된 것으로 보며, 이로써 회사가 지급하여야 할 해약환급금이 있을 때에 는 제36조(해약환급금) 제1항에 따른 '
 '해약환급금을 계약자에게 지급합니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 36},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000091',
              'chunk_char_len': 239,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
