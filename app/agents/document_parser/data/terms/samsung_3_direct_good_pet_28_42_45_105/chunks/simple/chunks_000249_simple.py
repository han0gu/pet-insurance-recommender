from langchain_core.documents import Document

chunk = Document(
    page_content=('① 계약자는 제29조(보험료의 납입이 연체되는 경우 납입최고(독촉)와 특별약관의 해지) 에 따른 보험료의 납입최고(독촉)기간이 지나기 '
 '전까지 회사가 정한 방법에 따라 보험 료의 자동대출납입을 신청할 수 있으며, 이 경우 제36조(보험계약대출) 제1항에 따른 '
 '보험계약대출금으로 보험료가 자동으로 납입되어 계약은 유효하게 지속됩니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 54},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000249',
              'chunk_char_len': 183,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
