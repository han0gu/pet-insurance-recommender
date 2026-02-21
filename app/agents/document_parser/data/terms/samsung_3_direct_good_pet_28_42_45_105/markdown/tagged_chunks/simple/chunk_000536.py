from langchain_core.documents import Document

chunk = Document(
    page_content=('- 보험금이 지급된 경우에는 갱신계약에서 보상하지 않습니다.\n'
 '# 제4조 (갱신계약 제1회 보험료의 납입최고(독촉)와 갱신계약의 해제)① 계약자가 갱신계약의 제1회 보험료를 납입기일까지 납입하지 않은 '
 '때에는 보통약관\n'
 '제27조(보험료의 납입이 연체되는 경우 납입최고(독촉)와 계약의 해지)에 따라 납입최\n'
 '고(독촉)하며, 이 납입최고(독촉)기간 안에 보험료가 납입되지 않은 경우 납입최고(독\n'
 '촉)기간이 끝나는 날의 다음날 갱신 계약을 해제합니다.\n'
 '② 회사는 납입최고(독촉)기간 안에 발생한 사고에 대하여 약정한 보험금을 지급합니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000536',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
