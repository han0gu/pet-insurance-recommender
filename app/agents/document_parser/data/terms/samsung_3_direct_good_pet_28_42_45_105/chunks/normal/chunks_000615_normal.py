from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 갱신될 갱신형 특별약관(이하 「갱신계약」 이라 합니다.)의 보험기간이 회사가 이 보험의 사업방법서에서 정한 기간 내일 것 2. '
 '갱신전 계약의 보험기간이 끝난 날의 다음날(이하 「갱신일」 이라 합니다)에 피보험 자의 나이 또는 피보험자의 반려견 나이가 이 보험의 '
 '사업방법서에서 정한 범위 내일 것 3. 보통약관 제27조(보험료의 납입이 연체되는 경우 납입최고(독촉)와 계약의 해지)에 정한 '
 '납입최고(독촉)기간 내에 갱신전 계약의 보험료가 납입완료 되었을 것'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 98},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000615',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
