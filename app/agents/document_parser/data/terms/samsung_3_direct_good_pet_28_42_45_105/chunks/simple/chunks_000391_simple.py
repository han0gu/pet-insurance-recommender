from langchain_core.documents import Document

chunk = Document(
    page_content=('71 / 181\n'
 '알리지 않았을 경우 변경후 요율이 변경전 요율보다 높을 때에는 회사는 그 변경사실 을 안 날부터 1개월 이내에 계약자 또는 피보험자에게 '
 '제4항에 따라 보장됨을 통보하 고 이에 따라 보험금을 지급합니다.\n'
 '제13조 (알릴 의무 위반의 효과)\n'
 '① 회사는 아래와 같은 사실이 있을 경우에는 손해의 발생여부에 관계없이 이 특별약관 을 해지할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 72},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000391',
              'chunk_char_len': 202,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
