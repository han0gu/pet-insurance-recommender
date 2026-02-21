from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 납입최고(독촉)기간 안에 발생한 사고에 대하여 약정한 보험금을 지급합니다.\n'
 '이 경우 계약자는 즉시 갱신계약 보험료를 납입하여야 합니다. 만약, 이 보험료를 납\n'
 '입하지 않으면 회사는 지급할 보험금에서 이를 차감할 수 있습니다.- \n'
 '# 제5조 (갱신일 이후 부활(효력회복)을 청약하는 경우 연체된 보험료의 적용)보통약관 제28조(보험료의 납입을 연체하여 해지된 계약의 '
 '부활(효력회복)) 1항에서 정한\n'
 '연체된 보험료는 갱신일부터 부활(효력회복)을 청약한 날까지의 납입이 연체된 보험료를'),
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
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000537',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
