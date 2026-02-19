from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 사고일 또는 발병일부터 365일 이내인 경우에 한합니다. \uf000「반려동물 비용손해 관련 특별약관 일반조항」제15조(재 '
 '가입) 제6항에 따라 보험계약이 연장된 경우에는 종전 계약 의 보험기간을 연장하는 것으로 보아 제6항을 적용하지 않 습니다. \uf000 '
 '부활(효력회복)되는 이 특별약관의 보장개시는「반려동 물 비용손해 관련 특별약관 일반조항」제18조(보험료의 납 입을 연체하여 해지된 계약의 '
 '부활(효력회복))를 따릅니다. 이 경우 부활(효력회복)일을 계약일로 하여 제3항 및 제4항 의 보장개시일을 적용합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 119},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000356',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
