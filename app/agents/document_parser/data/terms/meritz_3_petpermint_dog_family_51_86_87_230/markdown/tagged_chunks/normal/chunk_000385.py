from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000「반려동물 비용손해 관련 특별약관 일반조항」제15조(재\n'
 '가입) 제6항에 따라 보험계약이 연장된 경우에는 종전 계약\n'
 '의 보험기간을 연장하는 것으로 보아 제6항을 적용하지 않\n'
 '습니다.\n'
 '\uf000 부활(효력회복)되는 이 특별약관의 보장개시는「반려동\n'
 '\uf000\n'
 '물 비용손해 관련 특별약관 일반조항」제18조(보험료의 납\n'
 '입을 연체하여 해지된 계약의 부활(효력회복))를 따릅니다.\n'
 '이 경우 부활(효력회복)일을 계약일로 하여 제5항 및 제6항\n'
 '의 보장개시일을 적용합니다.# ② 기본형\uf000 회사는 보험기간 중에 보험증권에 기재된 반려동물에게'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000385',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
