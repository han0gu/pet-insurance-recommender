from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 4-1. 반려묘 의료비(치과및구강질환포함)(재가입형) 특별약관 제16조(특별약관의 무효) 2. 보험증권에 기재된 반려묘가 '
 '보험계약일부터 제1조(보험금의 지급사유) 제3항에 정한 손해에 대한 보장개시일(책임개시일)의 전일 이전에 사망한 경우. 다만, 제6 '
 '조(보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))에 의하여 부활( 효력회복)된 특별약관의 부활(효력회복)일부터 '
 '제1조(보험금의 지급사유) 제3항에'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 111},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000693',
              'chunk_char_len': 234,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
