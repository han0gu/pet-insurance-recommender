from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 4-1. 반려견 의료비(치과및구강질환포함)(수술당일제외, 검사비포함)(재가입형) 특 별약관 제16조(특별약관의 무효) 2. '
 '보험증권에 기재된 반려견이 보험계약일부터 제1조(보험금의 지급사유) 제3항에 정한 손해에 대한 보장개시일(책임개시일)의 전일 이전에 '
 '사망한 경우. 다만, 제6 조(보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))에 의하여 부활( 효력회복)된 특별약관의 '
 '부활(효력회복)일부터 제1조(보험금의 지급사유) 제3항에'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 118},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000734',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
