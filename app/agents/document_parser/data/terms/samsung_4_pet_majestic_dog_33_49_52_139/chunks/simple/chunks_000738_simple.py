from langchain_core.documents import Document

chunk = Document(
    page_content=('제6조 (보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))\n'
 '부활(효력회복)되는 특별약관의 보장개시는 4-1. 반려견 의료비(치과및구강질환포함)(수 술당일제외, 검사비포함)(재가입형) 특별약관 '
 '제22조(보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))를 따릅니다. 이 경우 부활(효력회복)일을 보험계약일로 하여 '
 '제1조(보험금의 지급사유) 제3항을 적용합니다.\n'
 '제 7조 (특별약관의 자동갱신)\n'
 '이 특별약관은 제도성 특별약관 5-1. [갱신형] 특별약관의 자동갱신 특별약관에 따라 갱 신됩니다.\n'
 '제 8조 (준용규정)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 119},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000738',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
