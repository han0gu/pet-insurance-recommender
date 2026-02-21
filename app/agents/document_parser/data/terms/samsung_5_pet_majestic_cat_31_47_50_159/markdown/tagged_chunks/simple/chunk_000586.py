from langchain_core.documents import Document

chunk = Document(
    page_content=('- ③ 보험의 목적이 다수인 경우 제1항 내지 제2항은 보험의 목적별로 각각 적용합니다.\n'
 '제6조 (보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))부활(효력회복)되는 특별약관의 보장개시는 4-1. 반려묘 '
 '의료비(치과및구강질환포함) 특\n'
 '별약관(재가입형) 제22조(보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))\n'
 '를 따릅니다. 이 경우 부활(효력회복)일을 보험계약일로 하여 제1조(보험금의 지급사유)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000586',
              'chunk_char_len': 234,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
