from langchain_core.documents import Document

chunk = Document(
    page_content=('- 장해의 지급률을 합산한 지급률과 최초 장해의 지급률\n'
 '- 을 비교하여 그 중 높은 지급률을 적용한다.\n'
 '- 4) 의학적으로 뇌사판정을 받고 호흡기능과 심장박동기\n'
 '- 능을 상실하여 인공심박동기 등 장치에 의존하여 생명\n'
 '- 을 연장하고 있는 뇌사상태는 장해의 판정대상에 포함\n'
 '- 되지 않는다. 다만, 뇌사판정을 받은 경우가 아닌 식\n'
 '- 물인간상태(의식이 전혀 없고 사지의 자발적인 움직임\n'
 '- 이 불가능하여 일상생활에서 항시 간호가 필요한 상\n'
 '- 태)는 각 신체부위별 판정기준에 따라 평가한다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000589',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
