from langchain_core.documents import Document

chunk = Document(
    page_content=('- 라. 의학적으로 뇌사판정을 받고 호흡기능과 심장박동기능을 상실하여 인공심박동기\n'
 '- 등 장치에 의존하여 생명을 연장하고 있는 뇌사상태는 장해의 판정대상에 포함되\n'
 '- 지 않는다. 다만, 뇌사판정을 받은 경우가 아닌 식물인간상태(의식이 전혀 없고\n'
 '- 사지의 자발적인 움직임이 불가능하여 일상생활에서 항시 간호가 필요한 상태)는\n'
 '- 각 신체부위별 판정기준에 따라 평가한다.\n'
 '- 마. 장해진단서에는 ① 장해진단명 및 발생시기 ② 장해의 내용과 그 정도 ③ 사고와'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000733',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
