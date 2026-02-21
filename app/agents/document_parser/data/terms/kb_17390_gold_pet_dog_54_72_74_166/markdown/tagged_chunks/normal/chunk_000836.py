from langchain_core.documents import Document

chunk = Document(
    page_content=('- 4) 의학적으로 뇌사판정을 받고 호흡기능과 심장박동기능을 상실하여 인공심박\n'
 '- 동기 등 장치에 의존하여 생명을 연장하고 있는 뇌사상태는 장해의 판정대상\n'
 '- 에 포함되지 않는다. 다만, 뇌사판정을 받은 경우가 아닌 식물인간상태(의식\n'
 '- 이 전혀 없고 사지의 자발적인 움직임이 불가능하여 일상생활에서 항시 간호\n'
 '- 가 필요한 상태)는 각 신체부위별 판정기준에 따라 평가한다.\n'
 '- 5) 장해진단서에는 ① 장해진단명 및 발생시기 ② 장해의 내용과 그 정도 ③ 사'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000836',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
