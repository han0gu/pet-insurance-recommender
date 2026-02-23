from langchain_core.documents import Document

chunk = Document(
    page_content=('가. 국가- 나.「한국은행법」에 따른 한국은행\n'
 '- 다. 대통령령으로 정하는 금융회사\n'
 '- 라. 「자본시장과 금융투자업에 관한 법률」 제9조제15항제3호에 따른 주권\n'
 '- 상장법인(투자성 상품 중 대통령령으로 정하는 금융상품계약체결등\n'
 '- 을 할 때에는 전문금융소비자와 같은 대우를 받겠다는 의사를 금융상\n'
 '- 품판매업자등에게 서면으로 통지하는 경우만 해당한다)\n'
 '- 마. 그 밖에 금융상품의 유형별로 대통령령으로 정하는 자\n'
 '- 향후 관련법령이 개정된 경우 개정된 내용을 적용합니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000107',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
