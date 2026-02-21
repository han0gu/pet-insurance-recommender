from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- |\n'
 '- 156 -별표3 외모특정상해 분류표\n'
 '\uf000 약관에 규정하는 외모특정상해로 분류되는 상병은 제9차 개정 한국표준질병․사인\n'
 '분류(KCD, 통계청 고시 제2025-299호, 2026.1.1. 시행) 중 다음에 적은 상병을 말\n'
 '하며 이후 한국표준질병․사인분류가 개정되는 경우는 개정된 기준에 따라 이 약관| 에서 보장하는 외모특정상해 해당 여부를 | 판단합니다. '
 '|\n'
 '| --- | --- |\n'
 '| 대상이 되는 항목 | 분류번호 |\n'
 '| 머리의 손상 | S00-S09 |\n'
 '| 목의 손상 | S10-S19 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000957',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
