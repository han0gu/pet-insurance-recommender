from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- |\n'
 '\uf000 진단 당시의 한국표준질병․사인분류에 따라 이 약관에서 보장하는 질병에 대한 보\n'
 '험금 지급여부가 판단된 경우, 이후 한국표준질병․사인분류 개정으로 질병분류가\n'
 '변경되더라도 이 약관에서 보장하는 질병 해당 여부를 다시 판단하지 않습니다.|  | 별표13 천식지속상태 분류표 |\n'
 '| --- | --- |\n'
 '\uf000 약관에 규정하는 천식지속상태로 분류되는 질병은 제9차 개정 한국표준질병․사인분류(KCD, 통계청 고시 제2025-299호, '
 '2026.1.1. 시행) 중 다음에 적은 질병을 말'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001009',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
