from langchain_core.documents import Document

chunk = Document(
    page_content=('하며 이후 한국표준질병․사인분류가 개정되는 경우는 개정된 기준에 따라 이 약관\n'
 '에서 보장하는 천식지속상태 해당 여부를 판단합니다.\n'
 '대상이 되는 항목 분류번호| 천식지속상태 | J46 |\n'
 '| --- | --- |\n'
 '| 주) 1. 대상질병 분류표의 분류번호와 다르나 | 한국표준질병․사인분류의 기준에 따라 |\n'
 '분류번호를 동시에 부여가 가능한 경우 대상질병 분류에 포함합니다.\n'
 '2. 제10차 개정 이후 이 약관에서 보장하는 천식지속상태 해당여부는 피보험자가'),
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
 'indexing': {'chunk_id': 'chunk_001010',
              'chunk_char_len': 251,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
