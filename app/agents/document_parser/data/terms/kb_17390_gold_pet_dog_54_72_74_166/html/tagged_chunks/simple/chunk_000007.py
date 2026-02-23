from langchain_core.documents import Document

chunk = Document(
    page_content=('. 시행)를 말하며 이후 한국표준질병․ 사인분류가 개정되는 경우는 개정된 기준에 따라 이 약관에 서 보장하는 질병(상병) 해당 여부를 '
 '판단합니다. • 대상 질병(상병) 분류표의 분류번호와 다르나 한국표준 질병․사인분류의 기준에 따라 분류번호를 동시에 부여가 가능한 경우 '
 '대상 질병(상병) 분류에 포함합니다. • 제10차 개정 이후 이 약관에서 보장하는 질병(상병) 해당 여부는 피보험자가 진단된 당시 '
 '시행되고 있는 한국표준 질병․사인분류에 따라 판단합니다'),
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
 'indexing': {'chunk_id': 'chunk_000007',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
