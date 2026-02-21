from langchain_core.documents import Document

chunk = Document(
    page_content=("-</p><br><p id='115' data-category='paragraph' style='font-size:16px'>KB "
 "금쪽같은 펫보험(강아지)(무배당)(26.01) 163</p><p id='0' data-category='paragraph' "
 "style='font-size:14px'>\uf000 약관에 규정하는 특정정신질환으로 분류되는 질병은 제9차 개정 "
 '한국표준질병․사<br>인분류(KCD, 통계청 고시 제2025-299호, 2026.1.1'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001800',
              'chunk_char_len': 249,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
