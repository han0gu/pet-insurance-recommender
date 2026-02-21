from langchain_core.documents import Document

chunk = Document(
    page_content=("분류되는 질병은 제9차 개정 한국표준질병․사인</p><br><p id='95' data-category='paragraph' "
 "style='font-size:14px'>분류(KCD, 통계청 고시 제2025-299호, 2026.1.1"),
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
 'indexing': {'chunk_id': 'chunk_001775',
              'chunk_char_len': 128,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
