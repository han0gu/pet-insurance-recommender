from langchain_core.documents import Document

chunk = Document(
    page_content=('"<br>해<br>소득세법 시행규칙 별지 제38호 서식에 의한 장애인증명서의 원본 또는 사본"(이하<br>및<br>"장애인증명서"라 '
 '합니다)을 제출하여 제1조(적용범위) 제1항 제2호에서 정한 조<br>질<br>건에 해당함을 회사에 알려야 합니다'),
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
 'indexing': {'chunk_id': 'chunk_001424',
              'chunk_char_len': 137,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
