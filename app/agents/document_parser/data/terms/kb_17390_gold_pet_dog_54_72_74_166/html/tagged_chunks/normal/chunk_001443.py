from langchain_core.documents import Document

chunk = Document(
    page_content=('병력과 관련이 있는 특정 질병(【별표17】(반려동물(강아지) 특정 질병 분류<br>표) 참조)으로 제한하여 적용하며, 그 판단기준은 '
 '회사에서 정한 계약사정기준(계<br>약인수지침 등)을 따릅니다'),
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
 'indexing': {'chunk_id': 'chunk_001443',
              'chunk_char_len': 108,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
