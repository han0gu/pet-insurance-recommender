from langchain_core.documents import Document

chunk = Document(
    page_content=('정하는 바에 따라 반려동물 사망 당시 이 특별약 병<br>관의 계약자적립액 및 미경과보험료를 계약자에게 지급합니다.</p><br><p '
 "id='130' data-category='list' style='font-size:14px'>1"),
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
 'indexing': {'chunk_id': 'chunk_001126',
              'chunk_char_len': 129,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
