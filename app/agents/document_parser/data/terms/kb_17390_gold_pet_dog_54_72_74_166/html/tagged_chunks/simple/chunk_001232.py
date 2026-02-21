from langchain_core.documents import Document

chunk = Document(
    page_content=("무효)</p><br><h1 id='32' style='font-size:16px'>계약을 맺을 때에</h1><br><p id='33' "
 "data-category='paragraph' style='font-size:16px'>보험의 목적에 이미 사고가 발생하였을 경우 이 "
 "특별약관은 무효</p><br><p id='34' data-category='paragraph' "
 "style='font-size:16px'>로 하며 이미 납입한 이 특별약관의 보험료를 돌려 드립니다"),
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
 'indexing': {'chunk_id': 'chunk_001232',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
