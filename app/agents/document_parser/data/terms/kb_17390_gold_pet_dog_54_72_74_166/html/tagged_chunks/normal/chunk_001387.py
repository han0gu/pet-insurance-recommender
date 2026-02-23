from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>반</p><br><p id='30' data-category='paragraph' "
 "style='font-size:16px'>납입최</p><br><p id='31' data-category='list' "
 "style='font-size:16px'>려<br>고(독촉)와 계약의 해지)에 정한 납입최고(독촉)기간 내에 갱신 전 보장계약의 "
 '보<br>동<br>험료를 납입 완료하고, 제2조(보장특약의 자동갱신)에 의해 보장특약이 자동 갱신 물<br>된 경우에는 갱신보장특약의 '
 '제1회 보험료를 갱신 일까지'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001387',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
