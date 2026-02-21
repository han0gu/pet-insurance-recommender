from langchain_core.documents import Document

chunk = Document(
    page_content=('제4조(보상하는 손해의 범위) 제1호 및 제2호 "다"목 또는 "라"목의 비용에</p><br><h1 id=\'169\' '
 "style='font-size:14px'>대하여</h1><br><p id='170' data-category='paragraph' "
 "style='font-size:14px'>보상한도액을 한도로 보상하여 드립니다.</p><p id='171' "
 "data-category='paragraph' style='font-size:14px'>제7조(보험금의 청구)</p><br><p "
 "id='172' data-category='list'"),
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
 'indexing': {'chunk_id': 'chunk_001160',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
