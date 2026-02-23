from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>- 99 -</p><h1 id='101' style='font-size:20px'>제4장 "
 "반려동물 관련 특별약관</h1><h1 id='102' style='font-size:18px'>반려동물(강아지) 일반조항</h1><h1 "
 "id='103' style='font-size:14px'>제1조(목적)</h1><br><p id='104' "
 "data-category='paragraph' style='font-size:14px'>이 특별약관은 계약자와 회사 사이에 보험증권에 "
 '기재된 반려동물의 상해 또는'),
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
 'indexing': {'chunk_id': 'chunk_000753',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
