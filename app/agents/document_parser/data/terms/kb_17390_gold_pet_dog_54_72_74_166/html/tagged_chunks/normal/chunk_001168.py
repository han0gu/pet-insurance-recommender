from langchain_core.documents import Document

chunk = Document(
    page_content=(". 이 경우 보상한도액과 자</p><br><p id='179' data-category='paragraph' "
 "style='font-size:14px'>기부담금은 각각 보험증권에 기재된 금액을 말합니다.<br>1. 제4조(보상하는 손해의 범위) "
 "제1호의 손해배상금 : 보상한도액을 한도로 보</p><br><p id='180' data-category='list' "
 "style='font-size:14px'>상하되, 자기부담금이 약정된 경우에는 그 자기부담금을 초과한 부분만 보상<br>합니다.<br>2"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001168',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
