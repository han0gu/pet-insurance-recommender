from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>\uf000 회사는 제1조(보험금의</p><br><p id='77' "
 "data-category='list' style='font-size:14px'>지급한 경우에는 그 지급사유가 발생한 때부터 이 특별약관 "
 '계약은 소멸되며 이<br>도<br>특별약관의 해약환급금을 지급하지 않습니다.<br>성<br>\uf000 보험증권에 기재된 반려동물이 '
 '보험기간 중에 이 특별약관에서 보장하지 않는 사<br>특<br>유로 사망하였을 경우 회사는 "보험료 및 해약환급금 산출방법서"에서 정하는 '
 "바 약</p><br><p id='78'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001094',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
