from langchain_core.documents import Document

chunk = Document(
    page_content=(". 이륜자동차 운전중 상해 부담보</h1><br><p id='183' data-category='list' "
 "style='font-size:14px'>제1조(계약의 체결 및 효력)<br>\uf000 이 특별약관은 보험계약(보통약관 및 다른 "
 '특별약관이 부가된 경우에는 그 특별약<br>관도 포함합니다, 이하 "보험계약"이라 합니다)을 체결할 때 계약자의 청약과 회<br>사의 '
 '승낙으로 보험계약에 부가하여 이루어집니다.<br>\uf000 이 특별약관의 효력발생일은 보통약관 제1절 일반조항 제25조(제1회 보험료 '
 '및 회<br>사의 보장개시)에서 정한'),
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
 'indexing': {'chunk_id': 'chunk_001317',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
