from langchain_core.documents import Document

chunk = Document(
    page_content=('보통약관 제1절 일반조항 제25조(제1회 보험료 및 회<br>사의 보장개시)에서 정한 보장개시일과 동일합니다.<br>\uf000 '
 '보험계약이 해지, 기타사유에 의하여 효력을 가지지 않게 된 경우에는 이 특별약<br>관도 더 이상 효력을 가지지 '
 '않습니다.<br>\uf000 이 특별약관은 피보험자가 이륜자동차를 소유, 사용(직업, 직무 또는 동호회 활동<br>과 출퇴근용도 등으로 '
 "주로 사용하는 경우에 한하며 일회적인 사용은 제외), 관리<br>하는 경우에 한하여 부가하여 이루어집니다.</p><p id='184'"),
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
 'indexing': {'chunk_id': 'chunk_001318',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
