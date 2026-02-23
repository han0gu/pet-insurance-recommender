from langchain_core.documents import Document

chunk = Document(
    page_content=('# 제5조(보험금의 지급절차)# \uf000 회사는 제4조(보험금의청구)에서 정한 서류를 접수한 때에는 접수증을 드리고- 102 '
 '-휴대전화 문자메시지 또는 전자우편 등으로도 송부하며, 그 서류를 접수한 날부\n'
 '터 3영업일 이내에 보험금을 지급합니다.\uf000 회사가 보험금 지급사유를 조사․확인하기 위해 필요한 기간이 제1항의 지급기일을\n'
 '초과할 것이 명백히 예상되는 경우에는 그 구체적인 사유와 지급예정일 및 보험\n'
 '금 가지급제도(회사가 추정하는 보험금의 50% 이내를 지급)에 대하여 피보험자'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000469',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
