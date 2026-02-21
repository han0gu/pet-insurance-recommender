from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약자의 전자문서 수신이 확인되기 전까지는 그 전자문서는 송신<br>되지 않은 것으로 봅니다. 회사는 전자문서가 수신되지 않은 것을 '
 '확인한 경우에<br>는 서면(등기우편 등)으로 다시 알려드립니다.<br>\uf000 제4항에도 불구하고 손해가 제1항 제1호 및 제2호의 '
 "사실로 생긴 것이 아님을 계</p><br><p id='19' data-category='list' "
 "style='font-size:14px'>약을 해지할 수 없습니다.<br>1. 회사가 최초 계약 체결 당시에 그 사실을 알았거나 과실로 "
 '알지 못하였을 때<br>2'),
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
 'indexing': {'chunk_id': 'chunk_001221',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
