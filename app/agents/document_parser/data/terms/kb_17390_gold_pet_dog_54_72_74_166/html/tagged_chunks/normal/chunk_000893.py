from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 때 회사는 해지 전 발생한 보험금 지급사유를 이유로 부활<br>(효력회복)을 거절하지 않습니다.<br>\uf000 제1항에서 정한 '
 '특별약관의 부활(효력회복)이 이루어진 경우라도 계약자 또는 피<br>보험자가 최초계약 청약시(2회 이상 부활이 이루어진 경우 종전 모든 '
 '부활 청약<br>포함) 제7조(계약 전 알릴 의무)를 위반한 경우에는 제9조(알릴 의무 위반의 효<br>과)가 적용됩니다.<br>용 어 '
 "풀 이 부활</p><br><table id='66'"),
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
 'indexing': {'chunk_id': 'chunk_000893',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
