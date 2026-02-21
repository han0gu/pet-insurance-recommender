from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사는<br>계약자 등이 회사에 보험금을 청구하는 등 계약자에게 연락이 닿으면 제3항의 내<br>용과 90일 이내 계약자의 '
 '재가입의사가 확인되지 않는 경우 계약이 해지된다는<br>사실을 알려드립니다.<br>\uf000 제7항에 따라 계약자에게 해지된다는 사실을 '
 '알려드린 최초시점부터 90일 이내에<br>계약자의 재가입 의사가 확인되지 않는 경우 해당 시점부터 계약은 해지됩니다.<br>\uf000 '
 '제5항에 따라 보험계약이 연장된 경우 계약자는 회사에 재가입 의사를 표시할 수<br>있습니다'),
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
 'indexing': {'chunk_id': 'chunk_000927',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
