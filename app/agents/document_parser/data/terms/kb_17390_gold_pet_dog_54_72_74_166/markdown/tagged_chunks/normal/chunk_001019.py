from langchain_core.documents import Document

chunk = Document(
    page_content=('| 폐 | [B05.2+: 폐렴이 합병된 홍역(J17.1*)] | B05.2+ |\n'
 '| 폐 | 거대세포바이러스폐렴(J17.1*)] | B25.0+ |\n'
 '| 폐 | [B25.0+: [B58.3+: 폐 톡소포자충증(J17.3*)] | B58.3+ |\n'
 '| 폐 | 상세불명 병원체의 폐렴 | J18 |\n'
 '| 구분 |  |  |\n'
 '| --- | --- | --- |\n'
 '| 외부요인에 의한 폐질환 | 대상이 되는 질병 | 분류번호 |\n'
 '| 외부요인에 의한 폐질환 | 탄광부 진폐증 | J60 공 |'),
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
 'indexing': {'chunk_id': 'chunk_001019',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
