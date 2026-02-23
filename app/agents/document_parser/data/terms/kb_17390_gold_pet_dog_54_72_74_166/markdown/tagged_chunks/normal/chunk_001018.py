from langchain_core.documents import Document

chunk = Document(
    page_content=('| 폐 | 기관지염 급성 세기관지염 | J21 |\n'
 '| 폐 | 달리 분류되지 않은 바이러스폐렴 폐렴연쇄알균에 의한 폐렴 | J12 |\n'
 '| 폐 | 인플루엔자균에 의한 폐렴 | J13 J14 |\n'
 '| 폐 | 달리 분류되지 않은 세균성 폐렴 | J15 |\n'
 '| 폐 | 달리 분류되지 않은 기타 감염성 병원체에 의한 폐렴 | J16 |\n'
 '| 폐 | 렴 | J17 |\n'
 '| 폐 | 달리 분류된 질환에서의 폐렴 [B01.2+: 수두폐렴(J17.1*)] | B01.2+ |\n'
 '| 폐 | [B05.2+: 폐렴이 합병된 홍역(J17.1*)] | B05.2+ |'),
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
 'indexing': {'chunk_id': 'chunk_001018',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
