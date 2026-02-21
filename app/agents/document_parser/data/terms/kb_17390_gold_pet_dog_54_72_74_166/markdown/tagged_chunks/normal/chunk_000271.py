from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 제1항에서 "의료기관"이라 함은 의료법 제3조(의료기관) 제2항에서 정한 국내의\n'
 '병원이나 의원 또는 국외의 의료관련법에서 정한 의료기관을 말합니다.|  |\n'
 '| --- |\n'
 '| 용 어 풀 이 ∙ 절단 : 특정부위를 잘라 내는 것 특 ∙ 절제 : 특정부위를 잘라 없애는 것 별 ∙ 흡인 : 주사기 등으로 '
 '빨아들이는 것 약 |\n'
 '∙ 천자 : 바늘 또는 관을 꽂아 체액․조직을 뽑아내거나 약물을 주입하는 것제5조(특별약관의 소멸)\n'
 '피보험자가 사망하였을 경우에는 이 특별약관 계약도 소멸되며 회사는 "보험료 및'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000271',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
