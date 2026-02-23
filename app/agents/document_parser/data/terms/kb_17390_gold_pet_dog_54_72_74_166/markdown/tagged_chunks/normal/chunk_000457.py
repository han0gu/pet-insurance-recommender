from langchain_core.documents import Document

chunk = Document(
    page_content=('3. 피보험자 본인 또는 배우자와 생계를 같이 하고, 보험증권에 기재된 주택의 주\n'
 '민등록상 동거중인 동거 친족(민법 제 777조)\n'
 '4. 피보험자 본인 또는 배우자와 생계를 같이하는 별거 중인 미혼자녀\n'
 '관 련 법 규 민법\n'
 '∙ 제777조(친족의 범위)에서 규정한 친족의 범위# 8촌 이내의 혈족, 4촌 이내의 인척, 배우자# 제4조(보험금의 청구)- '
 '\uf000 보험수익자는 다음의 서류를 제출하고 보험금을 청구하여야 합니다.\n'
 '- 1. 청구서(회사 양식)\n'
 '- 2. 국가동물 등록한 경우에는 동물등록증 또는 등록번호'),
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
 'indexing': {'chunk_id': 'chunk_000457',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
