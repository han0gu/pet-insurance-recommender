from langchain_core.documents import Document

chunk = Document(
    page_content=('만기에 환급되는 금액이 납입보험료를 초과하지 아니하는 보험으로서 보험계\n'
 '약 또는 보험료납입영수증에 보험료 공제대상임이 표시된 보험의 보험료를 말\n'
 '한다.# 2. 모든 피보험자 또는 모든 보험수익자가 소득세법 시행령 제107조(장애인의 범\n'
 '위) 제1항에서 규정한 장애인인 보험| 관 련 법 규 | 소득세법 |\n'
 '| --- | --- |'),
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
 'indexing': {'chunk_id': 'chunk_000800',
              'chunk_char_len': 184,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
