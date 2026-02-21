from langchain_core.documents import Document

chunk = Document(
    page_content=('. "국가유공자 등 예우 및 지원에 관한 법률"에 의한 상이자 및 이와 유 사한 사람으로서 근로능력이 없는 사람 3. 제1호 및 제2호 '
 '외에 항시 치료를 요하는 중증환자 ∙ 소득세법 시행규칙 제54조(장애아동의 범위) 영 제107조제1항제1호에서 "기획재정부령으로 정하는 '
 '사람"이란 "장애아동 복지지원법" 제21조제1항에 따른 발달재활서비스를 지원받고 있는 사람을 말한다. |'),
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
 'indexing': {'chunk_id': 'chunk_000803',
              'chunk_char_len': 210,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
