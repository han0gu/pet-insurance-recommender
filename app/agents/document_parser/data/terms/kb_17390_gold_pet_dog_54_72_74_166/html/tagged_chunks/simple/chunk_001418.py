from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제1호 및 제2호 외에 항시 치료를 요하는 중증환자 ∙ 소득세법 시행규칙 제54조(장애아동의 범위) 영 제107조제1항제1호에서 '
 '"기획재정부령으로 정하는 사람"이란 "장애아동 복지지원법" 제21조제1항에 따른 발달재활서비스를 지원받고 있는 사람을 '
 "말한다.</td></tr></tbody></table><br><table id='86' "
 "style='font-size:16px'><thead></thead><tbody><tr><td><table><thead></thead><tbody><tr><td>예</td></tr><tr><td>시"),
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
 'indexing': {'chunk_id': 'chunk_001418',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
