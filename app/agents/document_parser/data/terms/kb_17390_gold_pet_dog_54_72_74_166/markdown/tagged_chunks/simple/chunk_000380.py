from langchain_core.documents import Document

chunk = Document(
    page_content=('회사는 피보험자가 이 특별약관의 보험기간 중 질병으로 사망한 경우에는 이 특별약\n'
 '관의 보험가입금액 전액을 반려동물양육자금Ⅱ(질병사망)으로 보험수익자에게 지급# 합니다.- 제2조(보험금 지급에 관한 세부규정)\n'
 '- \uf000 "호스피스·완화의료 및 임종과정에 있는 환자의 연명의료 결정에 관한 법률"에\n'
 '- 따른 연명의료중단등결정 및 그 이행으로 피보험자가 사망하는 경우 연명의료중\n'
 '- 단등결정 및 그 이행은 제1조(보험금의 지급사유) "사망"의 원인 및 "사망보험금\n'
 '- "지급에 영향을 미치지 않습니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000380',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
