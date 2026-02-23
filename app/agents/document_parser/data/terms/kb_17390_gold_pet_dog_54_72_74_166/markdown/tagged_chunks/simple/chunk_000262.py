from langchain_core.documents import Document

chunk = Document(
    page_content=('76 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)# 약의 소멸) 및 제36조(중도인출)는 제외합니다.# 4. '
 '상해수술비제1조(보험금의 지급사유)\n'
 '회사는 피보험자가 이 특별약관의 보험기간 중에 상해의 직접결과로써 수술을 받은\n'
 '경우 이 특별약관의 보험가입금액을 상해수술비로 보험수익자에게 매 사고시마다 지# 급합니다.- 제2조(보험금 지급에 관한 세부규정)\n'
 '- \uf000 제1조(보험금의 지급사유)의 상해수술비는 같은 상해를 직접적인 원인으로 두 종\n'
 '- 류 이상의 상해수술을 받거나 같은 종류의 수술을 2회 이상 받은 경우에는 하나의'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000262',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
