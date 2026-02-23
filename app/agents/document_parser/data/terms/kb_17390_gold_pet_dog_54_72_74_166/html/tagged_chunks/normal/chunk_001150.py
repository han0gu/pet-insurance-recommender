from langchain_core.documents import Document

chunk = Document(
    page_content=('. 피보험자와 타인간에 손해배상에 관한 약정이 있는 경우 그 약정에 의하여 가<br>회사는 대한민국 내에서 이 특별약관의 보험기간 중에 '
 '피보험자가 보험증권에 기재<br>중된 배상책임. 그러나 약정이 없었더라도 법률규정에 의하여 피보험자가 부 상<br>된 반려동물의 행위에 '
 '기인하는 우연한 사고로 인하여 타인의 신체에 피해를 입히거<br>담하게 될 배상책임은 보상합니다. 해<br>나 타인 소유의 반려동물에 '
 '손해를 입혀 법률상의 배상책임을 부담함으로써 입은 손<br>3'),
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
 'indexing': {'chunk_id': 'chunk_001150',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
