from langchain_core.documents import Document

chunk = Document(
    page_content=("따라 손</td></tr></tbody></table><br><table id='118' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>해를</td><td>보상합니다.</td></tr><tr><td "
 'colspan="2">부 가 설 명 음식물 반려동물이 일상 생활 중 보호자 또는 생산자의 의도와 상관 없이 섭취할 수 있는 모든 식이 '
 '원료와 가공품 및 부산물(뼈, 과일 씨 등 폐기 대상물질)을 말 하며, 사람 및 다른 동물의 식이로 활용될 수 있는 모든 것을 포함합니다'),
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
 'indexing': {'chunk_id': 'chunk_000765',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
