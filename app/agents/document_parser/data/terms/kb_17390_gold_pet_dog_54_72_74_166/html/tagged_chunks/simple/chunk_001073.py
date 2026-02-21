from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우 부활<br>(효력회복)일을 보험계약일로 하여 제1조(보험금의 지급사유) 제6항 내지 제7항을<br>적용합니다.</p><h1 '
 "id='49' style='font-size:14px'>제9조(준용규정)</h1><br><p id='50' "
 "data-category='paragraph' style='font-size:14px'>\uf000 이 특별약관에서 정하지 않은 사항은 "
 '반려동물(강아지) 일반조항을 따릅니다.<br>\uf000 반려동물(강아지) 일반조항에서 정하지 않은 사항은 보통약관 제1절 '
 '일반조항을<br>따릅니다'),
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
 'indexing': {'chunk_id': 'chunk_001073',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
