from langchain_core.documents import Document

chunk = Document(
    page_content=('. 보험계약을 체결한 후 제4장 반려동물 관련 특별약관 반려동물(강아지) 일반조<br>항 제9조(알릴 의무 위반의 효과) 등으로 보장을 '
 "제한하는 경우</p><br><p id='115' data-category='paragraph' "
 "style='font-size:14px'>\uf000 제1항에 따라 보장을 제한하는 범위는 수의학적으로 인과관계가 있다고 입증된 "
 '경<br>우 혹은 경험통계적으로 인과관계가 유의성있게 입증된 경우 등 해당 반려동물의<br>과거 병력과 관련이 있는 특정 '
 '질병(【별표17】(반려동물(강아지) 특정 질병 분류<br>표)'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001442',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
