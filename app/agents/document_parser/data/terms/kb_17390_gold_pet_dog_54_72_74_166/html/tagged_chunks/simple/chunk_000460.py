from langchain_core.documents import Document

chunk = Document(
    page_content=("id='159' data-category='paragraph' style='font-size:16px'>회사는 보통약관 제1절 일반조항 "
 "제5조(보험금을</p><br><p id='160' data-category='paragraph' "
 "style='font-size:16px'>어느 한 가지 목적의 치료를 위한 보험금 지급사유가 발생한 때에는 보험금을 "
 "지급</p><br><p id='161' data-category='paragraph' style='font-size:16px'>지급하지 "
 '않는 사유) 및 다음 중</p><br><p'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000460',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
